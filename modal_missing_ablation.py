import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.dataset import MISSRecDataset
from missrec import MISSRec
from recbole.data import data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_mean_dict(rows):
    merged = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            merged[key].append(float(value))
    return {key: sum(values) / max(len(values), 1) for key, values in merged.items()}


def sample_available_ids(empty_mask, ratio, seed):
    if ratio <= 0:
        return torch.empty(0, dtype=torch.long)
    available = (~empty_mask).nonzero(as_tuple=False).view(-1)
    available = available[available != 0]
    if available.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    count = int(round(available.numel() * ratio))
    count = min(max(count, 0), available.numel())
    perm = torch.randperm(available.numel(), generator=generator)[:count]
    return available[perm]


def apply_missing_mask(model, originals, mode, ratio, seed):
    if "text" in model.modal_type:
        model.plm_embedding.weight.data.copy_(originals["text_weight"])
        model.plm_embedding_empty_mask.data.copy_(originals["text_empty_mask"])
    if "img" in model.modal_type:
        model.img_embedding.weight.data.copy_(originals["img_weight"])
        model.img_embedding_empty_mask.data.copy_(originals["img_empty_mask"])

    if mode in ["text", "both"] and "text" in model.modal_type:
        ids = sample_available_ids(originals["text_empty_mask"], ratio, seed)
        model.plm_embedding.weight.data[ids] = 0
        model.plm_embedding_empty_mask.data[ids] = True

    if mode in ["img", "both"] and "img" in model.modal_type:
        ids = sample_available_ids(originals["img_empty_mask"], ratio, seed + 9973)
        model.img_embedding.weight.data[ids] = 0
        model.img_embedding_empty_mask.data[ids] = True


@torch.no_grad()
def collect_diagnostics(model, data_loader, device, max_batches=20, topk=20):
    model.eval()
    rows = []
    for batch_idx, batched_data in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        interaction = batched_data[0] if isinstance(batched_data, tuple) else batched_data
        interaction = interaction.to(device)
        item_seq = interaction[model.ITEM_SEQ]
        item_seq_len = interaction[model.ITEM_SEQ_LEN]

        seq_output, _ = model._compute_seq_embeddings(item_seq, item_seq_len)
        diag = model.compute_fusion_diagnostics(item_seq, item_seq_len)

        if "text" in model.modal_type and "img" in model.modal_type:
            text_emb = model.text_adaptor(model.plm_embedding.weight)
            img_emb = model.img_adaptor(model.img_embedding.weight)
            diag.update(model.compute_item_fusion_weight_diagnostics(seq_output, text_emb, img_emb, topk=topk))

        diag["graph_compensation_need"] = diag.get("any_missing_rate", 0.0)
        diag["graph_compensation_signal"] = (
            diag.get("any_missing_rate", 0.0) * diag.get("graph_adaptive_weight", 0.0)
        )
        rows.append(diag)
    return tensor_mean_dict(rows)


def evaluate_model(config, model, train_data, test_data):
    trainer = Trainer(config, model)
    trainer.eval_collector.data_collect(train_data)
    return trainer.evaluate(test_data, load_best_model=False, show_progress=False)


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    modes = sorted({row["missing_mode"] for row in rows})

    for mode in modes:
        mode_rows = sorted([row for row in rows if row["missing_mode"] == mode], key=lambda x: x["missing_ratio"])
        x = [row["missing_ratio"] for row in mode_rows]

        plt.figure(figsize=(6, 4))
        for metric, label in [
            ("hit@10", "HIT@10"),
            ("ndcg@10", "NDCG@10"),
        ]:
            if metric in mode_rows[0]:
                plt.plot(x, [row[metric] for row in mode_rows], marker="o", label=label)
        plt.xlabel("Injected Missing Ratio")
        plt.ylabel("Test Performance")
        plt.title(f"Modality Missing Ablation ({mode})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"missing_ablation_{mode}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(6, 4))
        for metric, label in [
            ("text_available_rate", "Text Available"),
            ("img_available_rate", "Image Available"),
            ("graph_adaptive_weight", "Graph Adaptive Weight"),
            ("graph_compensation_signal", "Graph Compensation Signal"),
        ]:
            if metric in mode_rows[0]:
                plt.plot(x, [row[metric] for row in mode_rows], marker="o", label=label)
        plt.xlabel("Injected Missing Ratio")
        plt.ylabel("Mean Weight / Rate")
        plt.title(f"Fusion Weight Visualization ({mode})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"fusion_weights_{mode}.png"), dpi=300)
        plt.close()

    plt.figure(figsize=(6, 4))
    for mode in modes:
        mode_rows = sorted([row for row in rows if row["missing_mode"] == mode], key=lambda x: x["missing_ratio"])
        plt.plot(
            [row["missing_ratio"] for row in mode_rows],
            [row.get("graph_compensation_signal", 0.0) for row in mode_rows],
            marker="o",
            label=mode,
        )
    plt.xlabel("Injected Missing Ratio")
    plt.ylabel("Graph Compensation Signal")
    plt.title("Graph Compensation Under Modality Missing")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_compensation_summary.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/data/zyj/MM23-MISSRec/saved/MISSRec-May-26-2026_15-11-49.pth")
    parser.add_argument("--output-dir", default="picture/modal_missing")
    parser.add_argument("--ratios", default="0,0.1,0.2,0.3")
    parser.add_argument("--modes", default="text,img,both")
    parser.add_argument("--diagnostic-batches", type=int, default=20)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    config["device"] = torch.device(args.device)
    config["use_gpu"] = args.device.startswith("cuda")
    init_seed(config["seed"], config["reproducibility"])

    dataset = MISSRecDataset(config)
    train_data, _, test_data = data_preparation(config, dataset)
    model = MISSRec(config, train_data.dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.load_other_parameter(checkpoint.get("other_parameter"))

    originals = {}
    if "text" in model.modal_type:
        originals["text_weight"] = model.plm_embedding.weight.data.detach().clone()
        originals["text_empty_mask"] = model.plm_embedding_empty_mask.data.detach().clone()
    if "img" in model.modal_type:
        originals["img_weight"] = model.img_embedding.weight.data.detach().clone()
        originals["img_empty_mask"] = model.img_embedding_empty_mask.data.detach().clone()

    rows = []
    ratios = [float(value) for value in args.ratios.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]

    for mode in modes:
        for ratio in ratios:
            apply_missing_mask(model, originals, mode, ratio, args.seed)
            result = evaluate_model(config, model, train_data, test_data)
            diagnostics = collect_diagnostics(
                model,
                test_data,
                config["device"],
                max_batches=args.diagnostic_batches,
                topk=args.topk,
            )
            row = {
                "checkpoint": args.checkpoint,
                "dataset": config["dataset"],
                "missing_mode": mode,
                "missing_ratio": ratio,
            }
            row.update({key.lower(): value for key, value in result.items()})
            row.update(diagnostics)
            rows.append(row)
            print(row)

    apply_missing_mask(model, originals, "both", 0.0, args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "modal_missing_ablation.csv")
    write_csv(csv_path, rows)
    plot_results(rows, args.output_dir)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
