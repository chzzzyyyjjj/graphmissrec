import argparse
import csv
import json
import os
from collections import OrderedDict
from logging import getLogger

import numpy as np
import torch
from tqdm import tqdm

from data.dataset import MISSRecDataset
from missrec import MISSRec
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator
from recbole.evaluator.collector import DataStruct
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, set_color


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Scientific_mm_full by user-history-length buckets."
    )
    parser.add_argument("-d", "--dataset", default="Scientific_mm_full")
    parser.add_argument(
        "-p",
        "--checkpoint",
        required=True,
        help="Fine-tuned checkpoint path, e.g. saved/MISSRec-xxx.pth",
    )
    parser.add_argument(
        "-props",
        "--props",
        default="props/MISSRec.yaml,props/finetune.yaml",
        help="Comma-separated config files used to build the model and data.",
    )
    parser.add_argument(
        "--mode",
        default="transductive",
        choices=["transductive", "inductive"],
        help="Fine-tuning mode used by the checkpoint.",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=4,
        help="Number of quantile buckets when --boundaries is not set.",
    )
    parser.add_argument(
        "--boundaries",
        default="",
        help=(
            "Optional comma-separated upper bounds, e.g. '5,10,20'. "
            "This produces <=5, 6-10, 11-20, >20."
        ),
    )
    parser.add_argument("--output-dir", default="bucket_results")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--separate-buckets",
        action="store_true",
        help="Run model inference bucket by bucket to reduce peak GPU memory.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override eval_batch_size. Use a smaller value when GPU memory is limited.",
    )
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU evaluation.")
    parser.add_argument("--gpu-id", default=None, help="Override gpu_id in config.")
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading the checkpoint state_dict.",
    )
    return parser.parse_args()


def build_config(args):
    props = [p.strip() for p in args.props.split(",") if p.strip()]
    config_dict = {}
    if args.use_cpu:
        config_dict["use_gpu"] = False
    if args.gpu_id is not None:
        config_dict["gpu_id"] = args.gpu_id
    if args.eval_batch_size is not None:
        config_dict["eval_batch_size"] = args.eval_batch_size
    config = Config(
        model=MISSRec,
        dataset=args.dataset,
        config_file_list=props,
        config_dict=config_dict,
    )
    config["train_stage"] = args.mode + "_ft"
    config["show_progress"] = args.show_progress
    return config


def load_checkpoint(model, checkpoint_path, device, strict=False):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if "other_parameter" in checkpoint:
        model.load_other_parameter(checkpoint.get("other_parameter"))
    return missing, unexpected


def make_bucket_specs(lengths, num_buckets=4, boundaries=None):
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0:
        raise ValueError("No test users found for bucket evaluation.")

    if boundaries:
        upper_bounds = sorted({int(x.strip()) for x in boundaries.split(",") if x.strip()})
    else:
        quantiles = np.linspace(0, 1, num_buckets + 1)[1:-1]
        upper_bounds = sorted({higher_quantile(lengths, q) for q in quantiles})
        upper_bounds = [b for b in upper_bounds if lengths.min() <= b < lengths.max()]

    specs = []
    low = int(lengths.min())
    for high in upper_bounds:
        if high < low:
            continue
        specs.append((low, high, f"{low}-{high}" if low != high else str(low)))
        low = high + 1
    high = int(lengths.max())
    if low <= high:
        specs.append((low, high, f"{low}-{high}" if low != high else str(low)))
    return specs


def higher_quantile(values, q):
    try:
        return int(np.quantile(values, q, method="higher"))
    except TypeError:
        return int(np.quantile(values, q, interpolation="higher"))


def collect_full_sort_topk(config, trainer, eval_data, show_progress=False, length_range=None):
    if not isinstance(eval_data, FullSortEvalDataLoader):
        raise TypeError("This script expects full-sort evaluation. Check eval_args.mode in config.")

    trainer.model.eval()
    trainer.tot_item_num = eval_data.dataset.item_num
    if trainer.item_tensor is None:
        trainer.item_tensor = eval_data.dataset.get_item_feature().to(trainer.device)

    uid_field = config["USER_ID_FIELD"]
    len_field = config["ITEM_LIST_LENGTH_FIELD"]
    max_topk = max(config["topk"]) + 1

    rec_topk_parts = []
    uid_parts = []
    hist_len_parts = []

    iterator = tqdm(
        eval_data,
        total=len(eval_data),
        ncols=100,
        desc=set_color("BucketEval", "pink"),
    ) if show_progress else eval_data

    for batched_data in iterator:
        if length_range is not None:
            batched_data = filter_sequential_batch_by_length(config, batched_data, length_range)
            if batched_data is None:
                continue

        interaction, scores, positive_u, positive_i = trainer._full_sort_batch_eval(batched_data)
        positive_u = positive_u.to(scores.device)
        positive_i = positive_i.to(scores.device)

        _, topk_idx = torch.topk(scores, max_topk, dim=-1)
        pos_matrix = torch.zeros_like(scores, dtype=torch.int)
        pos_matrix[positive_u, positive_i] = 1
        pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
        pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
        rec_topk_parts.append(torch.cat((pos_idx, pos_len_list), dim=1).cpu())

        uid_parts.append(interaction[uid_field].cpu())
        hist_len_parts.append(interaction[len_field].cpu())

    if not rec_topk_parts:
        return {
            "rec.topk": torch.empty((0, max_topk + 2), dtype=torch.int),
            "user_id": np.array([], dtype=np.int64),
            "history_length": np.array([], dtype=np.int64),
        }

    return {
        "rec.topk": torch.cat(rec_topk_parts, dim=0),
        "user_id": torch.cat(uid_parts, dim=0).numpy(),
        "history_length": torch.cat(hist_len_parts, dim=0).numpy(),
    }


def filter_sequential_batch_by_length(config, batched_data, length_range):
    interaction, history_index, _, _ = batched_data
    if history_index is not None:
        raise TypeError("--separate-buckets currently supports sequential full-sort evaluation only.")

    low, high = length_range
    len_field = config["ITEM_LIST_LENGTH_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]
    lengths = interaction[len_field]
    mask = (lengths >= low) & (lengths <= high)
    if not bool(mask.any()):
        return None

    filtered = interaction[mask]
    inter_num = len(filtered)
    positive_u = torch.arange(inter_num)
    positive_i = filtered[iid_field]
    return filtered, None, positive_u, positive_i


def evaluate_subset(evaluator, rec_topk):
    struct = DataStruct()
    struct.set("rec.topk", rec_topk)
    return evaluator.evaluate(struct)


def evaluate_buckets(config, collected, num_buckets, boundaries):
    evaluator = Evaluator(config)
    rec_topk = collected["rec.topk"]
    user_ids = collected["user_id"]
    lengths = collected["history_length"]
    bucket_specs = make_bucket_specs(lengths, num_buckets=num_buckets, boundaries=boundaries)

    rows = []
    for low, high, label in bucket_specs:
        mask = (lengths >= low) & (lengths <= high)
        if not mask.any():
            continue
        result = evaluate_subset(evaluator, rec_topk[torch.from_numpy(mask)])
        row = OrderedDict()
        row["bucket"] = label
        row["min_history_len"] = low
        row["max_history_len"] = high
        row["user_count"] = int(len(np.unique(user_ids[mask])))
        row["sample_count"] = int(mask.sum())
        row.update(result)
        rows.append(row)

    all_result = evaluate_subset(evaluator, rec_topk)
    overall = OrderedDict()
    overall["bucket"] = "all"
    overall["min_history_len"] = int(lengths.min())
    overall["max_history_len"] = int(lengths.max())
    overall["user_count"] = int(len(np.unique(user_ids)))
    overall["sample_count"] = int(len(lengths))
    overall.update(all_result)
    return rows, overall


def evaluate_buckets_separately(config, trainer, eval_data, num_buckets, boundaries, show_progress):
    len_field = config["ITEM_LIST_LENGTH_FIELD"]
    lengths = eval_data.dataset.inter_feat[len_field].numpy()
    bucket_specs = make_bucket_specs(lengths, num_buckets=num_buckets, boundaries=boundaries)

    rows = []
    overall_parts = []
    for low, high, label in bucket_specs:
        collected = collect_full_sort_topk(
            config,
            trainer,
            eval_data,
            show_progress=show_progress,
            length_range=(low, high),
        )
        if len(collected["history_length"]) == 0:
            continue

        evaluator = Evaluator(config)
        result = evaluate_subset(evaluator, collected["rec.topk"])
        row = OrderedDict()
        row["bucket"] = label
        row["min_history_len"] = low
        row["max_history_len"] = high
        row["user_count"] = int(len(np.unique(collected["user_id"])))
        row["sample_count"] = int(len(collected["history_length"]))
        row.update(result)
        rows.append(row)
        overall_parts.append(collected)

        del collected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not overall_parts:
        raise ValueError("No test users found for bucket evaluation.")

    overall_collected = {
        "rec.topk": torch.cat([part["rec.topk"] for part in overall_parts], dim=0),
        "user_id": np.concatenate([part["user_id"] for part in overall_parts]),
        "history_length": np.concatenate([part["history_length"] for part in overall_parts]),
    }
    overall_result = evaluate_subset(Evaluator(config), overall_collected["rec.topk"])
    overall = OrderedDict()
    overall["bucket"] = "all"
    overall["min_history_len"] = int(overall_collected["history_length"].min())
    overall["max_history_len"] = int(overall_collected["history_length"].max())
    overall["user_count"] = int(len(np.unique(overall_collected["user_id"])))
    overall["sample_count"] = int(len(overall_collected["history_length"]))
    overall.update(overall_result)
    return rows, overall


def save_results(rows, overall, output_dir, dataset):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{dataset}_user_bucket_metrics.csv")
    json_path = os.path.join(output_dir, f"{dataset}_user_bucket_metrics.json")

    all_rows = rows + [overall]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with open(json_path, "w") as f:
        json.dump({"buckets": rows, "overall": overall}, f, indent=2)

    return csv_path, json_path


def print_table(rows, overall):
    all_rows = rows + [overall]
    headers = list(all_rows[0].keys())
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in all_rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in all_rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def main():
    args = parse_args()
    config = build_config(args)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = MISSRecDataset(config)
    train_data, _, test_data = data_preparation(config, dataset)
    model = MISSRec(config, train_data.dataset).to(config["device"])
    missing, unexpected = load_checkpoint(
        model, args.checkpoint, config["device"], strict=args.strict_load
    )
    if missing:
        logger.warning(f"Missing checkpoint keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected checkpoint keys: {unexpected}")

    trainer = Trainer(config, model)
    if args.separate_buckets:
        rows, overall = evaluate_buckets_separately(
            config,
            trainer,
            test_data,
            num_buckets=args.num_buckets,
            boundaries=args.boundaries,
            show_progress=args.show_progress,
        )
    else:
        collected = collect_full_sort_topk(config, trainer, test_data, args.show_progress)
        rows, overall = evaluate_buckets(
            config, collected, num_buckets=args.num_buckets, boundaries=args.boundaries
        )
    csv_path, json_path = save_results(rows, overall, args.output_dir, args.dataset)

    print_table(rows, overall)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
