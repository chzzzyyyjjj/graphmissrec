import argparse
import csv
import json
import math
import os
import re
import struct
import subprocess
import sys
import zlib


RESULT_PATTERN = re.compile(r"test result.*?\n(.+?)(?:\n|$)", re.IGNORECASE)
METRIC_PATTERN = re.compile(r"([a-zA-Z]+@\d+)\s*:\s*([0-9.]+)")
DIAGNOSTIC_PATTERN = re.compile(r"final modal weight diagnostics.*?:\s*(\{.*?\})", re.IGNORECASE)


def parse_test_result(log_path):
    if not log_path or not os.path.exists(log_path):
        return {}
    with open(log_path, "r", encoding="utf-8") as file:
        text = file.read()
    matches = RESULT_PATTERN.findall(text)
    if not matches:
        return {}
    metric_line = matches[-1]
    return {key.lower(): float(value) for key, value in METRIC_PATTERN.findall(metric_line)}


def parse_final_modal_weight_diagnostics(log_path):
    if not log_path or not os.path.exists(log_path):
        return {}
    with open(log_path, "r", encoding="utf-8") as file:
        text = file.read()
    matches = DIAGNOSTIC_PATTERN.findall(text)
    if not matches:
        return {}
    try:
        diagnostics = json.loads(matches[-1])
    except json.JSONDecodeError:
        return {}
    return {key: float(value) for key, value in diagnostics.items()}


def format_ratio(value):
    return f"{float(value):g}"


def parse_final_modal_weight_diagnostics_csv(dataset, missing_mode, ratio):
    path = os.path.join(
        "picture",
        "modal_weight_training",
        f"{dataset}_missing_train_{missing_mode}_{format_ratio(ratio)}",
        "final_modal_weight_diagnostics.csv",
    )
    if not os.path.exists(path):
        return {}
    with open(path, "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}
    diagnostics = {}
    for key, value in rows[-1].items():
        if key in {"dataset", "missing_mode"} or value == "":
            continue
        try:
            diagnostics[key] = float(value)
        except ValueError:
            pass
    return diagnostics


def find_new_log(before_logs):
    log_dir = os.path.join("log", "MISSRec")
    after_logs = {
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.endswith(".log")
    }
    new_logs = sorted(after_logs - before_logs, key=os.path.getmtime)
    return new_logs[-1] if new_logs else None


def run_command(command, dry_run=False, env=None):
    print(" ".join(command))
    if dry_run:
        return None
    log_dir = os.path.join("log", "MISSRec")
    os.makedirs(log_dir, exist_ok=True)
    before_logs = {
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.endswith(".log")
    }
    subprocess.run(command, check=True, env=env)
    return find_new_log(before_logs)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_png(path, width, height, pixels):
    def chunk(kind, data):
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(row) for row in pixels)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as file:
        file.write(png)


def draw_line(pixels, x0, y0, x1, y1, color):
    height = len(pixels)
    width = len(pixels[0]) // 3
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            idx = x0 * 3
            pixels[y0][idx:idx + 3] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_marker(pixels, x, y, color):
    height = len(pixels)
    width = len(pixels[0]) // 3
    for yy in range(y - 3, y + 4):
        for xx in range(x - 3, x + 4):
            if (xx - x) ** 2 + (yy - y) ** 2 <= 9 and 0 <= xx < width and 0 <= yy < height:
                idx = xx * 3
                pixels[yy][idx:idx + 3] = color


def plot_simple_png(output_dir, metric, label, rows):
    width, height = 640, 420
    left, right, top, bottom = 70, 30, 35, 60
    white = [255, 255, 255]
    black = [20, 20, 20]
    grid = [220, 220, 220]
    colors = [[31, 119, 180], [214, 39, 40], [44, 160, 44], [148, 103, 189]]
    pixels = [bytearray(white * width) for _ in range(height)]

    points_by_mode = {}
    for row in rows:
        if metric not in row:
            continue
        points_by_mode.setdefault(row["missing_mode"], []).append(
            (float(row["missing_ratio"]), float(row[metric]))
        )
    points_by_mode = {
        mode: sorted(points)
        for mode, points in points_by_mode.items()
        if points
    }
    if not points_by_mode:
        return

    all_points = [point for points in points_by_mode.values() for point in points]
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        pad = abs(min_y) * 0.1 or 0.1
        min_y -= pad
        max_y += pad
    else:
        pad = (max_y - min_y) * 0.12
        min_y -= pad
        max_y += pad

    plot_w = width - left - right
    plot_h = height - top - bottom

    def project(x, y):
        px = left + int(round((x - min_x) / (max_x - min_x) * plot_w))
        py = top + plot_h - int(round((y - min_y) / (max_y - min_y) * plot_h))
        return px, py

    for i in range(6):
        y = top + int(round(i * plot_h / 5))
        draw_line(pixels, left, y, width - right, y, grid)
    draw_line(pixels, left, top, left, top + plot_h, black)
    draw_line(pixels, left, top + plot_h, width - right, top + plot_h, black)

    for mode_idx, (mode, points) in enumerate(sorted(points_by_mode.items())):
        color = colors[mode_idx % len(colors)]
        projected = [project(x, y) for x, y in points]
        for start, end in zip(projected, projected[1:]):
            draw_line(pixels, start[0], start[1], end[0], end[1], color)
        for x, y in projected:
            draw_marker(pixels, x, y, color)

    save_png(os.path.join(output_dir, f"final_{metric}.png"), width, height, pixels)


def plot_final_modal_weight_summary(output_path, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:
        print(f"matplotlib is unavailable, using built-in PNG fallback: {err}")
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        for metric, label in [
            ("graph_adaptive_weight", "Graph Adaptive Weight"),
            ("graph_confidence", "Graph Confidence"),
            ("interest_uncertainty", "Interest Uncertainty"),
            ("text_logit_weight", "Text Logit Weight"),
            ("img_logit_weight", "Image Logit Weight"),
        ]:
            plot_simple_png(output_dir, metric, label, rows)
        return

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    modes = sorted({row["missing_mode"] for row in rows})
    metrics = [
        ("graph_adaptive_weight", "Graph Adaptive Weight"),
        ("graph_confidence", "Graph Confidence"),
        ("interest_uncertainty", "Interest Uncertainty"),
        ("text_logit_weight", "Text Logit Weight"),
        ("img_logit_weight", "Image Logit Weight"),
    ]

    for metric, label in metrics:
        if not any(metric in row for row in rows):
            continue
        plt.figure(figsize=(6, 4))
        for mode in modes:
            mode_rows = sorted(
                [row for row in rows if row["missing_mode"] == mode and metric in row],
                key=lambda item: item["missing_ratio"],
            )
            if not mode_rows:
                continue
            plt.plot(
                [row["missing_ratio"] for row in mode_rows],
                [row[metric] for row in mode_rows],
                marker="o",
                label=mode,
            )
        plt.xlabel("Training Missing Ratio")
        plt.ylabel(label)
        plt.title(f"Final {label} Under Training-time Missing")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"final_{metric}.png"), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Scientific_mm_full")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--props", default="props/MISSRec.yaml,props/finetune.yaml")
    parser.add_argument("--mode", default="transductive")
    parser.add_argument("--ratios", default="0,0.2,0.4,0.6")
    parser.add_argument("--missing-modes", default="text,img,both")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--output", default="picture/modal_missing/train_missing_results.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = []
    ratios = [float(value) for value in args.ratios.split(",") if value.strip()]
    missing_modes = [value.strip() for value in args.missing_modes.split(",") if value.strip()]

    for missing_mode in missing_modes:
        for ratio in ratios:
            note = f"missing_train_{missing_mode}_{ratio:g}"
            command = [
                sys.executable,
                "finetune.py",
                "-d",
                args.dataset,
                "-p",
                args.pretrained,
                "-props",
                args.props,
                "-mode",
                args.mode,
                "-note",
                note,
                f"--modality_missing_train_mode={missing_mode if ratio > 0 else 'none'}",
                f"--modality_missing_train_ratio={ratio}",
            ]
            env = os.environ.copy()
            if args.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = args.gpu

            print(f"\nRunning: mode={missing_mode}, ratio={ratio}")
            if args.dry_run:
                print(" ".join(command))
                log_path = None
            else:
                log_path = run_command(command, env=env)

            row = {
                "dataset": args.dataset,
                "missing_mode": missing_mode,
                "missing_ratio": ratio,
                "log_path": log_path or "",
            }
            row.update(parse_test_result(log_path))
            diagnostics = parse_final_modal_weight_diagnostics(log_path)
            if not diagnostics:
                diagnostics = parse_final_modal_weight_diagnostics_csv(args.dataset, missing_mode, ratio)
            row.update(diagnostics)
            rows.append(row)
            if not args.dry_run:
                write_csv(args.output, rows)
                plot_final_modal_weight_summary(args.output, rows)

    if args.dry_run:
        print("\nDry run only. Commands above were not executed.")
    else:
        write_csv(args.output, rows)
        plot_final_modal_weight_summary(args.output, rows)
        print(f"Saved summary: {args.output}")


if __name__ == "__main__":
    main()
