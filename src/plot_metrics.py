#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_label(path, data):
    dataset = data.get("dataset", "")
    stem = Path(path).stem
    return f"{dataset} - {stem}" if dataset else stem


def mean_first_n(values, n=3):
    vals = [v for v in values[:n] if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def save_bar_chart(labels, values, ylabel, title, output_path):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_grouped_bar_chart(group_names, series_labels, series_values, ylabel, title, output_path):
    plt.figure(figsize=(12, 7))
    x = np.arange(len(group_names))
    num_series = len(series_labels)
    width = 0.8 / max(num_series, 1)

    for idx, (label, values) in enumerate(zip(series_labels, series_values)):
        offset = (idx - (num_series - 1) / 2.0) * width
        plt.bar(x + offset, values, width=width, label=label)

    plt.xticks(x, group_names, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Gera gráficos a partir dos arquivos *_metrics.json.")
    parser.add_argument("--metrics", nargs="+", required=True, help="Lista de arquivos *_metrics.json")
    parser.add_argument("--labels", nargs="*", default=None, help="Rótulos opcionais, na mesma ordem de --metrics")
    parser.add_argument("--output-dir", default="results/plots", help="Diretório de saída dos gráficos")
    parser.add_argument("--actor-count-for-average", type=int, default=3, help="Nº de atores usados na média dos grupos ósseos")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) not in (0, len(args.metrics)):
        raise ValueError("Se --labels for usado, ele deve ter o mesmo número de itens de --metrics.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    labels = []

    for idx, metric_path in enumerate(args.metrics):
        data = load_json(metric_path)
        datasets.append(data)
        if args.labels and len(args.labels) == len(args.metrics):
            labels.append(args.labels[idx])
        else:
            labels.append(default_label(metric_path, data))

    pcp_values = [100.0 * float(d.get("avg_pcp_first_3_actors", 0.0)) for d in datasets]
    save_bar_chart(labels, pcp_values, "PCP3D médio (%)", "Comparação de PCP3D médio", output_dir / "01_avg_pcp3d.png")

    recall_values = [100.0 * float(d.get("recall", 0.0)) for d in datasets]
    save_bar_chart(labels, recall_values, "Recall@500mm (%)", "Comparação de Recall@500mm", output_dir / "02_recall.png")

    mpjpe_values = []
    for d in datasets:
        value = d.get("nearest_prediction_mpjpe_mm_mean", None)
        mpjpe_values.append(float(value) if value is not None else np.nan)
    save_bar_chart(labels, mpjpe_values, "MPJPE (mm)", "Comparação de MPJPE médio", output_dir / "03_mpjpe.png")

    all_actor_counts = [len(d.get("actor_pcp", [])) for d in datasets]
    max_actors = max(all_actor_counts) if all_actor_counts else 0
    actor_names = [f"Actor {i+1}" for i in range(max_actors)]
    actor_series = []
    for d in datasets:
        vals = [100.0 * float(v) for v in d.get("actor_pcp", [])]
        if len(vals) < max_actors:
            vals = vals + [np.nan] * (max_actors - len(vals))
        actor_series.append(vals)

    if max_actors > 0:
        save_grouped_bar_chart(actor_names, labels, actor_series, "PCP3D (%)", "PCP3D por ator", output_dir / "04_actor_pcp.png")

    all_groups = []
    for d in datasets:
        all_groups.extend(list(d.get("bone_group_pcp", {}).keys()))
    group_names = list(dict.fromkeys(all_groups))
    group_series = []
    for d in datasets:
        group_values = []
        bone_groups = d.get("bone_group_pcp", {})
        for group in group_names:
            vals = bone_groups.get(group, [])
            avg = mean_first_n(vals, args.actor_count_for_average)
            group_values.append(100.0 * avg if avg is not None else np.nan)
        group_series.append(group_values)

    if group_names:
        save_grouped_bar_chart(
            group_names,
            labels,
            group_series,
            "PCP3D médio (%)",
            f"PCP3D por grupo ósseo (média dos primeiros {args.actor_count_for_average} atores)",
            output_dir / "05_bone_groups.png",
        )

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for label, d in zip(labels, datasets):
            f.write(f"{label}\n")
            f.write(f"  dataset: {d.get('dataset')}\n")
            f.write(f"  avg_pcp_first_3_actors: {100.0 * float(d.get('avg_pcp_first_3_actors', 0.0)):.2f}%\n")
            f.write(f"  recall: {100.0 * float(d.get('recall', 0.0)):.2f}%\n")
            mpjpe = d.get("nearest_prediction_mpjpe_mm_mean", None)
            if mpjpe is not None:
                f.write(f"  nearest_prediction_mpjpe_mm_mean: {float(mpjpe):.2f} mm\n")
            actor_pcp = d.get("actor_pcp", [])
            for i, v in enumerate(actor_pcp, start=1):
                f.write(f"  Actor {i}: {100.0 * float(v):.2f}%\n")
            f.write("\n")

    print("Script salvo em:")
    print(Path(__file__).resolve())
    print()
    print("Quando você rodar, os gráficos serão gerados em:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
