"""Tabulate controls, seed variation, axial shifts and world-XY center errors."""
import csv
from pathlib import Path
import numpy as np
from .contracts import read_json, write_json, require
from .runner import case_directories


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    require(not output.exists(), "Use a new summary directory")
    folders = list(case_directories(run))
    rows = [read_json(folder / "metrics.json") for folder in folders]
    require(bool(rows), "No results")
    require(len({(r["method"], r["missing_pose_policy"]) for r in rows}) == 1, "Use one reconstruction method and missing-pose policy per run")
    clean = {(r["path"], r["speed_mm_s"], r["delay_frames"]): r for r in rows if r["seed"] is None}
    index = {(r["path"], r["speed_mm_s"], r["delay_frames"], r["seed"]): r for r in rows}
    require(len(index) == len(rows), "Duplicate condition keys")
    for row in rows:
        base = clean[row["path"], row["speed_mm_s"], 0]
        same = clean[row["path"], row["speed_mm_s"], row["delay_frames"]]
        seed_zero = index[row["path"], row["speed_mm_s"], 0, row["seed"]]
        row["delta_iou_vs_clean_zero"] = row["iou"] - base["iou"]
        row["delta_iou_vs_same_delay_clean"] = row["iou"] - same["iou"]
        row["delta_iou_vs_same_seed_zero"] = row["iou"] - seed_zero["iou"]
        row["delta_centroid_z_vs_clean_zero_mm"] = None if row["centroid_z_mm"] is None or base["centroid_z_mm"] is None else row["centroid_z_mm"] - base["centroid_z_mm"]
    output.mkdir(parents=True)
    write_csv(output / "all_conditions.csv", rows)
    groups = []
    for key in sorted(clean):
        members = [r for r in rows if (r["path"], r["speed_mm_s"], r["delay_frames"]) == key and r["seed"] is not None]
        if members:
            values = np.array([r["delta_iou_vs_same_delay_clean"] for r in members])
            groups.append(dict(path=key[0], speed_mm_s=key[1], delay_frames=key[2], seeds=len(members), delta_iou_mean=float(values.mean()), delta_iou_min=float(values.min()), delta_iou_max=float(values.max())))
    if groups:
        write_csv(output / "missing_seed_summary.csv", groups)
    paths = sorted({r["path"] for r in rows})
    for field, label, filename in [("iou", "Raw world-space IoU", "iou_delay"), ("translated_iou", "Centroid-translation-corrected IoU", "corrected_iou_delay"), ("center_rmse_mm", "Detected-section world-XY center RMSE [mm]", "center_rmse")]:
        fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 4), squeeze=False)
        for column, path in enumerate(paths):
            axis = axes[0, column]
            for speed in sorted({r["speed_mm_s"] for r in rows if r["path"] == path}):
                values = sorted((r for r in clean.values() if r["path"] == path and r["speed_mm_s"] == speed), key=lambda r: r["delay_frames"])
                axis.plot([r["delay_frames"] for r in values], [np.nan if r[field] is None else r[field] for r in values], "-o", label=f"{speed} mm/s")
            axis.set(title=path, xlabel="Ultrasound delay [frame]", ylabel=label)
            axis.grid(alpha=0.25)
            axis.legend()
        fig.tight_layout()
        fig.savefig(output / f"{filename}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 4), squeeze=False)
    speed = max(r["speed_mm_s"] for r in rows)
    by_id = {read_json(folder / "request.json")["case_id"]: folder for folder in folders}
    for column, path in enumerate(paths):
        axis = axes[0, column]
        for row in sorted(clean.values(), key=lambda r: r["delay_frames"]):
            if row["path"] == path and row["speed_mm_s"] == speed:
                line = np.loadtxt(by_id[row["case_id"]] / "centerline.csv", delimiter=",", skiprows=1)
                valid = (line[:, 4] > 0) & np.isfinite(line[:, 1])
                axis.plot(line[valid, 0], line[valid, 1], label=f"k={row['delay_frames']}")
        axis.axhline(read_json(Path(run) / "benchmark.json")["evaluation"]["cylinder_center_xy_mm"][0], color="gray", linestyle="--")
        axis.set(title=f"{path}, {speed} mm/s", xlabel="World Z [mm]", ylabel="Section center world X [mm]")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "centerlines.png", dpi=180)
    plt.close(fig)
    config = read_json(Path(run) / "benchmark.json")
    expected = {(p, v, k, s) for p in config["paths"] for v in config["speeds_mm_s"] for k in config["delays_frames"] for s in [None, *config["seeds"]]}
    write_json(output / "summary.json", dict(cases=len(rows), clean=len(clean), missing_pose_cases=len(rows) - len(clean), method=rows[0]["method"], missing_pose_policy=rows[0]["missing_pose_policy"], expected_full_benchmark_cases=len(expected), full_benchmark_complete=set(index) == expected))
