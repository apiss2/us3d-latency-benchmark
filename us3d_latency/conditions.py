"""Pair existing images with delayed poses; preserve every image and timestamp."""
import hashlib
from pathlib import Path
import numpy as np
from .contracts import load_dataset, require, write_json


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def missing_indices(frame_count, maximum_delay, count, seed):
    # 共通内部区間なら、全遅延で前後の有効poseが残る。
    eligible = np.arange(maximum_delay + 1, frame_count - 1)
    require(len(eligible) >= count, "Not enough interior frames for the requested missing-pose count")
    return np.sort(np.random.Generator(np.random.PCG64(seed)).choice(eligible, count, replace=False))


def paired_arrays(scan, delay, dropped):
    n = scan["frames"]
    tracking = np.arange(n) + delay
    valid = ~np.isin(tracking, dropped)
    poses = np.array(scan["arrays"]["poses"][tracking], dtype=np.float64)
    poses[~valid] = 0
    return {"poses": poses, "pose_valid": valid, "image_frame_index": np.arange(n), "tracking_frame_index": tracking,
            "image_times_s": np.array(scan["arrays"]["image_times_s"]), "pose_times_s": np.array(scan["arrays"]["pose_times_s"][tracking])}


def prepare(dataset_path, config, output):
    scans = load_dataset(dataset_path, config)
    output = Path(output).resolve()
    require(not output.exists(), "Use a new output directory to preserve existing runs")
    output.mkdir(parents=True)
    write_json(output / "benchmark.json", config)
    cases = []
    for scan in scans:
        inputs = {name: {"file": str(path), "sha256": digest(path)} for name, path in scan["files"].items()}
        for seed in [None, *config["seeds"]]:
            dropped = np.array([], dtype=int) if seed is None else missing_indices(scan["frames"], max(config["delays_frames"]), config["missing_pose_count"], seed)
            for delay in config["delays_frames"]:
                case_id = f"{scan['id']}_k{delay}_{'clean' if seed is None else 'seed' + str(seed)}"
                folder = output / case_id
                folder.mkdir()
                np.savez_compressed(folder / "condition.npz", **paired_arrays(scan, delay, dropped))
                request = {"format_version": 1, "case_id": case_id, "path": scan["path"], "speed_mm_s": scan["speed_mm_s"], "fps": config["fps"],
                           "delay_frames": delay, "delay_ms": delay * 1000 / config["fps"], "seed": seed, "image_count": scan["frames"],
                           "missing_tracker_indices": dropped.tolist(), "missing_count": len(dropped), "missing_rate": len(dropped) / scan["frames"],
                           "inputs": inputs, "image": config["image"], "units": "mm", "condition_file": str(folder / "condition.npz"),
                           "requested_output_voxel_mm": config["evaluation"]["voxel_mm"], "output_contract": "uint8 volume[z,y,x] and index_to_world_mm[4,4]"}
                write_json(folder / "request.json", request)
                cases.append(case_id)
    write_json(output / "cases.json", {"format_version": 1, "cases": cases})
    return cases
