"""Validate explicit array and coordinate contracts without exporting scene data."""
import json
import re
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_file(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), "Array paths must be relative to the dataset manifest")
    target = (root / name).resolve()
    require(target.is_relative_to(root.resolve()), "Array path escapes the dataset directory")
    require(target.is_file(), f"Array file does not exist: {name}")
    return target


def validate_config(config):
    require(config["format_version"] == 1, "Unsupported benchmark format")
    require(config["fps"] > 0, "FPS must be positive")
    require(config["delays_frames"] == list(range(max(config["delays_frames"]) + 1)), "Delays must start at zero and be consecutive integers")
    require(len(set(config["seeds"])) == len(config["seeds"]), "Seeds must be unique")
    require(config["missing_pose_count"] >= 0, "Missing count cannot be negative")
    e = config["evaluation"]
    require(e["voxel_mm"] > 0 and e["cylinder_radius_mm"] > 0 and e["cylinder_length_mm"] > 0, "Physical dimensions must be positive")
    roi = np.asarray(e["roi_xyz_mm"], dtype=float)
    require(roi.shape == (3, 2) and np.isfinite(roi).all() and (roi[:, 1] > roi[:, 0]).all(), "Invalid fixed ROI")
    require(0 < e["threshold"] <= 255, "Threshold must lie in (0,255]")


def rigid(poses):
    require(poses.ndim == 3 and poses.shape[1:] == (4, 4) and np.isfinite(poses).all(), "Poses must be finite M x 4 x 4")
    require(np.allclose(poses[:, 3], [0, 0, 0, 1]), "Invalid homogeneous row")
    rotation = poses[:, :3, :3]
    require(np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-5) and np.allclose(np.linalg.det(rotation), 1, atol=1e-5), "Poses must be right-handed rigid transforms")


def load_dataset(manifest_path, config):
    validate_config(config)
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    require(manifest["format_version"] == 1 and manifest["units"] == "mm", "Dataset format=1 and units=mm are required")
    require(manifest["fps"] == config["fps"], "Dataset FPS differs from benchmark")
    scans = []
    seen = set()
    for item in manifest["scans"]:
        scan_id = item["id"]
        require(re.fullmatch(r"[a-z][a-z0-9_]*", scan_id) and scan_id not in seen, "Scan IDs must be unique safe identifiers")
        seen.add(scan_id)
        require(item["path"] in config["paths"] and item["speed_mm_s"] in config["speeds_mm_s"], "Unknown path or speed")
        files = {key: relative_file(manifest_path.parent, item[key]) for key in ["images", "image_times_s", "poses", "pose_times_s"]}
        arrays = {key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in files.items()}
        images, poses = arrays["images"], arrays["poses"]
        n = len(images)
        require(n >= 2, "At least two image frames are required")
        require(images.dtype == np.uint8 and images.shape == (n, *config["image"]["shape_yx"]), "Images must be uint8 N x H x W")
        rigid(poses)
        require(len(poses) >= n + max(config["delays_frames"]), "Insufficient future pose padding for maximum delay")
        require(np.allclose(poses[:, :3, :3], np.eye(3), atol=1e-6), "This benchmark requires fixed identity orientation")
        require(np.allclose(poses[:, 1, 3], 0, atol=1e-5) and (np.diff(poses[:, 2, 3]) > 0).all(), "Probe depth must remain zero and world Z must increase")
        times = []
        for key, count in [("image_times_s", n), ("pose_times_s", len(poses))]:
            values = arrays[key]
            require(values.shape == (count,) and np.isfinite(values).all(), "Invalid timestamp array")
            require(np.allclose(values - values[0], np.arange(count) / config["fps"], atol=1e-7, rtol=0), "Timestamps must be regularly sampled at configured FPS")
            times.append(values)
        require(abs(times[0][0] - times[1][0]) < 1e-7, "Image and pose frame zero must share the same clock origin")
        scans.append({**item, "files": files, "arrays": arrays, "frames": n})
    require(bool(scans), "Dataset has no scans")
    pairs = [(s["path"], s["speed_mm_s"]) for s in scans]
    require(len(set(pairs)) == len(pairs), "Each path/speed pair may appear only once")
    return scans


def load_volume(path):
    with np.load(path, allow_pickle=False) as data:
        volume, transform = data["volume"].copy(), data["index_to_world_mm"].copy()
        metadata = json.loads(str(data["metadata_json"].item()))
    require(all(isinstance(metadata.get(key), str) and metadata[key] for key in ["case_id", "method", "missing_pose_policy"]), "Output metadata must identify the case, method and missing-pose policy")
    require(volume.ndim == 3 and volume.size > 0 and volume.dtype == np.uint8, "Reconstruction must be uint8 [z,y,x]")
    require(transform.shape == (4, 4) and np.isfinite(transform).all() and np.allclose(transform[3], [0, 0, 0, 1]), "Invalid index_to_world_mm")
    basis = transform[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    require((spacing > 0).all() and np.linalg.det(basis) > 0, "Output grid must be right-handed and nonsingular")
    rotation = basis / spacing
    require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5), "Output grid axes must be orthogonal")
    return volume, transform, metadata
