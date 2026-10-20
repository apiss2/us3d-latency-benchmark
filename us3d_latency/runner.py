"""Invoke a user command through a file contract and retain adapter identity."""
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from .conditions import digest
from .contracts import read_json, write_json, load_volume, require
from .evaluation import score_volume


def case_directories(run):
    run = Path(run).resolve()
    index = read_json(run / "cases.json")
    for case_id in index["cases"]:
        folder = run / case_id
        require(folder.resolve().is_relative_to(run), "Case path escapes the run directory")
        yield folder


def reconstruct(run, command):
    require(bool(command), "Supply an adapter command after --")
    run = Path(run).resolve()
    lock = run / ".adapter.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        verified = {}
        environment = os.environ.copy()
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
            environment.setdefault(key, "2")
        for folder in case_directories(run):
            request = read_json(folder / "request.json")
            for value in request["inputs"].values():
                name = value["file"]
                if name not in verified:
                    verified[name] = digest(name)
                require(verified[name] == value["sha256"], "Acquisition input changed after prepare")
            output = folder / "reconstruction.npz"
            if not output.exists():
                # 呼び出し専用の空ディレクトリなら、過去の出力を成功扱いできない。
                with TemporaryDirectory(prefix=".reconstruction-", dir=folder) as directory:
                    temporary = Path(directory) / "output.npz"
                    with (folder / "adapter.log").open("w", encoding="utf-8") as log:
                        subprocess.run([*command, "--request", str(folder / "request.json"), "--output", str(temporary)], check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
                    _, _, metadata = load_volume(temporary)
                    require(metadata["case_id"] == request["case_id"], "Adapter returned a different case ID")
                    temporary.replace(output)
            _, _, metadata = load_volume(output)
            require(metadata["case_id"] == request["case_id"], "Cached output belongs to another case")
            print(request["case_id"], "ready", flush=True)
    finally:
        os.close(descriptor)
        lock.unlink()


def evaluate(run):
    config = read_json(Path(run) / "benchmark.json")
    for folder in case_directories(run):
        request = read_json(folder / "request.json")
        volume, transform, metadata = load_volume(folder / "reconstruction.npz")
        require(metadata["case_id"] == request["case_id"], "Output case mismatch")
        metrics, lines = score_volume(volume, transform, config["evaluation"])
        metrics.update({key: request[key] for key in ["case_id", "path", "speed_mm_s", "delay_frames", "delay_ms", "seed", "image_count", "missing_count", "missing_rate"]})
        metrics.update(method=metadata["method"], missing_pose_policy=metadata["missing_pose_policy"], reconstruction_sha256=digest(folder / "reconstruction.npz"))
        write_json(folder / "metrics.json", metrics)
        np.savetxt(folder / "centerline.csv", lines, delimiter=",", header="world_z_mm,center_x_mm,center_y_mm,area_mm2,in_truth_length", comments="")
        print(request["case_id"], "scored", flush=True)
