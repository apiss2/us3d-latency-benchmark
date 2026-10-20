import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from us3d_latency.conditions import missing_indices, paired_arrays, prepare
from us3d_latency.contracts import read_json, write_json, load_dataset, load_volume, relative_file
from us3d_latency.evaluation import binary_iou, center_statistics, score_volume, resample_world
from us3d_latency.runner import reconstruct, evaluate
from us3d_latency.summary import summarize

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = read_json(ROOT / "tests" / "fixtures" / "protocol_toy.json")


def toy_config():
    config = copy.deepcopy(read_json(ROOT / "benchmark.json"))
    config.update(paths=["straight"], speeds_mm_s=[5], seeds=[])
    config["image"] = {"shape_yx": [1, 1], "spacing_xy_mm": [1, 1], "origin_xy_pixels": [0, 0]}
    config["evaluation"].update(voxel_mm=1, roi_xyz_mm=[[-1, 1], [19, 21], [-1, 3]], cylinder_radius_mm=1, cylinder_length_mm=2)
    return config


def toy_dataset(root):
    files = {}
    for name in ["images", "poses", "image_times_s", "pose_times_s"]:
        files[name] = name + ".npy"
        np.save(root / files[name], np.array(FIXTURE[name], dtype=np.uint8 if name == "images" else float))
    write_json(root / "dataset.json", {"format_version": 1, "units": "mm", "fps": 30, "scans": [{"id": "straight_v5", "path": "straight", "speed_mm_s": 5, **files}]})
    return root / "dataset.json"


class BenchmarkTests(unittest.TestCase):
    def test_delay_direction_keeps_images_and_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = load_dataset(toy_dataset(root), toy_config())[0]
            pair = paired_arrays(scan, 6, np.array([], dtype=int))
            np.testing.assert_array_equal(pair["tracking_frame_index"], [6, 7])
            np.testing.assert_array_equal(pair["image_frame_index"], [0, 1])
            np.testing.assert_array_equal(pair["image_times_s"], FIXTURE["image_times_s"])
            np.testing.assert_allclose(pair["pose_times_s"] - pair["image_times_s"], 0.2)
            np.testing.assert_allclose(pair["poses"][:, 2, 3], [0, 1 / 6])
            missing = paired_arrays(scan, 6, np.array([6]))
            np.testing.assert_array_equal(missing["pose_valid"], [False, True])
            self.assertEqual(len(missing["poses"]), 2)
            self.assertTrue((missing["poses"][0] == 0).all())
            np.testing.assert_array_equal(missing["image_times_s"], pair["image_times_s"])
            for array in scan["arrays"].values():
                array._mmap.close()

    def test_missing_mask_shared_across_delays_and_no_compaction(self):
        dropped = missing_indices(360, 6, 5, 1001)
        np.testing.assert_array_equal(dropped, [12, 185, 220, 294, 321])
        for delay in range(7):
            tracking = np.arange(360) + delay
            self.assertEqual(np.count_nonzero(np.isin(tracking, dropped)), 5)
            self.assertTrue((dropped - delay > 0).all())
            self.assertTrue((dropped - delay < 359).all())
        with self.assertRaises(ValueError):
            missing_indices(10, 6, 5, 1001)

    def test_world_transform_direction(self):
        volume = np.array(FIXTURE["volume"], dtype=np.uint8)
        transform = np.array(FIXTURE["index_to_world_mm"], dtype=float)
        axes = [np.array([-.5, .5]), np.array([19.5, 20.5]), np.array([.5, 1.5])]
        np.testing.assert_array_equal(resample_world(volume, transform, axes), volume)
        transform[0, 3] += 1
        moved = resample_world(volume, transform, axes)
        self.assertTrue((moved[:, :, 0] == 0).all())
        self.assertTrue((moved[:, :, 1] == 255).all())

    def test_xy_rmse_bias_variation_and_detection_are_separate(self):
        settings = toy_config()["evaluation"]
        axes = [np.array([-.5, .5]), np.array([19.5, 20.5]), np.array([.5, 1.5])]
        binary = np.array([[[0,1],[0,1]], [[1,0],[1,0]]], dtype=bool)
        metrics, _ = center_statistics(binary, axes, settings)
        self.assertAlmostEqual(metrics["bias_norm_mm"], 0)
        self.assertAlmostEqual(metrics["center_rmse_mm"], .5)
        self.assertAlmostEqual(metrics["center_variation_mm"], .5)
        binary[1] = False
        metrics, _ = center_statistics(binary, axes, settings)
        self.assertEqual(metrics["detection_rate"], .5)
        self.assertAlmostEqual(metrics["center_rmse_mm"], .5)
        self.assertEqual(metrics["center_variation_mm"], 0)
        binary[0] = [[0,1],[0,0]]
        metrics, _ = center_statistics(binary, axes, settings)
        self.assertAlmostEqual(metrics["center_rmse_mm"], np.sqrt(.5))

    def test_iou_and_empty_detection(self):
        self.assertEqual(binary_iou(np.array([1,1,0], bool), np.array([0,1,1], bool)), 1 / 3)
        volume = np.array(FIXTURE["volume"], dtype=np.uint8)
        transform = np.array(FIXTURE["index_to_world_mm"], dtype=float)
        metrics, _ = score_volume(volume, transform, toy_config()["evaluation"])
        self.assertEqual(metrics["iou"], 1)
        self.assertEqual(metrics["translated_iou"], 1)
        volume[:] = 0
        metrics, _ = score_volume(volume, transform, toy_config()["evaluation"])
        self.assertEqual(metrics["iou"], 0)
        self.assertEqual(metrics["detection_rate"], 0)
        self.assertIsNone(metrics["center_rmse_mm"])
        self.assertIsNone(metrics["translated_iou"])
        self.assertIsNone(metrics["centroid_z_mm"])

    def test_foreground_only_outside_truth_length_still_has_corrected_iou(self):
        settings = toy_config()["evaluation"]
        settings["roi_xyz_mm"][2] = [-1, 5]
        transform = np.array(FIXTURE["index_to_world_mm"], dtype=float)
        transform[2, 3] += 2
        metrics, _ = score_volume(np.array(FIXTURE["volume"], dtype=np.uint8), transform, settings)
        self.assertEqual(metrics["iou"], 0)
        self.assertEqual(metrics["detection_rate"], 0)
        self.assertEqual(metrics["undetected_section_rate"], 1)
        for name in ["bias_x_mm", "bias_y_mm", "bias_norm_mm", "center_rmse_mm", "center_variation_mm", "center_max_mm"]:
            self.assertIsNone(metrics[name])
        self.assertEqual(metrics["centroid_z_mm"], 3)
        self.assertEqual(metrics["translated_iou"], 1)

    def test_roi_overflow_and_path_escape_are_rejected(self):
        transform = np.array(FIXTURE["index_to_world_mm"], dtype=float)
        transform[0, 3] += 10
        with self.assertRaises(ValueError):
            score_volume(np.array(FIXTURE["volume"], dtype=np.uint8), transform, toy_config()["evaluation"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                relative_file(Path(directory), "../outside.npy")

    def test_adapter_pipeline_and_resume_use_hand_authored_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = toy_dataset(root)
            run = root / "run"
            self.assertEqual(len(prepare(dataset, toy_config(), run)), 7)
            command = [sys.executable, str(ROOT / "tests" / "fixture_adapter.py")]
            subprocess.run([sys.executable, "-m", "us3d_latency", "run", str(run), "--", *command], cwd=ROOT, check=True)
            first = run / "straight_v5_k0_clean" / "reconstruction.npz"
            before = first.stat().st_mtime_ns
            reconstruct(run, command)
            self.assertEqual(before, first.stat().st_mtime_ns)
            evaluate(run)
            summarize(run, root / "summary")
            self.assertTrue(read_json(root / "summary" / "summary.json")["full_benchmark_complete"])
            self.assertEqual(len(list((root / "summary").glob("*.png"))), 4)
            _, _, metadata = load_volume(first)
            self.assertEqual(metadata["method"], "hand-authored-test-fixture")

    def test_bad_clock_and_changed_inputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = toy_dataset(root)
            np.save(root / "image_times_s.npy", np.array([0, 1]))
            with self.assertRaises(ValueError):
                load_dataset(dataset, toy_config())
            dataset = toy_dataset(root)
            run = root / "run"
            prepare(dataset, toy_config(), run)
            np.save(root / "images.npy", np.array([[[1]], [[2]]], dtype=np.uint8))
            with self.assertRaises(ValueError):
                reconstruct(run, [sys.executable, str(ROOT / "tests" / "fixture_adapter.py")])

    def test_failed_adapter_output_cannot_be_promoted_by_a_no_output_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            prepare(toy_dataset(root), toy_config(), run)
            folder = run / "straight_v5_k0_clean"
            output = folder / "reconstruction.npz"
            attempts = []

            def write_fixture(command):
                request = read_json(command[command.index("--request") + 1])
                target = Path(command[command.index("--output") + 1])
                attempts.append(target)
                self.assertFalse(target.exists())
                metadata = {"case_id": request["case_id"], "method": "fixture", "missing_pose_policy": "fixture"}
                np.savez_compressed(target, volume=np.array(FIXTURE["volume"], dtype=np.uint8), index_to_world_mm=np.array(FIXTURE["index_to_world_mm"], dtype=float), metadata_json=np.array(json.dumps(metadata)))
                return target

            def fail_after_writing(command, **kwargs):
                write_fixture(command)
                raise subprocess.CalledProcessError(7, command)

            with patch("us3d_latency.runner.subprocess.run", side_effect=fail_after_writing):
                with self.assertRaises(subprocess.CalledProcessError):
                    reconstruct(run, ["test-adapter"])
            self.assertFalse(attempts[0].exists())
            self.assertFalse(output.exists())
            self.assertFalse((run / ".adapter.lock").exists())

            # 旧形式のvalid pendingが残っていても、今回の出力として使ってはいけない。
            legacy = folder / "reconstruction.pending.npz"
            metadata = {"case_id": "straight_v5_k0_clean", "method": "fixture", "missing_pose_policy": "fixture"}
            np.savez_compressed(legacy, volume=np.array(FIXTURE["volume"], dtype=np.uint8), index_to_world_mm=np.array(FIXTURE["index_to_world_mm"], dtype=float), metadata_json=np.array(json.dumps(metadata)))
            legacy_bytes = legacy.read_bytes()

            def succeed_without_output(command, **kwargs):
                attempts.append(Path(command[command.index("--output") + 1]))
                return subprocess.CompletedProcess(command, 0)

            with patch("us3d_latency.runner.subprocess.run", side_effect=succeed_without_output):
                with self.assertRaises(FileNotFoundError):
                    reconstruct(run, ["test-adapter"])
            self.assertNotEqual(attempts[0], attempts[1])
            self.assertFalse(output.exists())
            self.assertEqual(legacy.read_bytes(), legacy_bytes)
            self.assertFalse((run / ".adapter.lock").exists())

            def succeed_with_output(command, **kwargs):
                write_fixture(command)
                return subprocess.CompletedProcess(command, 0)

            with patch("us3d_latency.runner.subprocess.run", side_effect=succeed_with_output):
                reconstruct(run, ["test-adapter"])
            completed_bytes = output.read_bytes()
            with patch("us3d_latency.runner.subprocess.run") as adapter:
                reconstruct(run, ["test-adapter"])
                adapter.assert_not_called()
            self.assertEqual(output.read_bytes(), completed_bytes)
            self.assertFalse(list(folder.glob(".reconstruction-*")))

    def test_output_dtype_and_left_handed_grid_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.npz"
            transform = np.array(FIXTURE["index_to_world_mm"], dtype=float)
            metadata = np.array(json.dumps({"case_id": "toy", "method": "fixture", "missing_pose_policy": "fixture"}))
            np.savez_compressed(output, volume=np.array(FIXTURE["volume"], dtype=np.float32), index_to_world_mm=transform, metadata_json=metadata)
            with self.assertRaises(ValueError):
                load_volume(output)
            transform[0, 0] = -1
            np.savez_compressed(output, volume=np.array(FIXTURE["volume"], dtype=np.uint8), index_to_world_mm=transform, metadata_json=metadata)
            with self.assertRaises(ValueError):
                load_volume(output)


if __name__ == "__main__":
    unittest.main()
