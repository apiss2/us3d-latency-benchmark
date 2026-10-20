"""Score reconstruction output on one fixed world grid; do not reconstruct images."""
import numpy as np
from scipy.ndimage import map_coordinates, shift
from .contracts import require


def evaluation_grid(settings):
    step = settings["voxel_mm"]
    axes = [np.arange(lo + step / 2, hi, step) for lo, hi in settings["roi_xyz_mm"]]
    x, y, z = axes
    cx, cy = settings["cylinder_center_xy_mm"]
    truth = ((x[None, None, :] - cx) ** 2 + (y[None, :, None] - cy) ** 2 <= settings["cylinder_radius_mm"] ** 2)
    truth = truth & ((z[:, None, None] >= 0) & (z[:, None, None] < settings["cylinder_length_mm"]))
    require(truth.any(), "Truth is empty on the configured evaluation grid")
    return axes, truth


def resample_world(volume, transform, axes):
    inverse = np.linalg.inv(transform)
    x, y, z = axes
    yy, xx = np.meshgrid(y, x, indexing="ij")
    output = np.empty((len(z), len(y), len(x)), dtype=np.uint8)
    for start in range(0, len(z), 16):
        zz = z[start:start + 16, None, None]
        coordinates = [inverse[row, 0] * xx[None] + inverse[row, 1] * yy[None] + inverse[row, 2] * zz + inverse[row, 3] for row in [2, 1, 0]]
        output[start:start + len(zz)] = map_coordinates(volume, coordinates, order=1, output=np.uint8, mode="constant", cval=0, prefilter=False)
    return output


def require_contained_foreground(volume, transform, settings):
    roi = np.asarray(settings["roi_xyz_mm"])
    for z, plane in enumerate(volume):
        y, x = np.nonzero(plane >= settings["threshold"])
        if len(x):
            indices = np.column_stack((x, y, np.full(len(x), z)))
            points = indices @ transform[:3, :3].T + transform[:3, 3]
            require(((points >= roi[:, 0]) & (points < roi[:, 1])).all(), "Foreground extends beyond the fixed evaluation ROI; choose one common ROI before the experiment")


def binary_iou(first, second):
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 1.0


def center_statistics(binary, axes, settings):
    x, y, z = axes
    counts = binary.sum(axis=(1, 2))
    detected = counts > 0
    target = (z >= 0) & (z < settings["cylinder_length_mm"])
    centers = np.full((len(z), 2), np.nan)
    centers[detected, 0] = (binary.sum(axis=1) @ x)[detected] / counts[detected]
    centers[detected, 1] = (binary.sum(axis=2) @ y)[detected] / counts[detected]
    errors = centers[target & detected] - settings["cylinder_center_xy_mm"]
    metrics = {name: None for name in ["bias_x_mm", "bias_y_mm", "bias_norm_mm", "center_rmse_mm", "center_variation_mm", "center_max_mm"]}
    if len(errors):
        bias = errors.mean(axis=0)
        metrics.update(bias_x_mm=float(bias[0]), bias_y_mm=float(bias[1]), bias_norm_mm=float(np.linalg.norm(bias)),
                       center_rmse_mm=float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))),
                       center_variation_mm=float(np.sqrt(np.mean(np.sum((errors - bias) ** 2, axis=1)))),
                       center_max_mm=float(np.max(np.linalg.norm(errors, axis=1))))
    metrics.update(detected_sections=int(np.count_nonzero(target & detected)), truth_sections=int(target.sum()),
                   detection_rate=float(np.count_nonzero(target & detected) / target.sum()),
                   undetected_section_rate=float(np.count_nonzero(target & ~detected) / target.sum()))
    lines = np.column_stack((z, centers, counts * settings["voxel_mm"] ** 2, target.astype(int)))
    return metrics, lines


def score_volume(volume, transform, settings):
    require_contained_foreground(volume, transform, settings)
    axes, truth = evaluation_grid(settings)
    sampled = resample_world(volume, transform, axes)
    binary = sampled >= settings["threshold"]
    metrics, lines = center_statistics(binary, axes, settings)
    metrics.update(iou=binary_iou(binary, truth), volume_mm3=float(binary.sum() * settings["voxel_mm"] ** 3),
                   translated_iou=None, centroid_x_mm=None, centroid_y_mm=None, centroid_z_mm=None)
    indices = np.nonzero(binary)
    # 補正IoUはROI全体の前景で定義し、真値長さ内だけの中心統計とは分ける。
    if len(indices[0]):
        centroid = np.array([axes[0][indices[2]].mean(), axes[1][indices[1]].mean(), axes[2][indices[0]].mean()])
        target = np.array([*settings["cylinder_center_xy_mm"], settings["cylinder_length_mm"] / 2])
        correction = target - centroid
        corrected = shift(sampled, correction[::-1] / settings["voxel_mm"], order=1, mode="constant", cval=0, prefilter=False)
        metrics.update(translated_iou=binary_iou(corrected >= settings["threshold"], truth), centroid_x_mm=float(centroid[0]),
                       centroid_y_mm=float(centroid[1]), centroid_z_mm=float(centroid[2]))
    return metrics, lines
