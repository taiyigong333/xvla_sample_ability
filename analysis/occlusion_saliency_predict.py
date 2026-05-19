from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import json_numpy
import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Mapping:
    origin: np.ndarray
    r_new_to_base: np.ndarray

    def transform_xyz(self, xyz_base: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz_base, dtype=np.float64)
        if xyz.ndim == 1:
            return self.r_new_to_base.T @ (xyz - self.origin)
        return (self.r_new_to_base.T @ (xyz - self.origin).T).T


def load_mapping(path: Path) -> Mapping:
    payload = json.loads(path.read_text(encoding="utf-8"))
    definition = payload.get("definition", payload)
    origin = np.asarray(definition["origin_base_xyz_m"], dtype=np.float64)
    r_new_to_base = np.column_stack(
        [
            np.asarray(definition["x_new_in_base"], dtype=np.float64),
            np.asarray(definition["y_new_in_base"], dtype=np.float64),
            np.asarray(definition["z_new_in_base"], dtype=np.float64),
        ]
    )
    return Mapping(origin=origin, r_new_to_base=r_new_to_base)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_stats(values) -> dict[str, float | int]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def resolve_action_layout(lora_path: Path, requested_layout: str) -> str:
    if requested_layout != "auto":
        return requested_layout
    candidates = [
        lora_path / "xvla_training_metadata.json",
        lora_path.parent / "runnings" / lora_path.name / "xvla_training_metadata.json",
        lora_path.parent / "runnings" / "ckpt-5000" / "xvla_training_metadata.json",
    ]
    for metadata_path in candidates:
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        real_dim = int(metadata.get("inferred_action_dim", metadata.get("model_config_overrides", {}).get("real_action_dim", 0)))
        if real_dim == 7:
            return "ur7e_7d"
    return "ee6d_20d"


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64).reshape(3, 1))
    return matrix


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    return rotvec.reshape(3)


def rotate6d_to_rotmat(v6: np.ndarray) -> np.ndarray:
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-9)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def parse_action(action: np.ndarray, action_layout: str) -> tuple[np.ndarray, np.ndarray, float]:
    action = np.asarray(action, dtype=np.float64)
    if action_layout == "ur7e_7d":
        if action.shape[-1] < 7:
            raise ValueError(f"ur7e_7d action requires at least 7 dims, got {action.shape[-1]}")
        return action[:3].copy(), action[3:6].copy(), float(action[6])
    if action_layout != "ee6d_20d":
        raise ValueError(f"Unknown action layout: {action_layout}")
    left = action[:10]
    return left[:3].copy(), matrix_to_rotvec(rotate6d_to_rotmat(left[3:9])), float(left[9])


def tcp_to_proprio(tcp_pose: np.ndarray, gripper: float) -> np.ndarray:
    pos = np.asarray(tcp_pose[:3], dtype=np.float64)
    rotmat = rotvec_to_matrix(np.asarray(tcp_pose[3:6], dtype=np.float64))
    rot6d = rotmat[:, :2].flatten()
    left = np.concatenate([pos, rot6d, [float(gripper)]])
    right = np.zeros(10, dtype=np.float64)
    return np.concatenate([left, right]).astype(np.float32)


def action_chunk_to_tcp_and_gripper(actions: np.ndarray, action_layout: str, eval_steps: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    gripper = []
    for action in actions[:eval_steps]:
        pos, rotvec, grip = parse_action(action, action_layout)
        rows.append(np.concatenate([pos, rotvec]))
        gripper.append(grip)
    return np.asarray(rows, dtype=np.float64), np.asarray(gripper, dtype=np.float64)


def first_close_idx(gripper: np.ndarray, threshold: float) -> int | None:
    idx = np.flatnonzero((gripper[:-1] > threshold) & (gripper[1:] <= threshold)) + 1
    if idx.size:
        return int(idx[0])
    closed = np.flatnonzero(gripper <= threshold)
    return int(closed[0]) if closed.size else None


def choose_context_frames(tcp: np.ndarray, gripper: np.ndarray, args: argparse.Namespace, mapping: Mapping | None) -> list[int]:
    max_start = max(0, len(tcp) - args.eval_steps - 1)
    if args.context_policy == "sampled":
        frames = list(range(0, max_start + 1, max(1, args.context_stride)))
        if args.contexts_per_episode > 0 and len(frames) > args.contexts_per_episode:
            pick = np.linspace(0, len(frames) - 1, args.contexts_per_episode, dtype=int)
            frames = [frames[i] for i in pick]
        return [int(v) for v in frames]

    if args.context_policy == "around_close":
        close = first_close_idx(gripper, args.gripper_threshold)
        center = int(close) if close is not None else max_start
        offsets = [int(v.strip()) for v in args.context_offsets.split(",") if v.strip()]
        if not offsets:
            offsets = [-args.pre_event_frames, 0, args.post_event_frames]
        frames = sorted({int(np.clip(center + offset, 0, max_start)) for offset in offsets})
        return frames

    if args.context_policy == "min_z":
        xyz = mapping.transform_xyz(tcp[:, :3]) if mapping is not None else tcp[:, :3]
        center = int(np.argmin(xyz[:, 2]))
    else:
        close = first_close_idx(gripper, args.gripper_threshold)
        center = int(close) if close is not None else max_start

    if args.contexts_per_episode <= 1:
        return [int(np.clip(center - args.pre_event_frames, 0, max_start))]

    offsets = np.linspace(args.pre_event_frames * args.contexts_per_episode, args.pre_event_frames, args.contexts_per_episode, dtype=int)
    frames = sorted({int(np.clip(center - int(offset), 0, max_start)) for offset in offsets})
    return frames


def predict_actions(
    server_url: str,
    image_main: np.ndarray,
    image_wrist: np.ndarray,
    proprio: np.ndarray,
    instruction: str,
    domain_id: int,
    denoise_steps: int,
    seed: int | None,
    timeout_s: float,
) -> np.ndarray:
    payload = {
        "image0": json_numpy.dumps(image_main),
        "image1": json_numpy.dumps(image_wrist),
        "proprio": json_numpy.dumps(proprio),
        "language_instruction": instruction,
        "domain_id": domain_id,
        "steps": denoise_steps,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    response = requests.post(server_url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return np.asarray(data["action"], dtype=np.float32)


def patch_bounds(height: int, width: int, grid_size: int, row: int, col: int) -> tuple[int, int, int, int]:
    y0 = int(round(row * height / grid_size))
    y1 = int(round((row + 1) * height / grid_size))
    x0 = int(round(col * width / grid_size))
    x1 = int(round((col + 1) * width / grid_size))
    return x0, y0, x1, y1


def occlude_image(image: np.ndarray, bounds: tuple[int, int, int, int], mode: str) -> np.ndarray:
    out = image.copy()
    x0, y0, x1, y1 = bounds
    if mode == "black":
        value = np.zeros(3, dtype=np.float64)
    elif mode == "gray":
        value = np.full(3, 127.0, dtype=np.float64)
    elif mode == "mean":
        value = image.reshape(-1, image.shape[-1]).mean(axis=0)
    else:
        raise ValueError(mode)
    out[y0:y1, x0:x1] = np.clip(value, 0, 255).astype(out.dtype)
    return out


def compute_metrics(actions: np.ndarray, action_layout: str, eval_steps: int, gripper_threshold: float, mapping: Mapping) -> dict:
    tcp, gripper = action_chunk_to_tcp_and_gripper(actions, action_layout, eval_steps)
    xyz_base = tcp[:, :3]
    xyz_new = mapping.transform_xyz(xyz_base)
    endpoint_new = xyz_new[-1]
    min_z_idx = int(np.argmin(xyz_new[:, 2]))
    close_idx = first_close_idx(gripper, gripper_threshold)
    return {
        "action_dim": int(actions.shape[-1]),
        "evaluated_steps": int(len(tcp)),
        "min_z_new_m": float(xyz_new[min_z_idx, 2]),
        "min_z_new_step": min_z_idx,
        "endpoint_x_new_m": float(endpoint_new[0]),
        "endpoint_y_new_m": float(endpoint_new[1]),
        "endpoint_z_new_m": float(endpoint_new[2]),
        "min_gripper": float(gripper.min()),
        "mean_gripper": float(gripper.mean()),
        "final_gripper": float(gripper[-1]),
        "close_exists": bool(close_idx is not None),
        "first_close_step": int(close_idx) if close_idx is not None else "",
    }


def compare_metrics(base: dict, current: dict, eval_steps: int) -> dict:
    base_endpoint = np.asarray([base["endpoint_x_new_m"], base["endpoint_y_new_m"], base["endpoint_z_new_m"]], dtype=np.float64)
    cur_endpoint = np.asarray([current["endpoint_x_new_m"], current["endpoint_y_new_m"], current["endpoint_z_new_m"]], dtype=np.float64)
    base_close = base["first_close_step"]
    cur_close = current["first_close_step"]
    base_has_close = bool(base["close_exists"])
    cur_has_close = bool(current["close_exists"])
    if base_has_close and cur_has_close:
        close_step_delta = int(cur_close) - int(base_close)
        close_change_score = abs(close_step_delta) / max(1, eval_steps)
    elif base_has_close != cur_has_close:
        close_step_delta = ""
        close_change_score = 1.0
    else:
        close_step_delta = ""
        close_change_score = 0.0
    return {
        "delta_min_z_new_m": float(current["min_z_new_m"] - base["min_z_new_m"]),
        "abs_delta_min_z_new_m": float(abs(current["min_z_new_m"] - base["min_z_new_m"])),
        "delta_endpoint_z_new_m": float(current["endpoint_z_new_m"] - base["endpoint_z_new_m"]),
        "endpoint_xyz_new_delta_m": float(np.linalg.norm(cur_endpoint - base_endpoint)),
        "delta_min_gripper": float(current["min_gripper"] - base["min_gripper"]),
        "delta_mean_gripper": float(current["mean_gripper"] - base["mean_gripper"]),
        "close_exists_changed": bool(base_has_close != cur_has_close),
        "close_step_delta": close_step_delta,
        "close_change_score": float(close_change_score),
    }


def prefixed(prefix: str, row: dict) -> dict:
    return {f"{prefix}{key}": value for key, value in row.items()}


def make_heatmaps(rows: list[dict], image_lookup: dict[tuple[str, str, int, str], np.ndarray], out_dir: Path, grid_size: int) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_rows = [row for row in rows if row.get("occlusion_scope") == "grid"]
    metrics = [
        ("abs_delta_min_z_new_m", "abs min z_new change (mm)", 1000.0),
        ("endpoint_xyz_new_delta_m", "endpoint mapped XYZ change (mm)", 1000.0),
        ("delta_endpoint_z_new_m", "endpoint z_new change (mm)", 1000.0),
        ("close_change_score", "gripper close change score", 1.0),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}
    groups = sorted({(row["episode"], row["dataset"], int(row["context_frame"]), row["view"]) for row in grid_rows})
    for episode, dataset, context_frame, view in groups:
        image = image_lookup[(dataset, episode, context_frame, view)]
        group_rows = [
            row
            for row in grid_rows
            if row["episode"] == episode and row["dataset"] == dataset and int(row["context_frame"]) == context_frame and row["view"] == view
        ]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.8), dpi=150, constrained_layout=True)
        if len(metrics) == 1:
            axes = [axes]
        for ax, (metric_key, title, scale) in zip(axes, metrics):
            grid = np.full((grid_size, grid_size), np.nan, dtype=np.float64)
            for row in group_rows:
                grid[int(row["grid_row"]), int(row["grid_col"])] = float(row[metric_key]) * scale
            ax.imshow(image)
            heat = ax.imshow(grid, cmap="magma", alpha=0.52, extent=(0, image.shape[1], image.shape[0], 0))
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{dataset}/{episode} frame {context_frame} {view} occlusion saliency")
        path = out_dir / f"{dataset}_{episode}_frame_{context_frame}_{view}_heatmaps.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths[f"{dataset}/{episode}/{context_frame}/{view}"] = str(path)
    return plot_paths


def make_full_cover_plots(rows: list[dict], out_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full_rows = [row for row in rows if row.get("occlusion_scope") == "full_view"]
    if not full_rows:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}
    metrics = [
        ("abs_delta_min_z_new_m", "abs min z_new change (mm)", 1000.0),
        ("delta_endpoint_z_new_m", "endpoint z_new change (mm)", 1000.0),
        ("endpoint_xyz_new_delta_m", "endpoint mapped XYZ change (mm)", 1000.0),
        ("close_change_score", "close change score", 1.0),
    ]
    groups = sorted({(row["episode"], row["dataset"], int(row["context_frame"])) for row in full_rows})
    for episode, dataset, context_frame in groups:
        group_rows = [
            row
            for row in full_rows
            if row["episode"] == episode and row["dataset"] == dataset and int(row["context_frame"]) == context_frame
        ]
        views = [str(row["view"]) for row in group_rows]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.4), dpi=150, constrained_layout=True)
        if len(metrics) == 1:
            axes = [axes]
        for ax, (key, title, scale) in zip(axes, metrics):
            values = [float(row[key]) * scale for row in group_rows]
            ax.bar(views, values, color=["tab:blue" if view == "main" else "tab:green" for view in views])
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle(f"{dataset}/{episode} frame {context_frame} full-view occlusion")
        path = out_dir / f"{dataset}_{episode}_frame_{context_frame}_full_view_comparison.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths[f"{dataset}/{episode}/{context_frame}/full_view"] = str(path)
    return plot_paths


def episode_files(root: Path, start: int, limit: int) -> list[Path]:
    files = sorted(root.glob("episode_*.npz"))
    if start > 0:
        files = files[start:]
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files under {root}")
    return files


def ping_server(url: str) -> str:
    try:
        response = requests.post(url, json={}, timeout=3)
        return f"reachable_status_{response.status_code}"
    except Exception as exc:
        return f"unreachable: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch occlusion saliency for XVLA trajectory predictions.")
    parser.add_argument("--demos", type=Path, default=Path("datasets/raw_demos_old_plus_three_new"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/occlusion_saliency_old_plus_three_new"))
    parser.add_argument("--mapping", type=Path, default=Path("raw_demos/episode_0024_unpacked/episode_0024_motion_axes_mapping.json"))
    parser.add_argument("--lora", type=Path, default=Path("lora/ckpt-5000"))
    parser.add_argument("--server-url", default="http://127.0.0.1:8020/act")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--domain-id", type=int, default=12)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345, help="Fixed request seed for deterministic occlusion deltas. Use -1 to disable.")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--limit-episodes", type=int, default=1, help="Default is one episode for an affordable first run. Use 0 for all.")
    parser.add_argument("--contexts-per-episode", type=int, default=1)
    parser.add_argument("--context-policy", choices=("pre_close", "around_close", "min_z", "sampled"), default="pre_close")
    parser.add_argument("--context-offsets", default="-30,0,30", help="Comma-separated frame offsets for --context-policy around_close.")
    parser.add_argument("--pre-event-frames", type=int, default=90)
    parser.add_argument("--post-event-frames", type=int, default=30)
    parser.add_argument("--context-stride", type=int, default=90)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--views", nargs="+", choices=("main", "wrist"), default=["main", "wrist"])
    parser.add_argument("--skip-full-cover", action="store_true", help="Disable full-image occlusion for each selected view.")
    parser.add_argument("--occlusion", choices=("mean", "gray", "black"), default="mean")
    parser.add_argument("--action-layout", choices=("auto", "ur7e_7d", "ee6d_20d"), default="auto")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--dry-run", action="store_true", help="Only write the planned contexts and request count; do not call the model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.demos = args.demos.resolve()
    args.out_dir = args.out_dir.resolve()
    args.mapping = args.mapping.resolve()
    args.lora = args.lora.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.mapping.is_file():
        raise FileNotFoundError(f"Mapping file is required for mapped-coordinate saliency: {args.mapping}")
    mapping = load_mapping(args.mapping)
    action_layout = resolve_action_layout(args.lora, args.action_layout)
    request_seed = None if args.seed < 0 else int(args.seed)
    files = episode_files(args.demos, args.episode_start, args.limit_episodes)

    baseline_rows: list[dict] = []
    saliency_rows: list[dict] = []
    image_lookup: dict[tuple[str, str, int, str], np.ndarray] = {}
    plan_rows: list[dict] = []

    for ep_idx, npz_path in enumerate(files):
        with np.load(npz_path, allow_pickle=True) as data:
            required = ["images", "images_wrist", "tcp_poses", "gripper"]
            missing = [key for key in required if key not in data.files]
            if missing:
                raise ValueError(f"{npz_path} missing required fields: {missing}")
            images = data["images"]
            images_wrist = data["images_wrist"]
            tcp = data["tcp_poses"].astype(np.float64)
            gripper = data["gripper"].astype(np.float64).reshape(-1)
            instruction = str(data["instruction"].item()) if "instruction" in data.files else ""

            context_frames = choose_context_frames(tcp, gripper, args, mapping)
            for context_frame in context_frames:
                full_cover_count = 0 if args.skip_full_cover else len(args.views)
                request_count = 1 + len(args.views) * args.grid_size * args.grid_size + full_cover_count
                plan_rows.append(
                    {
                        "dataset": args.demos.name,
                        "episode": npz_path.stem,
                        "episode_index": ep_idx,
                        "context_frame": int(context_frame),
                        "views": ";".join(args.views),
                        "grid_size": int(args.grid_size),
                        "full_cover_views": "" if args.skip_full_cover else ";".join(args.views),
                        "planned_requests": int(request_count),
                        "instruction": args.instruction or instruction,
                    }
                )

    plan_path = args.out_dir / "occlusion_plan.csv"
    write_csv(plan_path, plan_rows)
    if args.dry_run:
        summary = {
            "demos": str(args.demos),
            "episodes": len(files),
            "episode_start": args.episode_start,
            "contexts": len(plan_rows),
            "planned_requests": int(sum(row["planned_requests"] for row in plan_rows)),
            "server_status": ping_server(args.server_url),
            "seed": request_seed,
            "plan_csv": str(plan_path),
        }
        (args.out_dir / "dry_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    for ep_idx, npz_path in enumerate(files):
        with np.load(npz_path, allow_pickle=True) as data:
            images = data["images"]
            images_wrist = data["images_wrist"]
            tcp = data["tcp_poses"].astype(np.float64)
            gripper = data["gripper"].astype(np.float64).reshape(-1)
            instruction = str(data["instruction"].item()) if "instruction" in data.files else ""
            context_frames = choose_context_frames(tcp, gripper, args, mapping)

            for context_frame in context_frames:
                context_frame = int(context_frame)
                proprio = tcp_to_proprio(tcp[context_frame], gripper[context_frame])
                image_main = images[context_frame]
                image_wrist = images_wrist[context_frame]
                current_instruction = args.instruction or instruction

                t0 = time.time()
                baseline_actions = predict_actions(
                    args.server_url,
                    image_main,
                    image_wrist,
                    proprio,
                    current_instruction,
                    args.domain_id,
                    args.denoise_steps,
                    request_seed,
                    args.request_timeout,
                )
                baseline_metrics = compute_metrics(baseline_actions, action_layout, args.eval_steps, args.gripper_threshold, mapping)
                base_row = {
                    "dataset": args.demos.name,
                    "episode": npz_path.stem,
                    "episode_index": ep_idx,
                    "context_frame": context_frame,
                    "instruction": current_instruction,
                    "action_layout": action_layout,
                    "coordinate_frame": "mapped_new",
                    "baseline_latency_s": float(time.time() - t0),
                    **baseline_metrics,
                }
                baseline_rows.append(base_row)

                view_images = {"main": image_main, "wrist": image_wrist}
                for view in args.views:
                    image_lookup[(args.demos.name, npz_path.stem, context_frame, view)] = view_images[view]
                    height, width = view_images[view].shape[:2]
                    for grid_row in range(args.grid_size):
                        for grid_col in range(args.grid_size):
                            bounds = patch_bounds(height, width, args.grid_size, grid_row, grid_col)
                            occluded_main = image_main
                            occluded_wrist = image_wrist
                            occluded = occlude_image(view_images[view], bounds, args.occlusion)
                            if view == "main":
                                occluded_main = occluded
                            else:
                                occluded_wrist = occluded
                            t0 = time.time()
                            actions = predict_actions(
                                args.server_url,
                                occluded_main,
                                occluded_wrist,
                                proprio,
                                current_instruction,
                                args.domain_id,
                                args.denoise_steps,
                                request_seed,
                                args.request_timeout,
                            )
                            metrics = compute_metrics(actions, action_layout, args.eval_steps, args.gripper_threshold, mapping)
                            delta = compare_metrics(baseline_metrics, metrics, args.eval_steps)
                            saliency_rows.append(
                                {
                                    "dataset": args.demos.name,
                                    "episode": npz_path.stem,
                                    "episode_index": ep_idx,
                                    "context_frame": context_frame,
                                    "view": view,
                                    "occlusion_scope": "grid",
                                    "grid_row": grid_row,
                                    "grid_col": grid_col,
                                    "x0": bounds[0],
                                    "y0": bounds[1],
                                    "x1": bounds[2],
                                    "y1": bounds[3],
                                    "occlusion": args.occlusion,
                                    "latency_s": float(time.time() - t0),
                                    **prefixed("baseline_", baseline_metrics),
                                    **prefixed("perturbed_", metrics),
                                    **delta,
                                }
                            )
                            print(
                                f"{npz_path.stem} frame {context_frame} {view} "
                                f"patch {grid_row},{grid_col}: "
                                f"d_min_z_new={delta['delta_min_z_new_m']*1000:.1f}mm "
                                f"d_endpoint_new={delta['endpoint_xyz_new_delta_m']*1000:.1f}mm "
                                f"close_score={delta['close_change_score']:.2f}",
                                flush=True,
                            )
                    if not args.skip_full_cover:
                        bounds = (0, 0, width, height)
                        occluded_main = image_main
                        occluded_wrist = image_wrist
                        occluded = occlude_image(view_images[view], bounds, args.occlusion)
                        if view == "main":
                            occluded_main = occluded
                        else:
                            occluded_wrist = occluded
                        t0 = time.time()
                        actions = predict_actions(
                            args.server_url,
                            occluded_main,
                            occluded_wrist,
                            proprio,
                            current_instruction,
                            args.domain_id,
                            args.denoise_steps,
                            request_seed,
                            args.request_timeout,
                        )
                        metrics = compute_metrics(actions, action_layout, args.eval_steps, args.gripper_threshold, mapping)
                        delta = compare_metrics(baseline_metrics, metrics, args.eval_steps)
                        saliency_rows.append(
                            {
                                "dataset": args.demos.name,
                                "episode": npz_path.stem,
                                "episode_index": ep_idx,
                                "context_frame": context_frame,
                                "view": view,
                                "occlusion_scope": "full_view",
                                "grid_row": "",
                                "grid_col": "",
                                "x0": bounds[0],
                                "y0": bounds[1],
                                "x1": bounds[2],
                                "y1": bounds[3],
                                "occlusion": args.occlusion,
                                "latency_s": float(time.time() - t0),
                                **prefixed("baseline_", baseline_metrics),
                                **prefixed("perturbed_", metrics),
                                **delta,
                            }
                        )
                        print(
                            f"{npz_path.stem} frame {context_frame} {view} full-cover: "
                            f"d_min_z_new={delta['delta_min_z_new_m']*1000:.1f}mm "
                            f"d_endpoint_new={delta['endpoint_xyz_new_delta_m']*1000:.1f}mm "
                            f"close_score={delta['close_change_score']:.2f}",
                            flush=True,
                        )

    baseline_csv = args.out_dir / "baseline_metrics.csv"
    saliency_csv = args.out_dir / "occlusion_saliency_metrics.csv"
    write_csv(baseline_csv, baseline_rows)
    write_csv(saliency_csv, saliency_rows)
    plots = make_heatmaps(saliency_rows, image_lookup, args.out_dir / "heatmaps", args.grid_size)
    full_cover_plots = make_full_cover_plots(saliency_rows, args.out_dir / "full_cover")
    grid_rows = [row for row in saliency_rows if row.get("occlusion_scope") == "grid"]
    full_rows = [row for row in saliency_rows if row.get("occlusion_scope") == "full_view"]
    summary = {
        "demos": str(args.demos),
        "server_url": args.server_url,
        "mapping": str(args.mapping),
        "coordinate_frame": "mapped_new",
        "primary_height_axis": "z_new",
        "episodes": len(files),
        "episode_start": args.episode_start,
        "contexts": len(baseline_rows),
        "grid_patch_predictions": len(grid_rows),
        "full_cover_predictions": len(full_rows),
        "total_perturbation_predictions": len(saliency_rows),
        "action_layout": action_layout,
        "seed": request_seed,
        "baseline_csv": str(baseline_csv),
        "saliency_csv": str(saliency_csv),
        "plan_csv": str(plan_path),
        "heatmaps": plots,
        "full_cover_plots": full_cover_plots,
        "metrics": {
            "grid_abs_delta_min_z_new_mm": finite_stats(float(row["abs_delta_min_z_new_m"]) * 1000.0 for row in grid_rows),
            "grid_endpoint_xyz_new_delta_mm": finite_stats(float(row["endpoint_xyz_new_delta_m"]) * 1000.0 for row in grid_rows),
            "full_cover_abs_delta_min_z_new_mm": finite_stats(float(row["abs_delta_min_z_new_m"]) * 1000.0 for row in full_rows),
            "full_cover_endpoint_xyz_new_delta_mm": finite_stats(float(row["endpoint_xyz_new_delta_m"]) * 1000.0 for row in full_rows),
            "close_change_score": finite_stats(float(row["close_change_score"]) for row in saliency_rows),
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
