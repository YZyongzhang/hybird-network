#!/usr/bin/env python3
"""
Validate offline pickle dataset files used by this project.

Checks include:
- pickle readability
- top-level structure and required keys
- trajectory length consistency
- obs fields: rgb / depth / spectrogram basic validity
"""

import argparse
import collections
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


REQUIRED_TOP_LEVEL_KEYS = ("obs", "reward", "done", "action_id", "info")
REQUIRED_OBS_KEYS = ("rgb", "depth", "spectrogram")


@dataclass
class FileReport:
    path: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    num_steps: int = 0
    top_keys: list[str] = field(default_factory=list)
    obs_keys: list[str] = field(default_factory=list)
    top_schema: dict[str, dict[str, Any]] = field(default_factory=dict)
    obs_schema: dict[str, dict[str, Any]] = field(default_factory=dict)
    stereo_checked_steps: int = 0
    stereo_ok_steps: int = 0
    unclear_audio_steps: int = 0
    sound_ids: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check pickle structure and media data quality.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing pickle files.")
    parser.add_argument(
        "--exts",
        type=str,
        default=".pkl,.pickle",
        help="Comma-separated filename extensions to include.",
    )
    parser.add_argument(
        "--include-no-ext",
        action="store_true",
        help="Also include files without extension (useful for some collected datasets).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit checked files. 0 means all files.",
    )
    parser.add_argument(
        "--max-steps-per-file",
        type=int,
        default=0,
        help="Limit checked obs steps per file. 0 means all steps.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="",
        help="Optional output json path for detailed report.",
    )
    return parser.parse_args()


def should_include(path: Path, exts: set[str], include_no_ext: bool) -> bool:
    if path.name == "a.md":
        return False
    if path.suffix.lower() in exts:
        return True
    if include_no_ext and path.suffix == "":
        return True
    return False


def discover_files(root: Path, exts: set[str], include_no_ext: bool) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and should_include(p, exts, include_no_ext):
            files.append(p)
    files.sort()
    return files


def add_error(rep: FileReport, msg: str) -> None:
    rep.ok = False
    rep.errors.append(msg)


def add_warning(rep: FileReport, msg: str) -> None:
    rep.warnings.append(msg)


def as_array(value: Any, field_name: str, rep: FileReport) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value)
    except Exception as exc:
        add_error(rep, f"{field_name}: cannot convert to numpy array ({exc})")
        return None
    if arr.size == 0:
        add_error(rep, f"{field_name}: empty array")
        return None
    if not np.issubdtype(arr.dtype, np.number):
        add_error(rep, f"{field_name}: non-numeric dtype={arr.dtype}")
        return None
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        add_error(rep, f"{field_name}: contains {bad} non-finite values (NaN/Inf)")
        return None
    return arr


def flatten_1d_sequence(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            if isinstance(item, (list, tuple, np.ndarray)):
                out.extend(flatten_1d_sequence(item))
            else:
                out.append(item)
        return out
    return [value]


def schema_of_value(value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, np.ndarray):
        out["shape"] = list(value.shape)
        out["dtype"] = str(value.dtype)
    elif isinstance(value, (list, tuple)):
        out["len"] = len(value)
        if len(value) > 0:
            out["item_type"] = type(value[0]).__name__
    elif isinstance(value, dict):
        out["keys"] = sorted([str(k) for k in value.keys()])
    return out


def check_rgb(rgb: Any, rep: FileReport, idx: int) -> None:
    arr = as_array(rgb, f"obs[{idx}].rgb", rep)
    if arr is None:
        return
    if arr.ndim != 3:
        add_error(rep, f"obs[{idx}].rgb: expected 3 dims (H,W,C), got {arr.shape}")
        return
    if arr.shape[2] not in (3, 4):
        add_error(rep, f"obs[{idx}].rgb: expected channel=3/4, got {arr.shape}")
        return
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmin < 0.0 or vmax > 255.0:
        add_warning(rep, f"obs[{idx}].rgb: value range suspicious [{vmin:.3f}, {vmax:.3f}]")
    if float(arr.std()) == 0.0:
        add_warning(rep, f"obs[{idx}].rgb: all pixels identical (std=0)")


def check_depth(depth: Any, rep: FileReport, idx: int) -> None:
    arr = as_array(depth, f"obs[{idx}].depth", rep)
    if arr is None:
        return
    if arr.ndim not in (2, 3):
        add_error(rep, f"obs[{idx}].depth: expected 2/3 dims, got {arr.shape}")
        return
    if arr.ndim == 3 and arr.shape[2] != 1:
        add_warning(rep, f"obs[{idx}].depth: expected last channel=1, got {arr.shape}")
    if float(arr.min()) < 0.0:
        add_warning(rep, f"obs[{idx}].depth: contains negative values")
    if float(arr.std()) == 0.0:
        add_warning(rep, f"obs[{idx}].depth: all pixels identical (std=0)")


def check_spectrogram(spec: Any, rep: FileReport, idx: int) -> None:
    field_name = f"obs[{idx}].spectrogram"

    def _is_waveform_shape(shape: tuple[int, ...]) -> bool:
        if len(shape) == 1 and shape[0] >= 256:
            return True
        if len(shape) == 2 and (shape[0] in (1, 2) or shape[1] in (1, 2)):
            return max(shape) >= 256
        return False

    def _has_stereo(arr: np.ndarray) -> bool:
        if arr.ndim == 1:
            return False
        if arr.ndim == 2 and (arr.shape[0] == 2 or arr.shape[1] == 2):
            return True
        if arr.ndim >= 3 and (arr.shape[0] == 2 or arr.shape[-1] == 2):
            return True
        return False

    def _clarity_check(arr: np.ndarray, label: str) -> None:
        p05 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))
        dynamic_range = p95 - p05
        near_silent_ratio = float((np.abs(arr) < 1e-6).sum()) / float(arr.size)
        std = float(arr.std())
        unclear = False
        reasons: list[str] = []
        if dynamic_range < 1e-3:
            unclear = True
            reasons.append(f"low_dynamic_range={dynamic_range:.6f}")
        if near_silent_ratio > 0.95:
            unclear = True
            reasons.append(f"near_silent_ratio={near_silent_ratio:.4f}")
        if std < 1e-6:
            unclear = True
            reasons.append(f"low_std={std:.6f}")
        if unclear:
            rep.unclear_audio_steps += 1
            add_warning(rep, f"{label}: unclear audio ({', '.join(reasons)})")
        zero_ratio = float((arr == 0).sum()) / float(arr.size)
        if zero_ratio > 0.99:
            add_warning(rep, f"{label}: >99% zeros (zero_ratio={zero_ratio:.4f})")

    def _check_waveform(arr: np.ndarray, label: str) -> None:
        if arr.ndim not in (1, 2):
            add_warning(rep, f"{label}: waveform expected ndim 1/2, got {arr.shape}")
        if arr.ndim == 2 and not (arr.shape[0] in (1, 2) or arr.shape[1] in (1, 2)):
            add_warning(rep, f"{label}: waveform has no explicit channel axis, shape={arr.shape}")
        peak = float(np.max(np.abs(arr)))
        if peak > 1e4:
            add_warning(rep, f"{label}: unusually large amplitude peak={peak:.3f}")
        _clarity_check(arr, label)

    def _check_spec(arr: np.ndarray, label: str) -> None:
        if arr.ndim < 2:
            add_warning(rep, f"{label}: spectrogram expected ndim>=2, got {arr.shape}")
        _clarity_check(arr, label)

    # 先处理你这种常见结构：list/tuple 长度=2（左右声道），每个元素可单独转 ndarray
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        rep.stereo_checked_steps += 1
        ch0 = as_array(spec[0], f"{field_name}[0]", rep)
        ch1 = as_array(spec[1], f"{field_name}[1]", rep)
        if ch0 is None or ch1 is None:
            return

        # 混合结构：一个是频谱图，一个是原始波形（例如 (65,26,2) + (2,16000)）
        is0_wave = _is_waveform_shape(ch0.shape)
        is1_wave = _is_waveform_shape(ch1.shape)
        if is0_wave != is1_wave:
            if is0_wave:
                audio_arr, spec_arr = ch0, ch1
                audio_label, spec_label = f"{field_name}.audio", f"{field_name}.spec"
            else:
                audio_arr, spec_arr = ch1, ch0
                audio_label, spec_label = f"{field_name}.audio", f"{field_name}.spec"
            if _has_stereo(ch0) or _has_stereo(ch1):
                rep.stereo_ok_steps += 1
            _check_waveform(audio_arr, audio_label)
            _check_spec(spec_arr, spec_label)
            return

        # 传统结构：两个元素都作为双声道拆分项
        rep.stereo_ok_steps += 1
        if ch0.shape != ch1.shape:
            add_warning(rep, f"{field_name}: L/R shape mismatch {ch0.shape} vs {ch1.shape}")
        _check_waveform(ch0, f"{field_name}[0]")
        _check_waveform(ch1, f"{field_name}[1]")
        return

    # 其它情况按常规 ndarray 处理
    arr = as_array(spec, field_name, rep)
    if arr is None:
        return
    if arr.ndim < 2:
        add_error(rep, f"{field_name}: expected ndim>=2, got shape={arr.shape}")
        return

    channel_axis = None
    if arr.ndim >= 3:
        if arr.shape[0] == 2:
            channel_axis = 0
        elif arr.shape[-1] == 2:
            channel_axis = arr.ndim - 1
    rep.stereo_checked_steps += 1
    if channel_axis is None:
        add_warning(rep, f"{field_name}: no clear stereo axis(2ch), shape={arr.shape}")
    else:
        rep.stereo_ok_steps += 1

    _check_spec(arr, field_name)


def collect_sound_ids(data: dict[str, Any]) -> list[str]:
    ids: set[str] = set()
    if "sound_id" in data:
        for v in flatten_1d_sequence(data["sound_id"]):
            if v is not None:
                ids.add(str(v))
    info = data.get("info")
    if isinstance(info, (list, tuple)):
        for item in info:
            if isinstance(item, dict) and "sound" in item and item["sound"] is not None:
                ids.add(str(item["sound"]))
    return sorted(ids)


def validate_one_file(path: Path, max_steps_per_file: int) -> FileReport:
    rep = FileReport(path=str(path))
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        add_error(rep, f"pickle load failed: {exc}")
        return rep

    if not isinstance(data, dict):
        add_error(rep, f"top-level must be dict, got {type(data).__name__}")
        return rep
    rep.top_keys = sorted([str(k) for k in data.keys()])
    rep.top_schema = {str(k): schema_of_value(v) for k, v in data.items()}
    rep.sound_ids = collect_sound_ids(data)

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            add_error(rep, f"missing top-level key: {key}")
    if not rep.ok:
        return rep

    obs = data["obs"]
    reward = data["reward"]
    done = data["done"]
    action_id = data["action_id"]

    if not isinstance(obs, (list, tuple)):
        add_error(rep, f"obs must be list/tuple, got {type(obs).__name__}")
        return rep
    if len(obs) == 0:
        add_error(rep, "obs is empty")
        return rep

    rep.num_steps = len(obs)

    action_flat = flatten_1d_sequence(action_id)
    reward_flat = flatten_1d_sequence(reward)
    done_flat = flatten_1d_sequence(done)

    len_action = len(action_flat)
    len_reward = len(reward_flat)
    len_done = len(done_flat)
    if not (len_action == len_reward == len_done):
        add_error(rep, f"len mismatch: action={len_action}, reward={len_reward}, done={len_done}")
    if len(obs) != len_action + 1:
        add_warning(rep, f"len(obs) != len(action)+1 ({len(obs)} vs {len_action}+1)")

    step_limit = len(obs) if max_steps_per_file <= 0 else min(len(obs), max_steps_per_file)
    for i in range(step_limit):
        oi = obs[i]
        if not isinstance(oi, dict):
            add_error(rep, f"obs[{i}] must be dict, got {type(oi).__name__}")
            continue
        if i == 0:
            rep.obs_keys = sorted([str(k) for k in oi.keys()])
            rep.obs_schema = {str(k): schema_of_value(v) for k, v in oi.items()}
        for k in REQUIRED_OBS_KEYS:
            if k not in oi:
                add_error(rep, f"obs[{i}] missing key: {k}")
        if "rgb" in oi:
            check_rgb(oi["rgb"], rep, i)
        if "depth" in oi:
            check_depth(oi["depth"], rep, i)
        if "spectrogram" in oi:
            check_spectrogram(oi["spectrogram"], rep, i)

    return rep


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Invalid root directory: {root}")

    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    files = discover_files(root, exts=exts, include_no_ext=args.include_no_ext)
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        print(f"[ERROR] no candidate files found under: {root}")
        return

    print(f"[info] root={root}")
    print(f"[info] files_to_check={len(files)}")
    print(f"[info] include_no_ext={args.include_no_ext}")

    reports: list[FileReport] = []
    for idx, fpath in enumerate(files, 1):
        rep = validate_one_file(fpath, max_steps_per_file=args.max_steps_per_file)
        reports.append(rep)
        sid_text = ",".join(rep.sound_ids[:5]) if rep.sound_ids else "N/A"
        sid_tail = "" if len(rep.sound_ids) <= 5 else f"...(+{len(rep.sound_ids)-5})"
        if not rep.ok:
            print(f"[{idx}/{len(files)}] FAIL {fpath} sound_id={sid_text}{sid_tail}")
        elif rep.warnings:
            print(
                f"[{idx}/{len(files)}] WARN {fpath} "
                f"({len(rep.warnings)} warnings) sound_id={sid_text}{sid_tail}"
            )
        else:
            print(f"[{idx}/{len(files)}] OK   {fpath} sound_id={sid_text}{sid_tail}")

    total = len(reports)
    failed = sum(1 for r in reports if not r.ok)
    warned = sum(1 for r in reports if r.ok and r.warnings)
    checked_steps = sum(r.num_steps for r in reports)
    stereo_checked_steps = sum(r.stereo_checked_steps for r in reports)
    stereo_ok_steps = sum(r.stereo_ok_steps for r in reports)
    unclear_audio_steps = sum(r.unclear_audio_steps for r in reports)

    print("\n===== SUMMARY =====")
    print(f"total_files      : {total}")
    print(f"failed_files     : {failed}")
    print(f"warning_files    : {warned}")
    print(f"total_obs_steps  : {checked_steps}")
    if stereo_checked_steps > 0:
        print(
            "stereo_ok_steps  : "
            f"{stereo_ok_steps}/{stereo_checked_steps} "
            f"({(100.0 * stereo_ok_steps / stereo_checked_steps):.2f}%)"
        )
    print(f"unclear_audio_steps: {unclear_audio_steps}")

    # Generic key/value structure summary.
    top_key_counter: collections.Counter[str] = collections.Counter()
    obs_key_counter: collections.Counter[str] = collections.Counter()
    top_schema_first: dict[str, dict[str, Any]] = {}
    obs_schema_first: dict[str, dict[str, Any]] = {}
    for r in reports:
        for k in r.top_keys:
            top_key_counter[k] += 1
            if k not in top_schema_first and k in r.top_schema:
                top_schema_first[k] = r.top_schema[k]
        for k in r.obs_keys:
            obs_key_counter[k] += 1
            if k not in obs_schema_first and k in r.obs_schema:
                obs_schema_first[k] = r.obs_schema[k]

    if top_key_counter:
        print("\n[top-level key/value structure]")
        for key, count in sorted(top_key_counter.items(), key=lambda x: (-x[1], x[0])):
            schema = top_schema_first.get(key, {})
            print(f"- {key}: seen={count}/{total}, schema={schema}")
    if obs_key_counter:
        print("\n[obs key/value structure]")
        for key, count in sorted(obs_key_counter.items(), key=lambda x: (-x[1], x[0])):
            schema = obs_schema_first.get(key, {})
            print(f"- {key}: seen={count}/{total}, schema={schema}")

    if failed:
        print("\n[failed files]")
        for r in reports:
            if r.ok:
                continue
            print(f"- {r.path}")
            for e in r.errors[:5]:
                print(f"  * {e}")

    if args.report_json:
        out = Path(args.report_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(root),
            "total_files": total,
            "failed_files": failed,
            "warning_files": warned,
            "total_obs_steps": checked_steps,
            "stereo_checked_steps": stereo_checked_steps,
            "stereo_ok_steps": stereo_ok_steps,
            "unclear_audio_steps": unclear_audio_steps,
            "reports": [
                {
                    "path": r.path,
                    "ok": r.ok,
                    "num_steps": r.num_steps,
                    "sound_ids": r.sound_ids,
                    "top_keys": r.top_keys,
                    "obs_keys": r.obs_keys,
                    "top_schema": r.top_schema,
                    "obs_schema": r.obs_schema,
                    "stereo_checked_steps": r.stereo_checked_steps,
                    "stereo_ok_steps": r.stereo_ok_steps,
                    "unclear_audio_steps": r.unclear_audio_steps,
                    "errors": r.errors,
                    "warnings": r.warnings,
                }
                for r in reports
            ],
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[info] json report saved: {out}")


if __name__ == "__main__":
    main()

# python tools/check_pickle_data.py  --root dataset-data3/dataset/pickle/offline(greedy) --report-json tmp/pickle_check_report.json
