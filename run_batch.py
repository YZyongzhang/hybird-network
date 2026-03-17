#!/usr/bin/env python3
"""
Batch runner for OfflineRL hyper-parameter sweeps.

Design goals:
- Run OfflineRL v1 or v1_5.
- Sweep parameter combinations sequentially overnight.
- Save per-run parameter snapshot for reproducibility.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch


@dataclass
class RunResult:
    run_id: str
    status: str
    params: Dict[str, float]
    run_dir: str = ""
    error: str = ""
    train_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep OfflineRL parameters and keep machine busy overnight."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="v1",
        choices=["v1", "v1_5"],
        help="TRAIN.OFFLINE.model to sweep",
    )

    parser.add_argument("--epochs", type=int, default=80, help="TRAIN.OFFLINE.num_epochs")
    parser.add_argument(
        "--online-test-epoch",
        type=int,
        default=0,
        help="TRAIN.OFFLINE.online_test_epoch (0 means disable during training)",
    )
    parser.add_argument("--split", type=str, default="val_multiple", help="dataset split")

    # shared parameter sweeps for v1 / v1_5
    parser.add_argument("--betas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--actor-lrs", type=float, nargs="+", default=[2e-4])
    parser.add_argument("--critic-lrs", type=float, nargs="+", default=[5e-4])
    parser.add_argument("--alpha-lrs", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.99])
    parser.add_argument("--taus", type=float, nargs="+", default=[0.005])
    parser.add_argument("--target-entropies", type=float, nargs="+", default=[1.08])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[512])

    parser.add_argument(
        "--grid-json",
        type=str,
        default="",
        help=(
            "optional json file to define explicit param list. "
            "format: [{\"beta\":0.5,\"actor_lr\":0.0002,...}, ...]"
        ),
    )
    parser.add_argument("--max-runs", type=int, default=0, help="truncate run count (0 means no limit)")

    parser.add_argument("--ckpt-root", type=str, default="media/TRAIN_BATCH_V1")
    parser.add_argument("--report", type=str, default="")
    parser.add_argument("--stop-on-error", action="store_true")

    return parser.parse_args()


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _remove_checkpoints(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    removed = 0
    for p in dir_path.rglob("*.pth"):
        if p.is_file():
            p.unlink()
            removed += 1
    return removed


def _load_run_module():
    run_path = Path(__file__).resolve().parent / "run.py"
    spec = importlib.util.spec_from_file_location("run_entry", run_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load run module from: {run_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _short(v: float) -> str:
    s = f"{v:.6g}"
    return s.replace("-", "m").replace(".", "p")


def _run_id(idx: int, model: str, params: Dict[str, float]) -> str:
    model_tag = model.replace(".", "_")
    return (
        f"{idx:03d}_{model_tag}"
        f"_beta{_short(float(params['beta']))}"
        f"_alr{_short(float(params['actor_lr']))}"
        f"_clr{_short(float(params['critic_lr']))}"
        f"_ent{_short(float(params['target_entropy']))}"
    )


def _build_param_list(args: argparse.Namespace) -> List[Dict[str, float]]:
    if args.grid_json:
        with open(args.grid_json, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list) or not items:
            raise ValueError("grid-json must be a non-empty list")
        out: List[Dict[str, float]] = []
        required = {
            "beta",
            "actor_lr",
            "critic_lr",
            "alpha_lr",
            "gamma",
            "tau",
            "target_entropy",
            "batch_size",
        }
        for i, obj in enumerate(items):
            if not isinstance(obj, dict):
                raise ValueError(f"grid-json item#{i} must be object")
            missing = required - set(obj.keys())
            if missing:
                raise ValueError(f"grid-json item#{i} missing keys: {sorted(missing)}")
            out.append(
                {
                    "beta": float(obj["beta"]),
                    "actor_lr": float(obj["actor_lr"]),
                    "critic_lr": float(obj["critic_lr"]),
                    "alpha_lr": float(obj["alpha_lr"]),
                    "gamma": float(obj["gamma"]),
                    "tau": float(obj["tau"]),
                    "target_entropy": float(obj["target_entropy"]),
                    "batch_size": int(obj["batch_size"]),
                }
            )
        return out

    keys = [
        "beta",
        "actor_lr",
        "critic_lr",
        "alpha_lr",
        "gamma",
        "tau",
        "target_entropy",
        "batch_size",
    ]
    values = [
        args.betas,
        args.actor_lrs,
        args.critic_lrs,
        args.alpha_lrs,
        args.gammas,
        args.taus,
        args.target_entropies,
        args.batch_sizes,
    ]
    out: List[Dict[str, float]] = []
    for combo in itertools.product(*values):
        out.append(dict(zip(keys, combo)))
    return out


def _prepare_config_for_model(
    get_config_fn,
    model: str,
    params: Dict[str, float],
    epochs: int,
    online_test_epoch: int,
    split: str,
    run_dir: Path,
):
    cfg = get_config_fn()
    cfg.defrost()
    cfg.TASK_CONFIG.defrost()

    task_cfg = cfg.TASK_CONFIG
    task_cfg.COLLECT.OPEN = False
    task_cfg.PT.OPEN = False
    task_cfg.EVAL.OPEN = False
    task_cfg.TRAIN.OPEN = True
    task_cfg.TRAIN.TYPE = "OfflineRL"
    task_cfg.DATASET.SPLIT = split

    offline = task_cfg.TRAIN.OFFLINE
    offline.TYPE = "OfflineRL"
    offline.model = model
    offline.num_epochs = int(epochs)
    offline.batch_size = int(params["batch_size"])

    offline.beta = float(params["beta"])
    offline.actor_lr = float(params["actor_lr"])
    offline.critic_lr = float(params["critic_lr"])
    offline.alpha_lr = float(params["alpha_lr"])
    offline.gamma = float(params["gamma"])
    offline.tau = float(params["tau"])
    offline.target_entropy = float(params["target_entropy"])

    if online_test_epoch >= 0:
        if online_test_epoch == 0:
            offline.online_test_epoch = max(1, int(epochs) + 1)
        else:
            offline.online_test_epoch = int(online_test_epoch)

    offline.EXPERIMENT_CKPT_DIR = str(run_dir)

    # snapshot params for reproducibility
    snapshot = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "model": model,
        "params": params,
        "epochs": int(epochs),
        "online_test_epoch": int(offline.online_test_epoch),
        "split": split,
        "offline_cfg_yaml": offline.dump(),
    }
    with open(run_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    cfg.TASK_CONFIG.freeze()
    cfg.freeze()
    return cfg


def _print_result(res: RunResult) -> None:
    if res.status == "ok":
        print(f"[OK] {res.run_id} train={res.train_seconds:.1f}s")
    else:
        print(f"[FAIL] {res.run_id} error={res.error}")


def main() -> int:
    args = parse_args()
    from configs.default import get_config as get_config_fn

    run_module = _load_run_module()
    model = str(args.model).strip()
    if args.ckpt_root == "media/TRAIN_BATCH_V1" and model == "v1_5":
        args.ckpt_root = "media/TRAIN_BATCH_V1_5"

    params_list = _build_param_list(args)
    if args.max_runs and args.max_runs > 0:
        params_list = params_list[: args.max_runs]
    if not params_list:
        raise RuntimeError("No run config generated.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = Path(args.ckpt_root) / ts
    batch_root.mkdir(parents=True, exist_ok=True)

    report_path = (
        Path(args.report)
        if args.report
        else Path("tmp") / f"run_batch_{model}_{ts}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"Batch start: {datetime.now().isoformat(timespec='seconds')}")
    print(f"model={model} | total_runs={len(params_list)}")
    print(f"epochs={args.epochs} split={args.split}")
    print(f"batch_root={batch_root}")
    print("=" * 100)

    results: List[RunResult] = []

    for idx, params in enumerate(params_list, start=1):
        run_id = _run_id(idx, model, params)
        print(f"\n[{idx}/{len(params_list)}] running {run_id}")

        run_dir = batch_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        res = RunResult(run_id=run_id, status="failed", params=params, run_dir=str(run_dir))
        stop_now = False

        try:
            cfg = _prepare_config_for_model(
                get_config_fn=get_config_fn,
                model=model,
                params=params,
                epochs=args.epochs,
                online_test_epoch=args.online_test_epoch,
                split=args.split,
                run_dir=run_dir,
            )

            t0 = time.time()
            run_module._run_train(cfg, cfg.TASK_CONFIG.TRAIN)
            res.train_seconds = time.time() - t0

            _remove_checkpoints(run_dir)
            res.status = "ok"
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            if args.stop_on_error:
                stop_now = True
        finally:
            # save per-run result file
            with open(run_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(asdict(res), f, indent=2, ensure_ascii=False)

            results.append(res)
            _print_result(res)
            _cleanup_cuda()

            payload = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": f"offline_{model}_param_sweep",
                "args": vars(args),
                "batch_root": str(batch_root),
                "results": [asdict(r) for r in results],
            }
            ok_results = [r for r in results if r.status == "ok"]
            payload["ok_runs"] = len(ok_results)
            payload["failed_runs"] = len(results) - len(ok_results)

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

        if stop_now:
            break

    ok_results = [r for r in results if r.status == "ok"]
    print("\n" + "=" * 100)
    print(f"Finished runs: total={len(results)} ok={len(ok_results)} failed={len(results) - len(ok_results)}")
    print(f"Report saved to: {report_path}")
    print("=" * 100)

    return 0 if ok_results else 1


if __name__ == "__main__":
    raise SystemExit(main())
# python run_batch.py --epochs 100 --grid-json grid_json.json
# python run_batch.py --model v1_5 --epochs 150 --grid-json grid_json.json --online-test-epoch 10
# 
