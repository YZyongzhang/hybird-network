#!/usr/bin/env python3
"""
Batch runner for OfflineRL model variants.

It trains each requested OfflineRL version sequentially, then runs a quick
online rollout evaluation, and finally reports the best-performing version.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class RunResult:
    model: str
    status: str
    train_ckpt: str = ""
    avg_reward: Optional[float] = None
    avg_spl: Optional[float] = None
    error: str = ""
    train_seconds: float = 0.0
    eval_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OfflineRL versions one-by-one overnight and pick the best one."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["v1", "v1_3", "v1_4", "v1_5", "v1_6", "v2", "v4", "v5"],
        help="Offline model variants to run in order.",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Override TRAIN.OFFLINE.num_epochs.")
    parser.add_argument(
        "--online-test-epoch",
        type=int,
        default=0,
        help="Override TRAIN.OFFLINE.online_test_epoch (0 means disable during training).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=16,
        help="Rollout episodes per model after training.",
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=200,
        help="Max steps per rollout episode during evaluation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val_multiple",
        help="Dataset split for post-train evaluation.",
    )
    parser.add_argument(
        "--ckpt-root",
        type=str,
        default="media/TRAIN_BATCH",
        help="Root directory for this batch's checkpoints.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Optional output json path. Default: tmp/run_batch_<timestamp>.json",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the whole batch immediately when one model fails.",
    )
    return parser.parse_args()


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _latest_checkpoint(dir_path: Path) -> str:
    if not dir_path.exists():
        return ""
    files = sorted(
        [p for p in dir_path.rglob("*.pth") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(files[0]) if files else ""


def _load_run_module():
    run_path = Path(__file__).resolve().parent / "run.py"
    spec = importlib.util.spec_from_file_location("run_entry", run_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load run module from: {run_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_config_for_model(
    get_config_fn,
    model: str,
    epochs: int,
    online_test_epoch: int,
    split: str,
    ckpt_root: Path,
) -> tuple:
    cfg = get_config_fn()
    cfg.defrost()
    cfg.TASK_CONFIG.defrost()

    task_cfg = cfg.TASK_CONFIG
    task_cfg.COLLECT.OPEN = False
    task_cfg.PT.OPEN = False
    task_cfg.EVAL.OPEN = False
    task_cfg.TRAIN.OPEN = True
    task_cfg.TRAIN.TYPE = "OfflineRL"

    offline = task_cfg.TRAIN.OFFLINE
    offline.TYPE = "OfflineRL"
    offline.model = model
    offline.num_epochs = int(epochs)

    if online_test_epoch >= 0:
        if online_test_epoch == 0:
            offline.online_test_epoch = max(1, int(epochs) + 1)
        else:
            offline.online_test_epoch = int(online_test_epoch)

    model_ckpt_dir = ckpt_root / model
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)
    offline.EXPERIMENT_CKPT_DIR = str(model_ckpt_dir)

    task_cfg.DATASET.SPLIT = split

    cfg.TASK_CONFIG.freeze()
    cfg.freeze()
    return cfg, model_ckpt_dir


def _evaluate_with_online_test(run_module, cfg, ckpt_path: str, eval_episodes: int, eval_max_steps: int):
    cfg = cfg.clone()
    cfg.defrost()
    cfg.TASK_CONFIG.defrost()
    cfg.TASK_CONFIG.TRAIN.OFFLINE.MAX_STEPS = int(eval_max_steps)
    cfg.TASK_CONFIG.freeze()
    cfg.freeze()

    offline_cfg = cfg.TASK_CONFIG.TRAIN.OFFLINE

    sac_model, _trainer, online_test = run_module._build_offline_agent(cfg, offline_cfg)
    state = torch.load(ckpt_path, map_location=run_module._get_device())
    sac_model.load_state_dict(state, strict=False)
    sac_model.eval()

    old_num_episodes = int(online_test.env._env.number_of_episodes)
    try:
        if int(eval_episodes) > 0:
            online_test.env._env.number_of_episodes = int(eval_episodes)
    except Exception:
        pass

    reward, spl = online_test.rollout(epoch=0, sac_model=sac_model, logger=run_module.logger)

    try:
        online_test.env._env.number_of_episodes = old_num_episodes
    except Exception:
        pass

    return float(reward), float(spl)


def _print_result(res: RunResult) -> None:
    if res.status == "ok":
        print(
            f"[OK] model={res.model} avg_spl={res.avg_spl:.4f} avg_reward={res.avg_reward:.4f} "
            f"train={res.train_seconds:.1f}s eval={res.eval_seconds:.1f}s ckpt={res.train_ckpt}"
        )
    else:
        print(f"[FAIL] model={res.model} error={res.error}")


def main() -> int:
    args = parse_args()
    from configs.default import get_config as get_config_fn

    run_module = _load_run_module()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_root = Path(args.ckpt_root) / ts
    ckpt_root.mkdir(parents=True, exist_ok=True)

    if args.report:
        report_path = Path(args.report)
    else:
        report_path = Path("tmp") / f"run_batch_{ts}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"Batch start: {datetime.now().isoformat(timespec='seconds')}")
    print(f"models={args.models}")
    print(f"epochs={args.epochs} eval_episodes={args.eval_episodes} split={args.split}")
    print(f"ckpt_root={ckpt_root}")
    print("=" * 90)

    results: List[RunResult] = []

    for idx, model in enumerate(args.models, start=1):
        print(f"\n[{idx}/{len(args.models)}] running model={model}")
        res = RunResult(model=model, status="failed")
        stop_now = False

        try:
            cfg, model_ckpt_dir = _prepare_config_for_model(
                get_config_fn=get_config_fn,
                model=model,
                epochs=args.epochs,
                online_test_epoch=args.online_test_epoch,
                split=args.split,
                ckpt_root=ckpt_root,
            )

            t0 = time.time()
            run_module._run_train(cfg, cfg.TASK_CONFIG.TRAIN)
            res.train_seconds = time.time() - t0

            ckpt_path = _latest_checkpoint(model_ckpt_dir)
            if not ckpt_path:
                raise RuntimeError(f"No checkpoint found under: {model_ckpt_dir}")
            res.train_ckpt = ckpt_path

            t1 = time.time()
            avg_reward, avg_spl = _evaluate_with_online_test(
                run_module=run_module,
                cfg=cfg,
                ckpt_path=ckpt_path,
                eval_episodes=args.eval_episodes,
                eval_max_steps=args.eval_max_steps,
            )
            res.eval_seconds = time.time() - t1

            res.avg_reward = avg_reward
            res.avg_spl = avg_spl
            res.status = "ok"
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            if args.stop_on_error:
                stop_now = True
        finally:
            results.append(res)
            _print_result(res)
            _cleanup_cuda()

            payload = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "args": vars(args),
                "results": [asdict(r) for r in results],
            }
            ok_results = [r for r in results if r.status == "ok" and r.avg_spl is not None]
            if ok_results:
                best = sorted(
                    ok_results,
                    key=lambda x: (float(x.avg_spl), float(x.avg_reward if x.avg_reward is not None else -1e9)),
                    reverse=True,
                )[0]
                payload["best"] = asdict(best)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

        if stop_now:
            break

    ok_results = [r for r in results if r.status == "ok" and r.avg_spl is not None]
    print("\n" + "=" * 90)
    if ok_results:
        best = sorted(
            ok_results,
            key=lambda x: (float(x.avg_spl), float(x.avg_reward if x.avg_reward is not None else -1e9)),
            reverse=True,
        )[0]
        print(
            f"Best model: {best.model} | avg_spl={best.avg_spl:.4f} | "
            f"avg_reward={best.avg_reward:.4f}"
        )
    else:
        print("No successful model run in this batch.")
    print(f"Report saved to: {report_path}")
    print("=" * 90)

    return 0 if ok_results else 1


if __name__ == "__main__":
    raise SystemExit(main())
