import argparse
import glob
import os
import time
import torch


def benchmark_single(file_path, repeat, map_location):
    times = []
    size_bytes = os.path.getsize(file_path)
    for _ in range(repeat):
        t0 = time.perf_counter()
        print(f"begin load")
        _ = torch.load(file_path, map_location=map_location)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    throughput_mb_s = (size_bytes / (1024 * 1024)) / avg if avg > 0 else 0.0
    print(
        f"[torch.load] file={file_path} size={size_bytes / (1024 * 1024):.2f}MB "
        f"repeat={repeat} avg={avg * 1000:.2f}ms min={min(times) * 1000:.2f}ms "
        f"max={max(times) * 1000:.2f}ms throughput={throughput_mb_s:.2f}MB/s"
    )


def benchmark_multi(pattern, repeat, map_location, limit):
    files = sorted(glob.glob(pattern))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    total_size = 0
    total_time = 0.0
    total_count = 0

    for f in files:
        size_bytes = os.path.getsize(f)
        total_size += size_bytes * repeat
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = torch.load(f, map_location=map_location ,_use_new_zipfile_serialization=True)
            total_time += time.perf_counter() - t0
            total_count += 1

    avg = total_time / total_count
    throughput_mb_s = (total_size / (1024 * 1024)) / total_time if total_time > 0 else 0.0
    print(
        f"[torch.load] files={len(files)} total_loads={total_count} "
        f"avg={avg * 1000:.2f}ms total={total_time:.2f}s throughput={throughput_mb_s:.2f}MB/s"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark torch.load speed.")
    parser.add_argument("--file", type=str, default="", help="Single .pt/.pth file path")
    parser.add_argument("--glob", type=str, default="", help="Glob pattern for multiple files")
    parser.add_argument("--repeat", type=int, default=5, help="Repeat count per file")
    parser.add_argument("--limit", type=int, default=0, help="Limit matched files (0=all)")
    parser.add_argument(
        "--map-location",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="map_location passed to torch.load",
    )
    args = parser.parse_args()

    if not args.file and not args.glob:
        raise ValueError("Please provide --file or --glob")
    if args.file and args.glob:
        raise ValueError("Use only one of --file or --glob")

    if args.file:
        benchmark_single(args.file, args.repeat, args.map_location)
    else:
        benchmark_multi(args.glob, args.repeat, args.map_location, args.limit)


if __name__ == "__main__":
    main()
    
# python ai_code_rubbish/use_ray.py --file /path/to/a.pt --repeat 10 --map-location cpu
