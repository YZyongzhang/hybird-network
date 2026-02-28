#!/usr/bin/env python3
"""
Monitor a training process and send an email if it stops.

Examples:
  1) Monitor by PID
     python tools/monitor_process_email.py \
       --pid 12345 \
       --smtp-host smtp.gmail.com --smtp-port 465 --smtp-ssl \
       --smtp-user you@gmail.com --smtp-password-env SMTP_PASS \
       --from-email you@gmail.com --to-email you@gmail.com

  2) Monitor by process name pattern
     python tools/monitor_process_email.py \
       --name "python run.py" \
       --smtp-host smtp.qq.com --smtp-port 465 --smtp-ssl \
       --smtp-user your@qq.com --smtp-password-env SMTP_PASS \
       --from-email your@qq.com --to-email target@qq.com
"""

import argparse
import os
import smtplib
import socket
import subprocess
import sys
import time
from datetime import datetime
from email.message import EmailMessage
from typing import List, Optional


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    proc_path = f"/proc/{pid}"
    if not os.path.exists(proc_path):
        return False

    # Exclude zombie process.
    try:
        stat = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "stat="], text=True
        ).strip()
        if not stat:
            return False
        return "Z" not in stat.upper()
    except Exception:
        return False


def _find_pids_by_name(pattern: str) -> List[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
    except subprocess.CalledProcessError:
        return []

    pids: List[int] = []
    self_pid = os.getpid()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        pids.append(pid)

    # Filter out zombies
    return [pid for pid in pids if _is_pid_alive(pid)]


def _is_target_alive(pid: Optional[int], name: Optional[str]) -> bool:
    if pid is not None:
        return _is_pid_alive(pid)
    if name:
        return len(_find_pids_by_name(name)) > 0
    return False


def _send_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    use_ssl: bool,
    use_starttls: bool,
    from_email: str,
    to_email: str,
    subject: str,
    body: str,
) -> None:
    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    if use_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            if use_starttls:
                server.starttls()
                server.ehlo()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)


def _format_target(pid: Optional[int], name: Optional[str]) -> str:
    if pid is not None:
        return f"PID={pid}"
    return f"NAME={name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor process and send email on stop")

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=int, help="Target process pid")
    target.add_argument("--name", type=str, help="Process name pattern for pgrep -f")

    parser.add_argument("--check-interval", type=int, default=20, help="Check interval seconds")
    parser.add_argument(
        "--grace-seconds",
        type=int,
        default=30,
        help="Only alert if process is continuously missing for this duration",
    )

    parser.add_argument("--smtp-host", required=True, help="SMTP server host")
    parser.add_argument("--smtp-port", type=int, required=True, help="SMTP server port")
    parser.add_argument("--smtp-user", required=True, help="SMTP login username")
    parser.add_argument(
        "--smtp-password-env",
        required=True,
        help="Environment variable name storing SMTP password or app password",
    )
    parser.add_argument("--smtp-ssl", action="store_true", help="Use SMTP SSL")
    parser.add_argument("--smtp-starttls", action="store_true", help="Use STARTTLS")

    parser.add_argument("--from-email", required=True, help="Sender email")
    parser.add_argument("--to-email", required=True, help="Receiver email")
    parser.add_argument(
        "--subject-prefix",
        default="[Process Stopped]",
        help="Email subject prefix",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.smtp_ssl and args.smtp_starttls:
        print("[ERROR] --smtp-ssl and --smtp-starttls cannot both be enabled", file=sys.stderr)
        return 2

    smtp_password = os.environ.get(args.smtp_password_env, "")
    if not smtp_password:
        print(
            f"[ERROR] Missing SMTP password env: {args.smtp_password_env}",
            file=sys.stderr,
        )
        return 2

    target_desc = _format_target(args.pid, args.name)
    host = socket.gethostname()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[INFO] Monitor started at {start_time}, target={target_desc}, host={host}")
    print(
        f"[INFO] check_interval={args.check_interval}s grace_seconds={args.grace_seconds}s"
    )

    missing_since: Optional[float] = None

    try:
        while True:
            alive = _is_target_alive(args.pid, args.name)
            now = time.time()

            if alive:
                if missing_since is not None:
                    print("[INFO] Target recovered, reset missing timer")
                missing_since = None
            else:
                if missing_since is None:
                    missing_since = now
                    print("[WARN] Target not found, start grace timer")
                missing_for = now - missing_since
                if missing_for >= args.grace_seconds:
                    stop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    subject = f"{args.subject_prefix} {target_desc} @ {host}"
                    body = (
                        "Detected target process has stopped.\n\n"
                        f"host: {host}\n"
                        f"target: {target_desc}\n"
                        f"detected_at: {stop_time}\n"
                        f"monitor_started_at: {start_time}\n"
                        f"grace_seconds: {args.grace_seconds}\n"
                    )
                    _send_email(
                        smtp_host=args.smtp_host,
                        smtp_port=args.smtp_port,
                        smtp_user=args.smtp_user,
                        smtp_password=smtp_password,
                        use_ssl=args.smtp_ssl,
                        use_starttls=args.smtp_starttls,
                        from_email=args.from_email,
                        to_email=args.to_email,
                        subject=subject,
                        body=body,
                    )
                    print("[ALERT] Email sent. Monitor exits.")
                    return 1

            time.sleep(max(1, args.check_interval))
    except KeyboardInterrupt:
        print("\n[INFO] Monitor stopped by user")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
