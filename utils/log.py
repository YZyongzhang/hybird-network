import logging
import os
from datetime import datetime


logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
logger.propagate = False

_is_configured = False


def setup_run_logger(base_dir: str = "logs", run_name: str = "") -> str:
    """
    Configure logger once per process.
    Returns the final log file path.
    """
    global _is_configured

    if _is_configured and logger.handlers:
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                return h.baseFilename
        return ""

    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name.strip() if run_name else "run"
    log_path = os.path.join(base_dir, f"{name}_{ts}.log")

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    _is_configured = True
    logger.info("logger initialized: %s", log_path)
    return log_path


def append_experiment_journal(
    run_name: str,
    note: str,
    journal_path: str = "logs/experiment_journal.md",
) -> str:
    """
    Append one short record per run, to prevent forgetting experiment context.
    """
    os.makedirs(os.path.dirname(journal_path) or ".", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_note = note.strip() if note and note.strip() else "no note"
    line = f"- [{ts}] run={run_name} note={final_note}\n"
    with open(journal_path, "a", encoding="utf-8") as f:
        f.write(line)
    return journal_path
