"""Allocate a Colab GPU, generate batches, and copy each batch back locally.

Stops only session r3035. Never stops another assignment.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BATCHES = RESULTS / "batches"
LOG = RESULTS / "drive.log"
SESSION = "r3035"
PROTOCOL = json.loads((SCRIPTS / "protocol.json").read_text())
# Observed L4 rate from the earlier 4B attempt and a sibling L4 run. Used only
# when this session's marginal rate cannot be isolated. T4/A100 defaults are
# conservative so a missed measurement still stops inside 5 CU.
DEFAULT_RATE = {"L4": 1.54, "T4": 1.20, "A100": 4.00}


def log(msg: str, echo: bool = False) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    RESULTS.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")
    if echo:
        print(msg, flush=True)


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    log("cmd " + " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def parse_usage(text: str) -> dict:
    bal = re.search(r"Current balance:\s*([0-9.]+)", text)
    rate = re.search(r"Usage rate:\s*([0-9.]+)", text)
    assigns = re.search(r"Active assignments:\s*([0-9]+)", text)
    return {
        "balance": float(bal.group(1)) if bal else None,
        "rate_per_h": float(rate.group(1)) if rate else None,
        "assignments": int(assigns.group(1)) if assigns else None,
        "raw": text.strip(),
    }


def usage() -> dict:
    proc = run(["colab", "usage"], timeout=60)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_usage(text)
    (RESULTS / "usage_latest.txt").write_text(text)
    return parsed


def colab_exec(script: Path, timeout: int) -> tuple[int, str]:
    proc = run(
        ["colab", "exec", "-s", SESSION, "-f", str(script), "--timeout", str(timeout)],
        timeout=timeout + 60,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log(f"exec {script.name} rc={proc.returncode} bytes={len(text)}")
    (RESULTS / f"exec_{script.name}.log").write_text(text[-20000:])
    return proc.returncode, text


def upload(local: Path, remote: str) -> None:
    proc = run(["colab", "upload", "-s", SESSION, str(local), remote], timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"upload failed {local} -> {remote}\n{proc.stdout}\n{proc.stderr}")


def download(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".partial")
    proc = run(["colab", "download", "-s", SESSION, remote, str(tmp)], timeout=180)
    if proc.returncode != 0:
        log(f"download miss {remote} {(proc.stderr or '')[-300:]}")
        if tmp.exists():
            tmp.unlink()
        return False
    tmp.replace(local)
    return True


def load_budget() -> dict:
    path = RESULTS / "budget.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"allocations": [], "cu_used": 0.0, "stopped_for_budget": False}


def save_budget(budget: dict) -> None:
    (RESULTS / "budget.json").write_text(json.dumps(budget, indent=2))


def capacity_error(text: str) -> bool:
    low = text.lower()
    keys = ("refused", "precondition", "too many", "capacity", "quota", "unavailable", "not available")
    return any(k in low for k in keys)


def allocate(budget: dict) -> tuple[str, float]:
    """Return (gpu, marginal CU/h). Raises SystemExit if no GPU can be started."""
    plan = [("L4", 15), ("T4", 8), ("A100", 6)]
    attempt_log = RESULTS / "allocation_log.txt"
    for gpu, tries in plan:
        for attempt in range(1, tries + 1):
            if budget["cu_used"] >= float(PROTOCOL["cu_stop_at"]):
                raise SystemExit("budget already consumed before allocation")
            before = usage()
            log(f"allocate attempt {gpu} {attempt}/{tries} before={before}", echo=True)
            with attempt_log.open("a") as handle:
                handle.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} {gpu} try {attempt} ---\n")
                handle.write(json.dumps(before) + "\n")
            proc = run(["colab", "new", "--gpu", gpu, "-s", SESSION], timeout=180)
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            with attempt_log.open("a") as handle:
                handle.write(text + "\n")
            if proc.returncode == 0 and "refused" not in text.lower():
                after = usage()
                marginal = None
                if (
                    before["rate_per_h"] is not None
                    and after["rate_per_h"] is not None
                    and before["assignments"] is not None
                    and after["assignments"] is not None
                    and after["assignments"] == before["assignments"] + 1
                ):
                    marginal = round(after["rate_per_h"] - before["rate_per_h"], 4)
                if marginal is None or marginal <= 0.2 or marginal > 20:
                    marginal = DEFAULT_RATE[gpu]
                    log(f"marginal rate not isolated, using default {marginal} for {gpu}")
                log(f"SESSION_READY {gpu} marginal_cu_per_h={marginal} after={after}", echo=True)
                budget["allocations"].append(
                    {
                        "session": SESSION,
                        "gpu": gpu,
                        "start": time.time(),
                        "stop": None,
                        "marginal_cu_per_h": marginal,
                        "balance_before": before["balance"],
                        "rate_before": before["rate_per_h"],
                        "rate_after": after["rate_per_h"],
                    }
                )
                save_budget(budget)
                (RESULTS / "hard_stop_epoch.txt").write_text(str(int(deadline_epoch(budget))))
                return gpu, marginal
            log(f"allocation failed {gpu} try {attempt}: {text[-500:]}")
            if not capacity_error(text) and "session" not in text.lower():
                # Auth or usage errors should not spin for 75 minutes.
                if "balance" in text.lower() and "50" in text:
                    raise SystemExit(text)
            if attempt < tries:
                log("waiting 300s before retry")
                time.sleep(300)
        log(f"{gpu} exhausted retries, trying the next accelerator", echo=True)
    raise SystemExit("no GPU could be allocated")


def active_allocation(budget: dict) -> dict | None:
    if budget["allocations"] and budget["allocations"][-1]["stop"] is None:
        return budget["allocations"][-1]
    return None


def deadline_epoch(budget: dict) -> float:
    alloc = active_allocation(budget)
    rate = alloc["marginal_cu_per_h"] if alloc else DEFAULT_RATE["L4"]
    remaining = float(PROTOCOL["cu_stop_at"]) - float(budget["cu_used"])
    seconds = max(60.0, remaining / rate * 3600.0 - 90.0)
    start = alloc["start"] if alloc else time.time()
    return start + seconds


def close_allocation(budget: dict, reason: str) -> None:
    alloc = active_allocation(budget)
    if not alloc:
        return
    alloc["stop"] = time.time()
    alloc["stop_reason"] = reason
    hours = (alloc["stop"] - alloc["start"]) / 3600.0
    alloc["cu"] = hours * alloc["marginal_cu_per_h"]
    budget["cu_used"] = round(sum(a.get("cu", 0.0) for a in budget["allocations"]), 4)
    if reason == "budget":
        budget["stopped_for_budget"] = True
    save_budget(budget)
    log(f"closed allocation cu_used={budget['cu_used']} reason={reason} hours={hours:.3f}")


def stop_session(budget: dict, reason: str) -> None:
    close_allocation(budget, reason)
    proc = run(["colab", "stop", "-s", SESSION], timeout=120)
    text = (proc.stdout or "") + (proc.stderr or "")
    log(f"colab stop rc={proc.returncode} {text.strip()}")
    (RESULTS / "stop.txt").write_text(text)
    epoch = RESULTS / "hard_stop_epoch.txt"
    if epoch.exists():
        epoch.unlink()


def remote_status() -> dict | None:
    rc, text = colab_exec(SCRIPTS / "remote_status.py", 90)
    if rc != 0 or "BEGIN_STATUS" not in text:
        log("status failed " + text[-400:])
        return None
    body = text.split("BEGIN_STATUS", 1)[1].split("END_STATUS", 1)[0]
    pid = ""
    alive = False
    progress = None
    error = False
    for line in body.splitlines():
        if line.startswith("PID "):
            pid = line.split(" ", 1)[1].strip()
        elif line.startswith("ALIVE "):
            alive = line.split(" ", 1)[1].strip() == "True"
        elif line.startswith("PROGRESS "):
            raw = line.split(" ", 1)[1].strip()
            if raw != "NONE":
                try:
                    progress = json.loads(raw)
                except json.JSONDecodeError:
                    progress = {"status": "unparsed"}
        elif line.startswith("ERROR "):
            error = line.split(" ", 1)[1].strip() == "YES"
    return {"pid": pid, "alive": alive, "progress": progress, "error": error, "raw": body}


def sync_down(progress: dict | None) -> int:
    download("/content/035r3/progress.json", RESULTS / "progress.json")
    download("/content/035r3/metadata.json", RESULTS / "metadata.json")
    download("/content/035r3/questions.json", RESULTS / "questions.json")
    if (RESULTS / "progress.json").exists():
        try:
            progress = json.loads((RESULTS / "progress.json").read_text())
        except json.JSONDecodeError:
            pass
    files = []
    if progress:
        files = progress.get("files") or []
    got = 0
    for name in files:
        dest = BATCHES / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        if download(f"/content/035r3/batches/{name}", dest):
            got += 1
            log(f"saved {name}", echo=False)
    if (RESULTS / "error.txt").exists() or (progress and progress.get("status") == "error"):
        download("/content/035r3/error.txt", RESULTS / "error.txt")
    return got


def local_completed() -> int:
    positions = set()
    for path in BATCHES.glob("b*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                positions.add(json.loads(line)["pos"])
    if not positions:
        return 0
    # Count only the contiguous prefix so a hole is not treated as done.
    n = 0
    while n in positions:
        n += 1
    return n


def push_resume_state() -> None:
    colab_exec(SCRIPTS / "remote_mkdir.py", 60)
    upload(SCRIPTS / "protocol.json", "/content/035r3/protocol.json")
    upload(SCRIPTS / "infer.py", "/content/infer.py")
    if (RESULTS / "questions.json").exists():
        upload(RESULTS / "questions.json", "/content/035r3/questions.json")
    # Upload local batches the remote does not have yet. A fresh VM has none.
    remote = remote_status() or {}
    have = set()
    if remote.get("progress"):
        have = set(remote["progress"].get("files") or [])
    for path in sorted(BATCHES.glob("b*.jsonl")):
        if path.name not in have:
            upload(path, f"/content/035r3/batches/{path.name}")


def write_config(budget: dict, gpu: str) -> None:
    dtype = "float16" if gpu == "T4" else "bfloat16"
    batch_size = 4 if gpu == "T4" else 10
    config = {
        "dtype": dtype,
        "batch_size": batch_size,
        "deadline_epoch": deadline_epoch(budget),
        "gpu_requested": gpu,
    }
    path = RESULTS / "config.json"
    path.write_text(json.dumps(config, indent=2))
    upload(path, "/content/035r3/config.json")


def first_batch_gate() -> str | None:
    """Return a problem string if the first saved batch does not look like non-thinking."""
    paths = sorted(BATCHES.glob("b*.jsonl"))
    if not paths:
        return None
    tokens = []
    trunc = 0
    n = 0
    for line in paths[0].read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for sample in row["samples"]:
            n += 1
            tokens.append(sample["tokens"])
            trunc += int(sample["finish_reason"] == "length")
    if n == 0:
        return "empty first batch"
    mean_tokens = sum(tokens) / n
    rate = trunc / n
    identical = 0
    n_q = 0
    for line in paths[0].read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n_q += 1
        texts = [sample["text"] for sample in row["samples"]]
        if len(set(texts)) < 2:
            identical += 1
    ident_rate = identical / n_q if n_q else 1.0
    log(
        f"gate mean_tokens={mean_tokens:.1f} truncation_rate={rate:.3f} "
        f"identical_questions={identical}/{n_q}"
    )
    if rate > float(PROTOCOL["truncation_gate"]):
        return f"truncation rate {rate:.3f} exceeds {PROTOCOL['truncation_gate']}"
    if mean_tokens > float(PROTOCOL["token_gate_mean"]):
        return f"mean tokens {mean_tokens:.0f} exceeds {PROTOCOL['token_gate_mean']}"
    # More than half the first batch identical means n=10 did not actually sample.
    if ident_rate > 0.5:
        return f"sampler collapse: {identical}/{n_q} first-batch questions have identical texts"
    return None


def reject_batches(reason: str) -> None:
    rejected = RESULTS / "rejected_protocol"
    rejected.mkdir(exist_ok=True)
    for path in BATCHES.glob("b*.jsonl"):
        shutil.move(str(path), rejected / path.name)
    (rejected / "reason.txt").write_text(reason)


def generate(budget: dict, gpu: str) -> str:
    """Run until complete, budget_stop, error, or stall. Session stays up on return."""
    log("installing vllm", echo=True)
    rc, text = colab_exec(SCRIPTS / "remote_setup.py", 1800)
    if rc != 0 or "SETUP_OK" not in text:
        (RESULTS / "setup_fail.txt").write_text(text[-20000:])
        return "setup_failed"
    push_resume_state()
    write_config(budget, gpu)
    rc, text = colab_exec(SCRIPTS / "remote_launch.py", 90)
    if rc != 0 or "LAUNCHED" not in text:
        (RESULTS / "launch_fail.txt").write_text(text[-5000:])
        return "launch_failed"
    log("infer launched", echo=True)
    launched = time.time()
    last_progress_mark = launched
    last_completed = local_completed()
    gated = last_completed > 0
    planned = int(PROTOCOL["n_questions"])
    while True:
        if time.time() > deadline_epoch(budget):
            log("local deadline reached", echo=True)
            colab_exec(SCRIPTS / "remote_stop_infer.py", 60)
            time.sleep(5)
            sync_down(None)
            download("/content/infer.log", RESULTS / "infer.log")
            return "budget_stop"
        time.sleep(25)
        status = remote_status()
        if status is None:
            # One retry before treating the session as dead.
            time.sleep(10)
            status = remote_status()
        if status is None:
            sync_down(None)
            return "session_lost"
        sync_down(status.get("progress"))
        completed = local_completed()
        progress = status.get("progress") or {}
        state = progress.get("status", "")
        if completed != last_completed:
            last_completed = completed
            last_progress_mark = time.time()
            if completed in (10, 100, 250, 400, planned) or completed == planned:
                log(f"PROGRESS {completed}/{planned}", echo=True)
            else:
                log(f"progress {completed}/{planned}")
        else:
            log(f"heartbeat completed={completed} status={state} alive={status['alive']}")
        if not gated and completed > 0:
            gated = True
            problem = first_batch_gate()
            if problem:
                log("PROTOCOL_GATE " + problem, echo=True)
                colab_exec(SCRIPTS / "remote_stop_infer.py", 60)
                reject_batches(problem)
                return "protocol_gate"
        if state == "complete" or completed >= planned:
            sync_down(status.get("progress"))
            download("/content/infer.log", RESULTS / "infer.log")
            return "complete"
        if state == "budget_stop":
            return "budget_stop"
        if state == "error" or status.get("error"):
            download("/content/035r3/error.txt", RESULTS / "error.txt")
            download("/content/infer.log", RESULTS / "infer.log")
            tail = ""
            if (RESULTS / "error.txt").exists():
                tail = (RESULTS / "error.txt").read_text()[-600:].replace("\n", " | ")
            log("remote error " + tail, echo=True)
            return "error"
        if (
            not status["alive"]
            and time.time() - launched > 90
            and state in ("prepared", "loading", "running", "error", "")
        ):
            log(f"infer process is gone during {state or 'startup'}")
            return "dead"
        if time.time() - last_progress_mark > 20 * 60:
            log("stall: no new completed question for 20 min", echo=True)
            return "stall"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    BATCHES.mkdir(parents=True, exist_ok=True)
    budget = load_budget()
    before = usage()
    (RESULTS / "usage_before.txt").write_text(before["raw"])
    log(f"usage before {before}", echo=True)
    if before["balance"] is not None and before["balance"] < 50:
        log("balance below 50 CU, not starting", echo=True)
        return 3
    session_up = False
    outcome = "not_started"
    failures = 0
    try:
        while local_completed() < int(PROTOCOL["n_questions"]):
            if budget["cu_used"] >= float(PROTOCOL["cu_stop_at"]):
                budget["stopped_for_budget"] = True
                save_budget(budget)
                outcome = "budget_stop"
                break
            gpu, _rate = allocate(budget)
            session_up = True
            outcome = generate(budget, gpu)
            log(f"generate outcome {outcome}")
            if outcome == "complete":
                break
            if outcome == "budget_stop":
                budget["stopped_for_budget"] = True
                save_budget(budget)
                break
            if outcome == "protocol_gate":
                stop_session(budget, "protocol_gate")
                session_up = False
                log("PROTOCOL_GATE stopped the session so the template can be fixed", echo=True)
                return 2
            # setup/launch/error/stall/session_lost: stop this VM, resume if budget remains.
            stop_session(budget, outcome)
            session_up = False
            failures += 1
            if failures >= 3:
                log("FAILED three session attempts", echo=True)
                outcome = "failed"
                break
            # The hard stop is 90s early, so accounted CU can sit just under the cap.
            # Do not open another VM for that sliver.
            if budget["cu_used"] >= float(PROTOCOL["cu_stop_at"]) - 0.10:
                budget["stopped_for_budget"] = True
                save_budget(budget)
                outcome = "budget_stop"
                log(f"budget remainder too small to resume cu_used={budget['cu_used']}", echo=True)
                break
            log("will try a fresh session for the remaining questions", echo=True)
        if session_up:
            stop_session(budget, "done" if outcome == "complete" else outcome)
            session_up = False
        after = usage()
        (RESULTS / "usage_after.txt").write_text(after["raw"])
        budget["balance_after"] = after["balance"]
        budget["account_rate_after"] = after["rate_per_h"]
        save_budget(budget)
        if local_completed() == 0:
            log("FAILED no questions saved", echo=True)
            return 1
        log(f"DONE outcome={outcome} n={local_completed()} cu={budget['cu_used']}", echo=True)
        (RESULTS / "EXPERIMENT_DONE").write_text(outcome + "\n")
        return 0
    except Exception:
        err = traceback.format_exc()
        log("FAILED exception\n" + err, echo=True)
        (RESULTS / "drive_error.txt").write_text(err)
        return 1
    finally:
        if session_up:
            try:
                stop_session(load_budget(), "finally")
            except Exception as exc:
                log(f"stop in finally failed: {exc}", echo=True)


if __name__ == "__main__":
    sys.exit(main())
