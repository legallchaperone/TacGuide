#!/usr/bin/env python3
"""Job queue for a shared GPU server (plan item 2.7). Standard library only.

Whenever a card is free, start the highest-priority job whose prerequisites exist,
one card per job. More free cards means more jobs in parallel; one card means the
jobs run one after another.

    python3 training/tacguide/gpu_queue.py run --queue training/tacguide/queue.gc.json   # in tmux
    python3 training/tacguide/gpu_queue.py run --queue ... --once --dry-run              # show the next decision
    python3 training/tacguide/gpu_queue.py status --queue ...
    python3 training/tacguide/gpu_queue.py yield --queue ... --gpu 3                     # give card 3 back now
    python3 training/tacguide/gpu_queue.py yield --queue ... --job train_gc_C

Rules it enforces:
  * A card is free only if its memory use is near zero AND no compute process is on
    it, and it must still be free on a second look just before launch.
  * It only ever signals processes it started itself.
  * A job that exits without its `done_if` file is restarted (with `resume_cmd` once
    `resume_if` exists) up to `max_attempts` times.
  * `yield` stops a job at once, keeps the scheduler off that card for a while and
    lets the job continue on another free card.

Queue file (JSON):
  {"state_dir": "...", "log_dir": "...", "cwd": "...",
   "jobs": [{"name": "...", "priority": 1, "cmd": "...", "resume_cmd": "...",
             "resume_if": "glob", "requires": ["glob", ...], "done_if": "glob",
             "vars_from": {"w": "file"}, "enabled": true, "max_attempts": 5}]}
Strings may use {name} placeholders (variables from `vars_from`, plus {gpu}) and
$ENV variables. A job whose variable file does not exist yet is not ready.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import glob
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

FREE_MEMORY_MIB = 500  # "near zero": an idle card reports a few MiB
CONFIRM_SECONDS = 20
GPU_QUERY = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
APP_QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]


@dataclasses.dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    memory_used_mib: int
    utilization: int
    pids: tuple[int, ...] = ()


def parse_nvidia_smi(gpu_csv: str, app_csv: str) -> list[Gpu]:
    apps: dict[str, list[int]] = {}
    for line in app_csv.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            apps.setdefault(parts[0], []).append(int(parts[1]))
    gpus = []
    for line in gpu_csv.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        utilization = int(parts[3]) if parts[3].isdigit() else 100  # unreadable counts as busy
        memory = int(parts[2]) if parts[2].isdigit() else 10**9
        gpus.append(Gpu(int(parts[0]), parts[1], memory, utilization, tuple(apps.get(parts[1], ()))))
    return gpus


def read_gpus() -> list[Gpu]:
    def run(command):
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=60).stdout

    return parse_nvidia_smi(run(GPU_QUERY), run(APP_QUERY))


def free_gpu_indices(gpus, *, taken=(), held=(), max_memory_mib=FREE_MEMORY_MIB) -> list[int]:
    """Cards with no process and near-zero memory that we neither use nor promised away."""
    return [
        gpu.index
        for gpu in gpus
        if not gpu.pids and gpu.memory_used_mib <= max_memory_mib and gpu.index not in taken and gpu.index not in held
    ]


# ---------------------------------------------------------------- queue file


def _expand(text: str, variables: dict) -> str:
    return os.path.expandvars(text.format(**variables))


def load_queue(path) -> dict:
    queue = json.loads(pathlib.Path(path).read_text())
    names = [job["name"] for job in queue["jobs"]]
    if len(names) != len(set(names)):
        raise ValueError("job names in the queue must be unique")
    queue.setdefault("cwd", str(pathlib.Path.cwd()))
    for key in ("cwd", "state_dir", "log_dir"):
        queue[key] = os.path.expandvars(queue[key])
        if "$" in queue[key]:
            raise ValueError(f"{key} = {queue[key]!r} has an unset variable; run `source training/tacguide_env.sh` first")
    return queue


def job_variables(job: dict, cwd: str) -> dict | None:
    """Values for the job's placeholders, or None while a variable file is missing."""
    variables = {}
    for name, source in job.get("vars_from", {}).items():
        path = pathlib.Path(cwd, os.path.expandvars(source))
        if not path.is_file() or not path.read_text().strip():
            return None
        variables[name] = path.read_text().strip()
    return variables


def _matches(pattern: str, cwd: str, variables: dict) -> bool:
    return bool(glob.glob(str(pathlib.Path(cwd, _expand(pattern, variables)))))


def job_is_done(job: dict, cwd: str) -> bool:
    variables = job_variables(job, cwd)
    return variables is not None and "done_if" in job and _matches(job["done_if"], cwd, variables)


def missing_requirements(job: dict, cwd: str) -> list[str]:
    variables = job_variables(job, cwd)
    if variables is None:
        return [f"variable file for {sorted(job['vars_from'])}"]
    return [pattern for pattern in job.get("requires", []) if not _matches(pattern, cwd, variables)]


def command_for(job: dict, cwd: str, gpu: int) -> str:
    variables = {**(job_variables(job, cwd) or {}), "gpu": gpu}
    resume = "resume_cmd" in job and "resume_if" in job and _matches(job["resume_if"], cwd, variables)
    return _expand(job["resume_cmd"] if resume else job["cmd"], variables)


def plan(queue: dict, state: dict, free_gpus: list[int]) -> list[tuple[str, int]]:
    """Pure decision: which pending jobs go on which free cards, best priority first."""
    cwd = queue["cwd"]
    ready = []
    for position, job in enumerate(queue["jobs"]):
        record = state["jobs"].get(job["name"], {})
        if not job.get("enabled", True) or record.get("status") in ("running", "done", "failed"):
            continue
        if job_is_done(job, cwd) or missing_requirements(job, cwd):
            continue
        ready.append((job.get("priority", 100), position, job["name"]))
    return [(name, gpu) for (_, _, name), gpu in zip(sorted(ready), sorted(free_gpus))]


# ---------------------------------------------------------------- state


def _state_path(queue) -> pathlib.Path:
    return pathlib.Path(queue["state_dir"]) / "state.json"


@contextlib.contextmanager
def locked_state(queue):
    """Read-modify-write of the state file under an exclusive lock."""
    directory = pathlib.Path(queue["state_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "state.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = _state_path(queue)
        state = json.loads(path.read_text()) if path.exists() else {}
        state.setdefault("jobs", {})
        state.setdefault("holds", {})
        yield state
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(path)


def _start_ticks(pid: int) -> str | None:
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def is_alive(record: dict) -> bool:
    """True while the process we started (not a later reuse of its pid) still runs."""
    pid = record.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False  # the pid now belongs to someone else
    try:
        fields = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return True  # no /proc (not Linux): the signal check above is all we have
    if fields[0] == "Z":
        return False  # exited, only waiting to be collected by its parent
    return record.get("start_ticks") in (None, fields[19])


def reap(queue: dict, state: dict, log=print) -> None:
    """Settle jobs whose process has exited, and mark jobs whose output already exists."""
    cwd = queue["cwd"]
    for job in queue["jobs"]:
        record = state["jobs"].setdefault(job["name"], {"status": "pending", "attempts": 0})
        if record["status"] == "running" and not is_alive(record):
            record.update(pid=None, ended_at=time.time())
            if job_is_done(job, cwd):
                record["status"] = "done"
                log(f"{job['name']}: finished")
            else:
                record["attempts"] = record.get("attempts", 0) + 1
                if record["attempts"] >= job.get("max_attempts", 5):
                    record["status"] = "failed"
                    log(f"{job['name']}: exited without its output {record['attempts']} times, giving up (log: {record.get('log')})")
                else:
                    record["status"] = "pending"
                    log(f"{job['name']}: exited without its output (attempt {record['attempts']}), will retry")
        elif record["status"] in ("pending", "failed") and job_is_done(job, cwd):
            record["status"] = "done"
    now = time.time()
    state["holds"] = {gpu: until for gpu, until in state["holds"].items() if until > now}


def taken_gpus(state: dict) -> set[int]:
    return {record["gpu"] for record in state["jobs"].values() if record.get("status") == "running"}


def held_gpus(state: dict) -> set[int]:
    return {int(gpu) for gpu in state["holds"]}


def launch(queue: dict, state: dict, name: str, gpu: int, log=print) -> None:
    job = next(job for job in queue["jobs"] if job["name"] == name)
    command = command_for(job, queue["cwd"], gpu)
    log_dir = pathlib.Path(queue["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.{time.strftime('%Y%m%d-%H%M%S')}.log"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    with open(log_path, "ab") as output:
        output.write(f"# gpu {gpu}: {command}\n".encode())
        output.flush()
        process = subprocess.Popen(  # noqa: S602 - the queue file is ours
            ["bash", "-c", command],
            cwd=queue["cwd"],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # own process group: survives the scheduler, can be stopped as a unit
        )
    record = state["jobs"].setdefault(name, {"attempts": 0})
    record.update(
        status="running",
        pid=process.pid,
        start_ticks=_start_ticks(process.pid),
        gpu=gpu,
        log=str(log_path),
        command=command,
        started_at=time.time(),
    )
    log(f"{name}: started on gpu {gpu}, pid {process.pid}, log {log_path}")


def stop_job(record: dict, grace_seconds: float = 30.0) -> None:
    """Stop a job we started: SIGTERM to its process group, SIGKILL after the grace period."""
    if not is_alive(record):
        return
    pid = record["pid"]
    for signum, wait in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            return
        deadline = time.time() + wait
        while time.time() < deadline:
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
            if not is_alive(record):
                return
            time.sleep(0.2)


# ---------------------------------------------------------------- commands


def tick(queue, *, read=read_gpus, dry_run=False, confirm_seconds=CONFIRM_SECONDS, log=print) -> list[tuple[str, int]]:
    """One scheduling round. Returns the (job, gpu) pairs that were started (or would be)."""
    with contextlib.suppress(ChildProcessError):
        while os.waitpid(-1, os.WNOHANG)[0]:
            pass
    with locked_state(queue) as state:
        reap(queue, state, log)
        free = free_gpu_indices(read(), taken=taken_gpus(state), held=held_gpus(state))
        candidates = plan(queue, state, free)
    if dry_run:
        for name, gpu in candidates:
            job = next(job for job in queue["jobs"] if job["name"] == name)
            log(f"would start {name} on gpu {gpu}: {command_for(job, queue['cwd'], gpu)}")
        if not candidates:
            log(f"nothing to start (free cards: {free or 'none'})")
        return candidates
    if not candidates:
        return []
    time.sleep(confirm_seconds)  # look again before taking a card: someone may have just started on it
    started = []
    with locked_state(queue) as state:
        reap(queue, state, log)
        still_free = free_gpu_indices(read(), taken=taken_gpus(state), held=held_gpus(state))
        for name, gpu in plan(queue, state, [gpu for _, gpu in candidates if gpu in still_free]):
            launch(queue, state, name, gpu, log)
            started.append((name, gpu))
    return started


def command_run(queue, args):
    directory = pathlib.Path(queue["state_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    singleton = open(directory / "scheduler.lock", "w")  # noqa: SIM115 - held for the process lifetime
    if not args.dry_run:
        try:
            fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another scheduler is already running on this queue") from None

    def log(message):
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {message}", flush=True)

    while True:
        try:
            tick(load_queue(args.queue), dry_run=args.dry_run, confirm_seconds=args.confirm, log=log)
        except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
            log(f"round skipped: {error!r}")  # e.g. nvidia-smi hiccup or a half-edited queue file
        if args.once:
            return
        time.sleep(args.interval)


def command_status(queue, args):
    with locked_state(queue) as state:
        reap(queue, state, log=lambda message: None)
        records, holds = dict(state["jobs"]), dict(state["holds"])
    try:
        gpus = read_gpus()
    except (subprocess.SubprocessError, OSError):
        gpus = []
    cwd = queue["cwd"]
    for job in sorted(queue["jobs"], key=lambda job: job.get("priority", 100)):
        record = records.get(job["name"], {})
        status = record.get("status", "pending")
        detail = ""
        if not job.get("enabled", True) and status == "pending":
            status = "disabled"
        elif status == "running":
            hours = (time.time() - record["started_at"]) / 3600
            detail = f"gpu {record['gpu']}, pid {record['pid']}, {hours:.1f} h, {record['log']}"
        elif status == "pending":
            missing = missing_requirements(job, cwd)
            detail = "ready" if not missing else "waiting for " + ", ".join(missing)
        elif status == "failed":
            detail = f"see {record.get('log')}"
        print(f"{job.get('priority', 100):>5}  {job['name']:<24} {status:<9} {detail}")
    if gpus:
        free = free_gpu_indices(gpus, taken={r["gpu"] for r in records.values() if r.get("status") == "running"})
        print(f"free cards: {free or 'none'}   held back: {sorted(int(g) for g in holds) or 'none'}")


def command_yield(queue, args):
    with locked_state(queue) as state:
        targets = [
            (name, record)
            for name, record in state["jobs"].items()
            if record.get("status") == "running"
            and ((args.job and name == args.job) or (args.gpu is not None and record.get("gpu") == args.gpu))
        ]
        gpus = {record["gpu"] for _, record in targets}
        if args.gpu is not None:
            gpus.add(args.gpu)
        for gpu in gpus:
            state["holds"][str(gpu)] = time.time() + args.hold_minutes * 60
        for name, record in targets:
            stop_job(record)
            record.update(status="pending", pid=None, ended_at=time.time())  # not counted as a failed attempt
            print(f"{name}: stopped, gpu {record['gpu']} released; it will resume on another free card")
        if not targets:
            print("no running job matched; the card is held back anyway" if gpus else "no running job matched")
        if gpus:
            print(f"cards {sorted(gpus)} are held back for {args.hold_minutes} minutes")


def command_retry(queue, args):
    with locked_state(queue) as state:
        record = state["jobs"].get(args.job)
        if not record or record.get("status") != "failed":
            raise SystemExit(f"{args.job} is not in the failed state")
        record.update(status="pending", attempts=0)
        print(f"{args.job}: back in the queue")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "yield", "retry"):
        sub = commands.add_parser(name)
        sub.add_argument("--queue", required=True)
        if name == "run":
            sub.add_argument("--interval", type=float, default=180.0, help="seconds between rounds")
            sub.add_argument("--confirm", type=float, default=CONFIRM_SECONDS, help="seconds before the second look")
            sub.add_argument("--once", action="store_true")
            sub.add_argument("--dry-run", action="store_true")
        if name == "yield":
            group = sub.add_mutually_exclusive_group(required=True)
            group.add_argument("--gpu", type=int)
            group.add_argument("--job")
            sub.add_argument("--hold-minutes", type=float, default=60.0)
        if name == "retry":
            sub.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    queue = load_queue(args.queue)
    {"run": command_run, "status": command_status, "yield": command_yield, "retry": command_retry}[args.command](
        queue, args
    )


if __name__ == "__main__":
    sys.exit(main())
