"""Scheduler checks (plan item 2.7). No GPU needed: nvidia-smi output is injected."""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tacguide import gpu_queue as q

GPU_CSV = """0, GPU-aaa, 41843, 100
1, GPU-bbb, 4, 0
2, GPU-ccc, 300, 0
3, GPU-ddd, 3, 0
"""
APP_CSV = """GPU-aaa, 1300642, 41832
GPU-ccc, 777, 290
"""


def gpus(*indices):
    return [q.Gpu(i, f"GPU-{i}", 3, 0, (), 97887) for i in indices]


def pairs(assignments):
    return [(a.name, a.gpus[0]) for a in assignments]


def make_queue(tmp_path, jobs):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({
        "cwd": str(tmp_path), "state_dir": str(tmp_path / "state"), "log_dir": str(tmp_path / "logs"), "jobs": jobs,
    }))
    return path


def wait_for(condition, seconds=10):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_only_cards_with_no_process_and_no_memory_are_free():
    parsed = q.parse_nvidia_smi(GPU_CSV, APP_CSV)
    assert [g.index for g in parsed] == [0, 1, 2, 3]
    assert parsed[0].pids == (1300642,)
    # card 0: busy; card 2: little memory but someone's process is on it -> not free
    assert q.free_gpu_indices(parsed) == [1, 3]
    assert q.free_gpu_indices(parsed, taken={1}) == [3]
    assert q.free_gpu_indices(parsed, held={3}, taken={1}) == []


def test_plan_takes_highest_priority_ready_job(tmp_path):
    (tmp_path / "data").touch()
    queue = q.load_queue(make_queue(tmp_path, [
        {"name": "D", "priority": 3, "cmd": "true", "requires": ["A.done", "B.done"], "vars_from": {"w": "w.txt"}},
        {"name": "C", "priority": 3.5, "cmd": "true", "requires": ["A.done"]},
        {"name": "A", "priority": 1, "cmd": "true", "requires": ["data"], "done_if": "A.done"},
        {"name": "B", "priority": 1, "cmd": "true", "requires": ["data"], "done_if": "B.done"},
        {"name": "off", "priority": 0, "cmd": "true", "enabled": False},
    ]))
    state = {"jobs": {}, "holds": {}}
    assert pairs(q.plan(queue, state, [5])) == [("A", 5)]                 # one card: serial, best priority first
    assert pairs(q.plan(queue, state, [5, 2])) == [("A", 2), ("B", 5)]    # two cards: parallel
    assert pairs(q.plan(queue, state, [])) == []
    (tmp_path / "A.done").touch()
    assert pairs(q.plan(queue, state, [1, 2, 3])) == [("B", 1), ("C", 2)]  # C needs only A; D still waits
    (tmp_path / "B.done").touch()
    assert pairs(q.plan(queue, state, [1, 2])) == [("C", 1)]              # D has no w yet
    (tmp_path / "w.txt").write_text("1.5\n")
    assert pairs(q.plan(queue, state, [1])) == [("D", 1)]                 # one card: D before C
    state["jobs"]["D"] = {"status": "running", "gpu": 1}
    assert pairs(q.plan(queue, state, [2])) == [("C", 2)]


def test_variables_fill_commands_and_resume_switches_in(tmp_path):
    (tmp_path / "w.txt").write_text("2")
    job = {"name": "D", "cmd": "train gc_D_w{w} gpu{gpu}", "resume_cmd": "train gc_D_w{w} --resume",
           "resume_if": "out/gc_D_w{w}/[0-9]*", "done_if": "out/gc_D_w{w}/10000", "vars_from": {"w": "w.txt"}}
    assert q.command_for(job, str(tmp_path), 4) == "train gc_D_w2 gpu4"
    (tmp_path / "out/gc_D_w2/5000").mkdir(parents=True)
    assert q.command_for(job, str(tmp_path), 4) == "train gc_D_w2 --resume"
    assert not q.job_is_done(job, str(tmp_path))
    (tmp_path / "out/gc_D_w2/10000").mkdir()
    assert q.job_is_done(job, str(tmp_path))


def test_run_launches_on_the_free_card_and_marks_done(tmp_path):
    queue = q.load_queue(make_queue(tmp_path, [
        {"name": "job", "priority": 1, "cmd": "echo $CUDA_VISIBLE_DEVICES > card.txt && touch job.done", "done_if": "job.done"},
    ]))
    assert q.tick(queue, read=lambda: gpus(6), dry_run=True, log=lambda m: None) == [("job", 6)]
    assert not (tmp_path / "card.txt").exists()                      # dry run starts nothing
    assert q.tick(queue, read=lambda: gpus(6), confirm_seconds=0, log=lambda m: None) == [("job", 6)]
    assert wait_for(lambda: (tmp_path / "job.done").exists())
    assert (tmp_path / "card.txt").read_text().strip() == "6"        # one card, set through CUDA_VISIBLE_DEVICES
    assert wait_for(lambda: q.tick(queue, read=lambda: gpus(6), confirm_seconds=0, log=lambda m: None) == []
                    and json.loads((tmp_path / "state/state.json").read_text())["jobs"]["job"]["status"] == "done")


def test_card_taken_between_the_two_looks_is_left_alone(tmp_path):
    queue = q.load_queue(make_queue(tmp_path, [{"name": "job", "cmd": "touch started", "done_if": "never"}]))
    looks = iter([gpus(1), [q.Gpu(1, "GPU-1", 30000, 90, (4242,))]])
    assert q.tick(queue, read=lambda: next(looks), confirm_seconds=0, log=lambda m: None) == []
    time.sleep(0.3)
    assert not (tmp_path / "started").exists()


def test_failed_job_is_retried_with_resume_then_given_up(tmp_path):
    queue = q.load_queue(make_queue(tmp_path, [
        {"name": "job", "cmd": "mkdir -p ckpt/1000 && echo first >> runs && false",
         "resume_cmd": "echo resume >> runs && false", "resume_if": "ckpt/[0-9]*", "done_if": "ckpt/10000", "max_attempts": 3},
    ]))
    status = lambda: json.loads((tmp_path / "state/state.json").read_text())["jobs"]["job"]  # noqa: E731
    for _ in range(40):
        q.tick(queue, read=lambda: gpus(0), confirm_seconds=0, log=lambda m: None)
        if status()["status"] == "failed":
            break
        time.sleep(0.1)
    assert status()["status"] == "failed" and status()["attempts"] == 3
    assert (tmp_path / "runs").read_text().split() == ["first", "resume", "resume"]


def test_yield_stops_only_our_job_and_holds_the_card(tmp_path):
    queue_path = make_queue(tmp_path, [{"name": "job", "cmd": "sleep 300", "done_if": "never"}])
    queue = q.load_queue(queue_path)
    q.tick(queue, read=lambda: gpus(2), confirm_seconds=0, log=lambda m: None)
    record = json.loads((tmp_path / "state/state.json").read_text())["jobs"]["job"]
    assert record["status"] == "running" and q.is_alive(record)
    q.main(["yield", "--queue", str(queue_path), "--gpu", "2", "--hold-minutes", "5"])
    assert not q.is_alive(record)
    state = json.loads((tmp_path / "state/state.json").read_text())
    assert state["jobs"]["job"]["status"] == "pending" and state["jobs"]["job"]["attempts"] == 0
    assert q.tick(queue, read=lambda: gpus(2), confirm_seconds=0, log=lambda m: None) == []        # card 2 is held back
    assert q.tick(queue, read=lambda: gpus(2, 5), confirm_seconds=0, log=lambda m: None) == [("job", 5)]  # resumes elsewhere
    q.main(["yield", "--queue", str(queue_path), "--job", "job"])


def test_shipped_queue_file_is_valid():
    path = pathlib.Path(q.__file__).with_name("queue.gc.json")
    queue = json.loads(path.read_text())
    names = [job["name"] for job in queue["jobs"]]
    assert {"train_gc_C", "train_gc_D"} <= set(names) and not {"train_gc_A", "train_gc_B"} & set(names)
    assert not {2, 5, 7} & set(queue["allowed_gpus"])
    for job in queue["jobs"]:
        assert {"cmd", "resume_cmd", "resume_if", "done_if", "priority"} <= set(job)


def test_unset_variable_in_paths_is_refused(tmp_path, monkeypatch):
    import pytest

    monkeypatch.delenv("TG_UNSET_FOR_TEST", raising=False)
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"cwd": str(tmp_path), "state_dir": "$TG_UNSET_FOR_TEST/queue", "log_dir": str(tmp_path), "jobs": []}))
    with pytest.raises(ValueError):
        q.load_queue(path)


SHARED = [
    q.Gpu(0, "GPU-0", 41843, 100, (11,), 97887),   # another user's job, 54.7 GB left
    q.Gpu(4, "GPU-4", 41813, 100, (12,), 97887),
    q.Gpu(5, "GPU-5", 40223, 0, (13,), 97887),     # most memory left, but not in the allowed list
    q.Gpu(6, "GPU-6", 41795, 100, (14, 99), 97887),  # one of the project's own processes is here
    q.Gpu(7, "GPU-7", 86628, 100, (15,), 97887),
]
CMDLINES = {99: "/data2/cwx/TacGuide/venv/bin/python training/run.py train", 11: "python train_eight_gpu.py"}


def shared_queue(tmp_path, **extra):
    job = {"name": "C", "cmd": "train {placement_args}", "done_if": "never", "placements": [
        {"min_free_mib": [90000]},
        {"min_free_mib": [54000, 22000], "args": "--teacher-device cuda:1"},
    ]}
    path = make_queue(tmp_path, [job])
    data = json.loads(path.read_text())
    data.update(extra)
    path.write_text(json.dumps(data))
    return q.load_queue(path)


def test_placement_shares_cards_but_avoids_reserved_and_own_cards(tmp_path):
    queue = shared_queue(tmp_path, allowed_gpus=[0, 2, 4, 6], own_marker="TacGuide")
    state = {"jobs": {}, "holds": {}}
    lookup = lambda pid: CMDLINES.get(pid, "")  # noqa: E731
    # No empty card; student takes one shared card, the teacher another; never 5, 7 or the card with our own job.
    assert q.plan(queue, state, SHARED, lookup) == [q.Assignment("C", (0, 4), "--teacher-device cuda:1")]
    # A completely empty allowed card is preferred and needs no second card.
    empty = [*SHARED, q.Gpu(2, "GPU-2", 4, 0, (), 97887)]
    assert q.plan(queue, state, empty, lookup) == [q.Assignment("C", (2,), "")]
    # With only one usable shared card the job waits.
    assert q.plan(queue, state, [g for g in SHARED if g.index != 4], lookup) == []


def test_two_card_job_gets_both_cards_and_releases_both(tmp_path):
    queue_path = tmp_path / "queue.json"
    queue = shared_queue(tmp_path, allowed_gpus=[0, 4])
    queue["jobs"][0]["cmd"] = "echo $CUDA_VISIBLE_DEVICES {placement_args} > cards.txt && sleep 300"
    queue_path.write_text(json.dumps(queue))
    queue = q.load_queue(queue_path)
    assert q.tick(queue, read=lambda: SHARED, confirm_seconds=0, log=lambda m: None, cmdline_of=lambda pid: "") == [("C", 0)]
    assert wait_for(lambda: (tmp_path / "cards.txt").exists() and (tmp_path / "cards.txt").read_text().strip())
    assert (tmp_path / "cards.txt").read_text().split() == ["0,4", "--teacher-device", "cuda:1"]
    state = json.loads((tmp_path / "state/state.json").read_text())
    assert state["jobs"]["C"]["gpus"] == [0, 4] and q.taken_gpus(state) == {0, 4}
    q.main(["yield", "--queue", str(queue_path), "--gpu", "4", "--hold-minutes", "5"])   # the second card counts too
    state = json.loads((tmp_path / "state/state.json").read_text())
    assert state["jobs"]["C"]["status"] == "pending" and set(state["holds"]) == {"0", "4"}
