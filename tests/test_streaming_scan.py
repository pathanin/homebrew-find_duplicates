"""Groups become reviewable while the startup scan is still running.

analyze() costs ~8-19x what hashing one file does, so on a library with
duplicates in it the analyze phase is most of the scan -- and the whole of it
used to be dead time before the first decision. build_groups now publishes
each group as soon as its own files are analyzed, and the startup scan
(only the startup scan) appends them into the live Session.

Three things are locked in here, and each one was a real hazard in the
design:

1. A group is complete and correctly ordered the first time anyone sees it.
   build_groups permutes a group's paths/results best-first; publishing
   before that permutation would hand out indices that later address a
   different file, and /api/thumb's j, current_pick and the manifest all
   ride on those indices. So group_callback must never fire on a group that
   is still going to change.

2. Decisions work mid-scan. That's the entire point -- _require_not_scanning
   is relaxed for a streaming scan.

3. /api/scan is still rejected mid-stream. It has its own "already running"
   check rather than going through _require_not_scanning; relaxing the
   latter must not have opened a path to two concurrent build_groups over
   the same caches, with both on_done callbacks racing to swap groups in.
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

import duplicates_core
import duplicates_web
from duplicates_web import ScanParams, create_app

TOKEN = "test-token"


def make_library(root: Path, n_groups: int) -> None:
    """n_groups pairs, each pair a distinct photo at two JPEG qualities."""
    rng = np.random.default_rng(0)
    for g in range(n_groups):
        base = rng.integers(0, 255, size=(160, 200, 3), dtype=np.uint8)
        # Smooth it so the two qualities still hash together: pure noise
        # survives requantization badly enough to land outside the threshold.
        img = Image.fromarray(base).resize((400, 320), Image.LANCZOS)
        img.save(root / f"photo{g:02d}_a.jpg", quality=95)
        img.save(root / f"photo{g:02d}_b.jpg", quality=60)


def client_for(root: Path):
    from fastapi.testclient import TestClient
    params = ScanParams(
        directory=root, threshold=duplicates_core.DEFAULT_HASH_THRESHOLD,
        recursive=False, dest_dir=root / "_duplicates", dry_run=False,
    )
    return TestClient(create_app(params, TOKEN))


def test_groups_publish_incrementally(tmp: Path) -> None:
    """group_callback fires per group during the scan, not once at the end,
    and every group it hands over is already final."""
    make_library(tmp, 6)

    seen: list[tuple[int, list[str]]] = []
    real_analyze = duplicates_core._analyze_one

    def slow_analyze(path_str):
        time.sleep(0.02)  # widen the window so "incremental" is observable
        return real_analyze(path_str)

    def on_group(group):
        # Snapshot the group as published; if anything reorders it later the
        # comparison against the returned list below will catch it.
        seen.append((len(seen), [p.name for p in group.paths]))

    duplicates_core._analyze_one = slow_analyze
    try:
        groups = duplicates_core.build_groups(
            tmp, duplicates_core.DEFAULT_HASH_THRESHOLD, group_callback=on_group,
        )
    finally:
        duplicates_core._analyze_one = real_analyze

    assert len(groups) == 6, f"expected 6 groups, got {len(groups)}"
    assert len(seen) == len(groups), f"callback fired {len(seen)}x for {len(groups)} groups"
    for (i, names), group in zip(seen, groups):
        assert names == [p.name for p in group.paths], (
            f"group {i} changed after publication: {names} -> {[p.name for p in group.paths]}"
        )
    print("ok  build_groups publishes each group once, already final")


def test_no_group_callback_is_unchanged(tmp: Path) -> None:
    """The default path (--auto, every test, the CLI) must be untouched."""
    make_library(tmp, 3)
    groups = duplicates_core.build_groups(tmp, duplicates_core.DEFAULT_HASH_THRESHOLD)
    assert len(groups) == 3, f"expected 3 groups, got {len(groups)}"
    for g in groups:
        assert len(g.paths) == 2
        assert g.suggested_idx == 0 and g.current_pick == 0
        # file_size is attached per result now, not in one pass at the end.
        assert all(r["file_size"] > 0 for r in g.results), "file_size missing"
    print("ok  build_groups without a callback behaves as before")


def test_file_size_present_on_a_cached_rescan(tmp: Path) -> None:
    """The cache-hit branch has to attach file_size too. It used to come from
    a single pass at the end of analyze_paths, which covered hits for free;
    per-result assignment does not, and /api/group's `sizes` would KeyError on
    the second scan of any file."""
    make_library(tmp, 2)
    cache: dict = {}
    duplicates_core.build_groups(tmp, duplicates_core.DEFAULT_HASH_THRESHOLD, analyze_cache=cache)
    assert cache, "analyze cache stayed empty; this test would prove nothing"
    groups = duplicates_core.build_groups(tmp, duplicates_core.DEFAULT_HASH_THRESHOLD, analyze_cache=cache)
    for g in groups:
        for r in g.results:
            assert "file_size" in r and r["file_size"] > 0, "cached result lost file_size"
    print("ok  cached results still carry file_size")


def test_review_works_mid_scan(tmp: Path) -> None:
    """A confirm lands while the startup scan is still running, and /api/scan
    is still refused."""
    make_library(tmp, 8)

    gate = threading.Event()
    real_analyze = duplicates_core._analyze_one
    calls = {"n": 0}

    def gated_analyze(path_str):
        calls["n"] += 1
        # Let the first pair through, then hold the scan open until the test
        # has finished exercising the mid-scan endpoints.
        if calls["n"] > 2:
            gate.wait(timeout=10)
        return real_analyze(path_str)

    duplicates_core._analyze_one = gated_analyze
    try:
        with client_for(tmp) as client:
            session = client.app.state.session
            deadline = time.time() + 10
            while time.time() < deadline:
                with session.lock:
                    if session.groups:
                        break
                time.sleep(0.02)

            state = client.get("/api/state", params={"token": TOKEN}).json()
            assert state["status"] == "scanning", state["status"]
            assert state["streaming"] is True, "startup scan should stream"
            assert state["groups"], "no group published while scanning"
            assert state["generation"] == 1, (
                f"generation must be live before the first group, got {state['generation']}"
            )

            # 2. a decision lands mid-scan
            r = client.post("/api/group/0/confirm", params={"token": TOKEN, "g": 1})
            assert r.status_code == 200, f"confirm mid-scan: {r.status_code} {r.text}"
            assert r.json()["status"] == "confirmed"
            moved = list((tmp / "_duplicates").glob("*.jpg"))
            assert len(moved) == 1, f"expected 1 file moved, found {moved}"

            # 3. a rescan is still refused: two concurrent build_groups over
            #    the same caches would race in on_done.
            r = client.post(
                "/api/scan", params={"token": TOKEN},
                json={"directory": str(tmp), "threshold": 10},
            )
            assert r.status_code == 409, f"rescan mid-stream should be 409, got {r.status_code}"

            gate.set()
            deadline = time.time() + 20
            while time.time() < deadline:
                with session.lock:
                    if session.status == "ready":
                        break
                time.sleep(0.02)
            with session.lock:
                assert session.status == "ready", session.status
                assert session.streaming is False, "streaming flag outlived the scan"
                # The manifest must survive: on_done resets nothing for a
                # streamed scan, or the mid-scan confirm would be forgotten
                # and unapply could never put those files back.
                assert len(session.manifest) == 1, f"manifest lost the mid-scan confirm: {session.manifest}"
                assert session.groups[0].status == "confirmed"
                assert session.generation == 1, "generation must not bump at the end of a stream"
    finally:
        gate.set()
        duplicates_core._analyze_one = real_analyze
    print("ok  confirm works mid-scan; rescan refused; manifest survives")


def test_progress_stream_reports_group_count(tmp: Path) -> None:
    """The SSE frame carries the published group count -- the frontend's only
    cue that there is something new to show."""
    make_library(tmp, 3)
    with client_for(tmp) as client:
        session = client.app.state.session
        deadline = time.time() + 20
        while time.time() < deadline:
            with session.lock:
                if session.status == "ready":
                    break
            time.sleep(0.02)
        with client.stream("GET", "/api/progress", params={"token": TOKEN}) as resp:
            payload = ""
            for line in resp.iter_lines():
                payload += line
                if line.startswith("data:"):
                    break
    assert '"groups":' in payload, f"progress frame has no group count: {payload}"
    print("ok  /api/progress reports the group count")


def main() -> int:
    import tempfile
    tests = [
        test_groups_publish_incrementally,
        test_no_group_callback_is_unchanged,
        test_file_size_present_on_a_cached_rescan,
        test_review_works_mid_scan,
        test_progress_stream_reports_group_count,
    ]
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
    print("all streaming-scan tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
