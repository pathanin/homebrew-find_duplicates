# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-purpose tool: scan a directory for near-duplicate images (the same photo at different sizes/quality), then pick which one to keep from a browser page — LAN-capable, so a library on a NAS or headless box can be reviewed from another machine. One front end (the web UI), plus a plain-pip installer. No package, no `requirements.txt`, no test framework.

## Running

```bash
python3 find_duplicates.py [directory] [--threshold N] [--dest DIR] [--recursive] [--auto] [--dry-run] [--host H] [--port N] [--no-browser]
python3 compare_image_quality.py imageA.jpg imageB.jpg   # standalone 2-image comparison
```

Without `--auto` this prints a tokened URL and runs until Ctrl-C — it does *not* exit when a review finishes. Rescans happen from the page's own control panel (`POST /api/scan`), not by restarting the process. `--auto` keeps each group's top-scored file and never starts the web server.

Runtime deps: `numpy opencv-python-headless pillow pillow-heif fastapi uvicorn`. Install via `./install.sh`; **add any new dependency to that script's `pip install` block**, since there is no other manifest.

## Tests

Each test file is a standalone script with its own `main()` that asserts, prints `ok` lines, and exits non-zero on failure. Run them individually:

```bash
python3 tests/test_auto_mode.py
python3 tests/test_claude_md_test_list_sync.py
python3 tests/test_confirm_hash.py
python3 tests/test_effective_resolution_downsampling.py
python3 tests/test_fast_scan.py
python3 tests/test_group_ordering.py
python3 tests/test_heic_support.py
python3 tests/test_help_and_labels.py
python3 tests/test_recursive_scan.py
python3 tests/test_scan_progress.py
python3 tests/test_score_group.py
python3 tests/test_shutdown.py
python3 tests/test_streaming_scan.py
python3 tests/test_unapply_crash_safety.py
python3 tests/test_vectorized_sweep.py
python3 tests/test_web_api.py
python3 tests/test_web_progress.py
```

`test_claude_md_test_list_sync.py` machine-checks that list against `tests/*.py` in both directions. It finds the list by regex — literal `## Tests` heading, **first** ```` ```bash ```` fence after it, `tests/`-prefixed paths. Adding a test file without a line here fails that check; so does putting another bash fence between the heading and the list.

Tests reach the modules via `sys.path.insert(0, ...parent.parent)` — there is no install step. `test_web_api.py` additionally needs `httpx < 0.28` (test-only; `install.sh` does not install it): FastAPI's `TestClient` constructs its client with the `app=` shortcut httpx 0.28 removed, and the resulting `TypeError: unexpected keyword argument 'app'` is a version mismatch, not a bug.

Many tests exist to lock in one specific past bug. **Read a test's docstring before changing the code it covers** — it usually names a failure mode the assertion alone doesn't reveal.

## Architecture

Four modules, layered so the bottom two never know about the web:

- **`compare_image_quality.py`** — per-image quality metrics (`analyze`): laplacian sharpness, FFT-based `effective_resolution` (resists fake upscaling), noise, blockiness. Also runs standalone on two files. `brisque`/`niqe` are optional imports that stay unresolved by design.
- **`duplicates_core.py`** — the whole scan/score/move pipeline. `find_images` → `group_duplicates` (perceptual hash + `UnionFind`) → `analyze_paths` → `score_group` → `build_groups`, returning `list[Group]`. Applying decisions: `apply_group`, `apply_pick`, `unapply`, `auto_apply_groups`.
- **`duplicates_web.py`** — FastAPI app (`create_app`) plus the `Session` dataclass holding all server-side state. Routes: `/api/state`, `/api/group/{i}` and its `pick`/`confirm`/`skip` posts, `/api/thumb|stage|full/{i}/{j}`, `/api/scan`, `/api/progress` (SSE), `/api/metrics-info`, and a token-gated `/static/{path}`.
- **`find_duplicates.py`** — CLI entry point, `--auto` path, signal handling, uvicorn startup.

Front end is vanilla JS in `static/app.js` (no build step, no framework), organized in commented sections: state/API, queue sidebar, stage, switcher strip, ledger, decision bar.

### Scan lifecycle

`_launch_scan` runs `build_groups` on `_scan_executor` (a dedicated 1-worker pool). Progress, group and completion callbacks fire on that worker thread and marshal back into `Session` under its locks. `session.status` moves `idle → scanning → ready|error`, and the `scanning` status is what stops a second concurrent scan — `/api/scan`'s own check, which is separate from `_require_not_scanning` and must stay that way.

The **startup scan streams** (`_launch_scan(..., stream=True)`, `Session.streaming`): `build_groups`' `group_callback` appends each finished group into the live session, so review begins on group 1 while the rest of the library is still being analyzed. `params` and `generation` are set up front and `on_done` swaps nothing, so an index handed out mid-scan keeps addressing the same group and a mid-scan confirm's manifest entry survives. `_require_not_scanning` is exempt while `streaming` (the pick/confirm/skip routes only). A **rescan never streams** — its `on_done` replaces groups and manifest wholesale, which is exactly what the guard exists to protect. `/api/state`'s `streaming` flag is how the frontend knows a "scanning" status still means the groups below are reviewable.

Publication order is `raw_groups` order, not completion order, and a group is scored, permuted best-first and filtered (`< 2` valid members) *before* it is ever handed out. Both matter: `tests/test_streaming_scan.py` locks them.

### Grouping is two-stage

`phash_pair` returns a 64-bit DCT hash and a 256-bit confirmation hash. The 64-bit hash (`DEFAULT_HASH_THRESHOLD`, default 10/64) *proposes* a pair; the 256-bit half (`CONFIRM_HASH_THRESHOLD`) *confirms* it. The 64-bit hash alone cannot tell a re-export from a different frame of the same scene.

`CONFIRM_HASH_THRESHOLD` is tuned for **recall**: a false positive costs one keypress in the review UI, a false negative is never surfaced at all. Some near-identical frames group on purpose. Tightening it drops real duplicates before it stops those. `tests/Test-image/` will not tell you if you went too far — every known pair there sits at distance 0, nowhere near the tail the constant is set against. The tail cases are aspect recrops (one artwork exported for two screen sizes) and heavy downscales; `tests/test_confirm_hash.py` is what actually covers them.

### Group ordering

`build_groups` reorders each group best-first so the suggested file is always index 0 — the reviewer's eye shouldn't hunt for a ★ in a different position each group. Both `paths` and `results` are permuted together, so every downstream index (`current_pick`, the manifest, `/api/thumb`'s `j`, digit-key shortcuts) stays consistent; there is no "original index" to translate back to. The sort is stable and `find_images` returns sorted paths, so ties keep filename order.

## Traps

- `duplicates_core.py` must stay importable without the web stack — never import FastAPI/uvicorn into it.
- The hash/scoring constants are empirically tuned, not arbitrary: `DEFAULT_HASH_THRESHOLD`, `CONFIRM_HASH_THRESHOLD`, `CLOSE_CALL_MARGIN`, `MIN_REDUCED_DECODE_SIDE`, `METRIC_WEIGHTS`. Re-verify changes against the real photos in `tests/Test-image/`, not just unit tests.
- `load_hash_gray`'s reduced-decode and full-decode paths must agree on the **64-bit** hash bits — check `MIN_REDUCED_DECODE_SIDE` before touching either. They do *not* agree on the 256-bit half, which reaches into mid frequencies where the decode paths genuinely differ. That drift is content-dependent (single digits on real photos, tens of bits on synthetic noise), expected, and absorbed by `CONFIRM_HASH_THRESHOLD`. Don't chase it as a bug.
- `METRIC_WEIGHTS`, `METRIC_DESCRIPTIONS`, and `METRIC_ROWS` get entries added and removed together; the UI's help sheet renders off the first.
- Moving files is the only genuinely destructive path (`apply_group`, `_compute_dest`, `apply_pick`, `unapply`, `auto_apply_groups`). Non-kept files are **moved to `_duplicates/`, never deleted** — preserve that invariant. The manifest is in-memory only; recovery after process exit is a manual move back out.
- `apply_group` never sets `group.status` — the caller owns that, including the "stays pending on failure" invariant.
- `auto_apply_groups` must read `file_size` *before* the move; the source path is gone once `apply_group` returns.
- Any numpy-derived value reaching the JSON API needs an explicit `bool()`/`float()`/`int()` cast **where it's computed**. `numpy.bool_` doesn't subclass `bool` and isn't JSON-serializable (this bit `build_groups`'s `is_close_call`).
- `compare_image_quality.load_gray` builds one shared **float32** buffer, and the metrics fuse multi-pass numpy into single cv2 calls (`cv2.absdiff`, `cv2.norm(..., NORM_L1)`); `effective_resolution` uses `cv2.dft`, not `np.fft.fft2`. Analyze is the per-image bottleneck — "simplifying" these back to plain numpy halves throughput.
- `duplicates_web.py` imports core functions **by name**, so a test patching one for a route handler must patch `duplicates_web.X`, not `duplicates_core.X` — the latter silently does nothing. Names called bare inside `duplicates_core.py` (`load_hash_gray`, `ThreadPoolExecutor`) are patched there instead.
- `Session` is guarded by a plain `threading.Lock`, not `asyncio.Lock`: scan callbacks run on an executor thread, not the event loop. `progress` has its own separate lock so a slow file move can't stall progress updates. Keep the `scanning` guards (`_require_not_scanning`) on the mutating endpoints — its `streaming` exemption is for the append-only startup scan only, and `/api/scan` must keep its own separate check or a rescan fired mid-stream would run a second `build_groups` over the same caches.
- `analyze_paths` attaches `file_size` to each result as that result lands, not in one pass at the end — a group published mid-scan has to be complete. The cache-hit branch needs it too, or `/api/group`'s `sizes` KeyErrors on the second scan of any file (`tests/test_streaming_scan.py` covers this).
- `image_cache` is keyed `(generation, group, file, max_side)`. `max_side` is in the key because the 800px switcher preview and 1600px stage render of the same file would otherwise collide; `generation` is in it because a render runs outside the lock and a rescan finishing mid-render must not drop stale bytes into the fresh cache.
- `hash_cache`/`analyze_cache` deliberately **survive** a rescan and are never reset — that's what makes the control panel's rescan fast on a large library. Don't "fix" the missing reset.
- The frontend appends `?g=<generation>` to image URLs because `(i, j)` indices get reused across rescans and the browser's HTTP cache doesn't know that.
- Every request needs the token, supplied as `?token=` or the `fd_token` cookie that `GET /` sets (an `<img src>` can't carry a header). `/static` is a token-gated route rather than a `StaticFiles` mount for that reason — and not a `BaseHTTPMiddleware` guard, which would wrap `StreamingResponse` and disturb `/api/progress`'s disconnect handling.
- Read the design-direction comment at the top of `static/index.html` before changing layout. The stage swap is deliberately transition-free: a cross-fade hides the very difference being judged.
- Bind keyboard shortcuts on `KeyboardEvent.code`, not `.key` — an alternate layout remaps `.key` before the browser sees it.
- `install.sh` is POSIX sh, not bash (the curl-piped invocation ignores the shebang): no arrays, no `[[ ]]`, no `pipefail`.
- Ctrl-C shutdown has two moving parts, both regression-tested in `tests/test_shutdown.py`. Scans run in `duplicates_web._scan_executor`, not the loop's default executor (asyncio's teardown joins the default one, so Ctrl-C mid-scan would hang until the scan finished). And `main()` drives `server.serve()` on a bare loop then calls `os._exit(0)` (`asyncio.run`'s SIGINT handler turns a quick second Ctrl-C into a lifespan-cancel traceback). `main()` also sets `duplicates_web.shutting_down` from the signal handler so an open `/api/progress` stream ends itself, and flushes stdout/stderr, which `os._exit` skips (block-buffered under a redirect, so the tokened URL would otherwise be lost).
- Don't `pkill -f find_duplicates.py` while manually testing in a browser — it kills the server under test and the connection failure reads as a product bug.

## LSP

Code intelligence (the `LSP` tool: hover, goToDefinition, findReferences) comes from two user-scope plugins, not from anything in this repo: `pyright-lsp@claude-plugins-official` for `.py`, and `web-lsp` (`~/.claude/skills/web-lsp`) for `.html`/`.css`/`.json`. Both need their servers on PATH — `pyright-langserver`, and the `vscode-*-language-server` binaries from `vscode-langservers-extracted`.

`pyrightconfig.json` is gitignored and machine-local: it points pyright at a dev venv holding cv2/numpy/fastapi. `brisque` and `pyiqa` stay unresolved on purpose — they are optional imports. Diagnostics in `tests/` are mostly stub noise (`cv2.imread` typed Optional, `PIL.Image.LANCZOS` missing from stubs, duck-typed fakes); `duplicates_core.py` and `duplicates_web.py` sit at zero, so a new error there is real.
