"""
duplicates_core.py

Shared, UI-agnostic core of the duplicate-image tool: directory scanning,
perceptual hashing + grouping, quality scoring, thumbnailing, and the one
destructive path (moving non-kept files out of the way). No web framework
dependency here -- this module must stay importable standalone (e.g. for
tests) without pulling in FastAPI/uvicorn.

find_duplicates.py and duplicates_web.py both import from this module
rather than duplicating any of it.
"""

import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import cv2
import numpy as np
from PIL import Image as PILImage

from compare_image_quality import analyze

# HEIC/HEIF (the default format Apple Photos/iPhone exports) has no reliable
# OS-level decoder behind cv2.imread, so PIL needs this optional plugin
# registered before PIL.Image.open can read those files. A missing package
# must never crash a scan -- HEIC files just fail to decode and get silently
# skipped like any other corrupt/unreadable file already does today.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif"}
DEFAULT_HASH_THRESHOLD = 10  # max Hamming distance out of 64 bits to call two images duplicates
# Max Hamming distance out of 256 bits for phash_pair's confirmation hash: a
# pair the 64-bit hash proposes is grouped only if the wider hash agrees too.
# Applied only at or below DEFAULT_HASH_THRESHOLD (see group_duplicates).
#
# Deliberately loose. The true-duplicate and same-scene distance ranges
# overlap, so no cut separates them: this sits *above* the worst duplicate
# observed rather than between the two, because a false positive costs one
# keypress in the review UI and a false negative is never surfaced at all.
# The binding case is an aspect recrop (one artwork exported for two phone
# screens), which moves far more bits than a rescale. Lowering this drops
# real duplicates well before it stops the near-identical frames that still
# get through -- 56 looked clean against re-exports alone and lost six real
# duplicates. tests/test_confirm_hash.py locks both directions.
CONFIRM_HASH_THRESHOLD = 88
PREVIEW_MAX_SIDE = 800
CLOSE_CALL_MARGIN = 0.08  # quality_score gap below which we flag "close call"
# phash resizes to 32x32; a reduced-scale decode smaller than this on either side
# would upsample instead of downsample there, drifting the hash. 64 gives margin,
# and images this small are cheap to fully decode anyway.
MIN_REDUCED_DECODE_SIDE = 64
# load_hash_gray/phash's cv2 calls (imread/resize/dct) release the GIL, so a
# thread pool gets real parallelism without ProcessPoolExecutor's process-spawn
# cost (benchmarked at ~0.6-3s on Apple M4/10 cores -- a serial loop used to
# beat a process pool below a few hundred files purely because of that spawn
# tax). Benchmarked on the same machine hashing synthetic 1600x1200 JPEGs: a
# thread pool beat both serial execution and a process pool at every batch
# size tried (30, 150, 3000 files) -- ~5-9x faster than serial at 30-150
# files (no spawn tax to pay), and matching or slightly beating a process
# pool's throughput at 3000. So hashing always parallelizes now; there's no
# serial fallback threshold to tune. cv2.setNumThreads(1) is set for the
# duration of the pool (see group_duplicates) so cv2's own internal thread
# pool doesn't oversubscription-fight these worker threads -- ~20% faster at
# 3000 files than leaving cv2's default thread count in place.
THREAD_POOL_WORKERS = os.cpu_count() or 1
# analyze()'s cv2 calls (resize/Laplacian/filter2D) and numpy's FFT release
# the GIL the same way hashing's do, so analyze_paths routes through a
# thread pool too -- same THREAD_POOL_WORKERS, no threshold. Benchmarked on
# the same machine: at a typical single-group batch (6 files), threads hit
# ~19.6 img/s against a process pool's ~8.7 img/s (indistinguishable from
# serial -- spawn overhead ate the whole benefit); at 300 files, ~36.3 img/s
# against ~23.1 img/s. Threads also sidestep the fork-after-cv2-threads
# macOS crash that motivated spawn-only ProcessPoolExecutor here before,
# since there's no forking or spawning at all.

# Weight > 0 means higher raw value is better; weight < 0 means lower raw value is better.
# effective_resolution_px_equiv is weighted heaviest since it's the metric most resistant
# to fake upscaling (true detail amount rather than just stored pixel count).
METRIC_WEIGHTS = {
    "effective_resolution_px_equiv": 0.35,
    "sharpness_normalized": 0.20,
    "effective_resolution_fraction": 0.15,
    "noise_sigma": -0.10,
    "blockiness": -0.10,
    "brisque": -0.10,
    "niqe": -0.10,
}

# One-line plain-English gloss per metric, shown in the in-app help (`?`).
# Keyed off METRIC_WEIGHTS so the help text can't drift out of sync with
# what's actually scored -- add a metric to the weights and its description
# is required here too, or the help screen would silently omit it.
METRIC_DESCRIPTIONS = {
    "effective_resolution_px_equiv": "true detail amount; resistant to fake upscaling",
    "sharpness_normalized": "edge/detail sharpness, compared at a common scale",
    "effective_resolution_fraction": "fraction of native resolution that's real detail, not just interpolated pixels",
    "noise_sigma": "sensor/compression noise",
    "blockiness": "JPEG block-edge artifacts",
    "brisque": "no-reference perceptual quality score (needs optional `brisque` package)",
    "niqe": "no-reference perceptual quality score (needs optional `pyiqa` package)",
}

class MetricRow(NamedTuple):
    """One row of the metrics table rendered by the web UI's /api/group/{i}
    response, structured so the frontend never has to re-derive a row's
    meaning by parsing `label` -- label text is for human display only (and
    free to be reworded), never a machine-readable contract. That used to be
    exactly what the web frontend did (matching a "(higher better)"/"(lower
    better)" suffix and a "Quality score" prefix out of the label string),
    which a purely cosmetic label rewording could silently break with no
    error and no test catching it.

    key: the METRIC_WEIGHTS key this row's raw value comes from, or None for
        rows that don't derive from a single weighted metric (Dimensions/File
        size, and quality_score itself -- kind="score" already says
        everything about that row).
    kind: "reference" (Dimensions/File size -- shown but never scored),
        "metric" (an individual weighted input -- comparable/highlightable
        against the other files in the group), or "score" (the single
        combined quality_score row).
    """
    label: str
    fn: Callable[[dict], str]
    key: str | None
    kind: str

    @property
    def direction(self) -> int:
        """+1 if a higher raw value is better, -1 if lower is better, 0 if
        this row isn't scored at all (kind == "reference"). Derived from
        METRIC_WEIGHTS' sign rather than hand-duplicated, so an edit to a
        weight's sign can't silently disagree with what this row claims."""
        if self.kind == "score":
            return 1  # a normalized composite of already-direction-corrected metrics: always higher-is-better
        if self.key is None:
            return 0
        return 1 if METRIC_WEIGHTS[self.key] > 0 else -1


# Every scored row is labeled with what a bigger/smaller number means, since
# a raw number is meaningless without knowing which direction is "better".
# Dimensions/file size carry no such label since they aren't part of the
# score at all -- that's explained once, in the '?' help screen, rather than
# on every row. Defined here (not in the web module) so it stays reusable
# by anything else that wants the same metrics-JSON shape without keeping a
# second label list in sync by hand.
METRIC_ROWS = [
    MetricRow("Dimensions", lambda r: f"{r['dimensions'][0]}x{r['dimensions'][1]}", None, "reference"),
    MetricRow("File size", lambda r: humansize(r["file_size"]), None, "reference"),
    MetricRow("Sharpness (higher better)", lambda r: f"{r['sharpness_normalized']:.1f}", "sharpness_normalized", "metric"),
    MetricRow("Eff. res. fraction (higher better)", lambda r: f"{r['effective_resolution_fraction']:.3f}", "effective_resolution_fraction", "metric"),
    MetricRow("Eff. res. px equiv (higher better)", lambda r: f"{r['effective_resolution_px_equiv']:.0f}", "effective_resolution_px_equiv", "metric"),
    MetricRow("Noise sigma (lower better)", lambda r: f"{r['noise_sigma']:.3f}", "noise_sigma", "metric"),
    MetricRow("Blockiness (lower better)", lambda r: f"{r['blockiness']:.3f}", "blockiness", "metric"),
    MetricRow("BRISQUE (lower better)", lambda r: f"{r['brisque']:.2f}" if r.get("brisque") is not None else "n/a", "brisque", "metric"),
    MetricRow("NIQE (lower better)", lambda r: f"{r['niqe']:.2f}" if r.get("niqe") is not None else "n/a", "niqe", "metric"),
    MetricRow("Quality score (higher better)", lambda r: f"{r['quality_score']:.3f}", None, "score"),
]


# ---------------------------------------------------------------------------
# Scanning + perceptual hashing + grouping
# ---------------------------------------------------------------------------

def find_images(directory: Path, recursive: bool = False, exclude_dir: Path | None = None) -> list[Path]:
    """Top-level-only scan by default. With *recursive*, walks subdirectories
    too -- *exclude_dir* (typically the move destination) is then required to
    keep a re-scan from picking up files already moved out by a prior run;
    it's meaningless (and ignored) in non-recursive mode since the default
    destination (<directory>/_duplicates) already sits below the top level
    iterdir() looks at."""
    if not recursive:
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    exclude_resolved = exclude_dir.resolve() if exclude_dir is not None else None
    found = []
    for p in directory.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if exclude_resolved is not None and p.resolve().is_relative_to(exclude_resolved):
            continue
        found.append(p)
    return sorted(found)


def _load_gray_via_pil(p: Path) -> np.ndarray | None:
    """Fallback decode for formats cv2 can't read at all (currently just
    HEIC/HEIF), via PIL + the registered pillow-heif opener. Always a full
    decode -- no reduced-scale trick like the cv2 path above, since
    correctness matters more than that specific optimization for this
    format. Returns None (rather than raising) on any decode failure so a
    HEIC file with no HEIF plugin installed, or a genuinely corrupt file,
    is silently skipped exactly like any other unreadable file today."""
    try:
        with PILImage.open(p) as pil_img:
            return np.array(pil_img.convert("L"))
    except Exception:
        return None


def load_hash_gray(p: Path) -> np.ndarray | None:
    """Grayscale decode for perceptual hashing. Uses a 1/8-scale DCT decode
    for speed (skips full-resolution JPEG decode just to shrink it to 32x32
    afterwards); falls back to a full decode when the image is small enough
    that the reduced decode would land below what the hash needs. Formats
    cv2 can't decode at all (e.g. HEIC/HEIF) fall through both cv2 attempts
    as None and get a full PIL-based decode instead."""
    img = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if img is None or min(img.shape) < MIN_REDUCED_DECODE_SIDE:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = _load_gray_via_pil(p)
    return img


def _low_freq_bits(block: np.ndarray) -> int:
    """Pack a square low-frequency DCT block into one big int, one bit per
    coefficient, thresholded against the block mean with DC excluded (DC is
    overall brightness and would swamp the mean). Row-major, first
    coefficient in the most significant bit."""
    avg = (block.sum() - block[0, 0]) / (block.size - 1)
    return int.from_bytes(np.packbits(block > avg).tobytes(), "big")


def phash_pair(gray: np.ndarray) -> tuple[int, int]:
    """Both perceptual hashes off a single DCT: the classic 64-bit one that
    proposes candidate pairs, and a 256-bit one that confirms them.

    The 64-bit hash keeps only an 8x8 low-frequency block, so it describes
    little more than a thumbnail's gross layout -- robust to resizing and
    recompression (the "same photo, different export" duplicate we're after),
    but by the same token nearly blind to the difference between two separate
    frames of one scene. The 16x16 block reaches into the mid frequencies
    where those frames actually diverge; see CONFIRM_HASH_THRESHOLD.

    Computing both here rather than in two passes is deliberate: the resize
    and DCT are shared, so the wider hash costs essentially nothing on top of
    a decode that dominates either way.

    The hash cache keys on path + mtime + size, not on this function's code,
    so a rescan within one process would serve old hashes after an edit here.
    That cache lives in memory only and dies with the process -- restart the
    tool after changing this or load_hash_gray and you're testing the new
    code, no cache file to hunt down and delete."""
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    return _low_freq_bits(dct[:8, :8]), _low_freq_bits(dct[:16, :16])


def phash(gray: np.ndarray) -> int:
    """The 64-bit hash alone, for callers that only need the grouping key."""
    return phash_pair(gray)[0]


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# Byte -> set-bit count, fallback for numpy < 2.0 (no np.bitwise_count).
_POPCOUNT8 = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)

# Byte budget for the sweep's XOR temp; shrink in tests to force >1 block.
_SWEEP_XOR_BYTE_BUDGET = 32_000_000


def _popcount(arr: np.ndarray) -> np.ndarray:
    """Per-element set-bit count of a uint64 array."""
    try:
        return np.bitwise_count(arr)
    except AttributeError:
        return _POPCOUNT8[arr.view(np.uint8).reshape(*arr.shape, -1)].sum(-1)


def _pack_hashes(values, nlanes: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack a list of int hashes (or None) into an (n, nlanes) uint64 array
    (big-endian bytes viewed as uint64 -- lane order and endianness don't
    matter to XOR or popcount). None rows are zeroed and flagged in `valid`."""
    n = len(values)
    arr = np.zeros((n, nlanes * 8), dtype=np.uint8)
    valid = np.zeros(n, dtype=bool)
    for i, v in enumerate(values):
        if v is not None:
            arr[i] = np.frombuffer(v.to_bytes(nlanes * 8, "big"), dtype=np.uint8)
            valid[i] = True
    return arr.view(np.uint64), valid


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def cached_hash(cache: dict, p: Path, st: os.stat_result) -> tuple[int, int] | None:
    """Returns the (grouping, confirmation) hash pair. None both when there's
    no entry and when the cached entry itself is None (the file failed to
    decode/hash last time) -- a permanently-corrupt file is simply re-attempted
    every run, no worse than today's uncached behavior for that one file."""
    entry = cache.get(str(p.resolve()))
    if entry is None or entry.get("mtime") != st.st_mtime_ns or entry.get("size") != st.st_size:
        return None
    return entry["hash"]


def store_hash(cache: dict, p: Path, st: os.stat_result,
               hash_value: tuple[int, int] | None) -> None:
    cache[str(p.resolve())] = {"mtime": st.st_mtime_ns, "size": st.st_size, "hash": hash_value}


def _stat_paths(paths: list[Path]) -> tuple[list[Path], dict[Path, os.stat_result]]:
    """stat() every path, dropping the ones that vanished or turned
    unreadable between listing and now (routine on a NAS or a syncing
    folder). A single missing file costs that file, not the whole scan."""
    kept: list[Path] = []
    stats: dict[Path, os.stat_result] = {}
    for p in paths:
        try:
            stats[p] = p.stat()
        except OSError:
            continue
        kept.append(p)
    return kept, stats


def _hash_one(p: Path) -> tuple[int, int] | None:
    img = load_hash_gray(p)
    return phash_pair(img) if img is not None else None


def _print_progress(label: str, done: int, total: int, tty: bool) -> None:
    """Incremental progress for a long-running scan phase. On a TTY,
    overwrites the same terminal line via carriage return so it doesn't spam
    scrollback; when stdout isn't a TTY (redirected to a file, running under
    test), falls back to occasional plain lines instead of \r-laden output."""
    if tty:
        print(f"\r{label}: {done}/{total}", end="", flush=True)
    elif done == total or done % 100 == 0:
        print(f"{label}: {done}/{total}")


def group_duplicates(
    paths: list[Path], threshold: int, cache: dict, progress_callback=None
) -> list[list[Path]]:
    """Groups `paths` by perceptual-hash Hamming distance, reusing `cache`
    for files whose (mtime, size) haven't changed (see cached_hash/
    store_hash above) so a re-scan of an already-hashed directory doesn't
    re-decode every old file. The uncached subset always hashes through a
    thread pool -- see THREAD_POOL_WORKERS for why threads (not a process
    pool) win here.

    *progress_callback*, if given, is called as progress_callback(label,
    done, total) as each uncached item completes, instead of the default
    TTY-aware print via _print_progress -- lets a caller (e.g. the web
    front end's SSE progress stream) route progress to something other than
    stdout without touching the CLI's default behavior."""
    paths, stats = _stat_paths(paths)
    hashes: dict[Path, tuple[int, int] | None] = {}
    to_compute = []
    for p in paths:
        cached = cached_hash(cache, p, stats[p])
        if cached is not None:
            hashes[p] = cached
        else:
            to_compute.append(p)

    if to_compute:
        total = len(to_compute)
        tty = sys.stdout.isatty()
        original_cv2_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
                computed = executor.map(_hash_one, to_compute)
                for done, (p, h) in enumerate(zip(to_compute, computed), start=1):
                    store_hash(cache, p, stats[p], h)
                    hashes[p] = h
                    if progress_callback is not None:
                        progress_callback("Hashing", done, total)
                    else:
                        _print_progress("Hashing", done, total, tty)
            if progress_callback is None and tty:
                print()
        finally:
            cv2.setNumThreads(original_cv2_threads)

    hash_list = [hashes[p] for p in paths]

    # Raising *threshold* is the documented way to ask for looser matching
    # ("Raise the threshold" is what the UI suggests when a scan finds too
    # little), and the two hashes are correlated enough that a fixed gate
    # would reject much of what the wider threshold just admitted -- turning
    # the knob into a partial no-op. Someone who widened it has already
    # accepted the false positives that come with it.
    confirm = threshold <= DEFAULT_HASH_THRESHOLD

    n = len(paths)
    uf = UnionFind(n)
    h64, valid = _pack_hashes([h[0] if h is not None else None for h in hash_list], 1)
    h64 = h64.reshape(-1)
    h256, _ = _pack_hashes([h[1] if h is not None else None for h in hash_list], 4)

    # ponytail: O(n^2) vectorized-popcount sweep -- bit-identical to the old
    # double loop, ~7-30x faster. Add pigeonhole-LSH banding if n routinely
    # exceeds ~100k. Blocked over rows to bound the XOR temp to the byte budget.
    block = max(1, _SWEEP_XOR_BYTE_BUDGET // (8 * max(n, 1)))
    for start in range(0, n, block):
        d64 = _popcount(h64[start:start + block, None] ^ h64[None, :])
        ii, jj = np.nonzero(d64 <= threshold)
        ii = ii + start
        sel = (jj > ii) & valid[ii] & valid[jj]
        ii, jj = ii[sel], jj[sel]
        if confirm and ii.size:
            d256 = _popcount(h256[ii] ^ h256[jj]).sum(axis=1, dtype=np.int16)
            keep = d256 <= CONFIRM_HASH_THRESHOLD
            ii, jj = ii[keep], jj[keep]
        for a, b in zip(ii.tolist(), jj.tolist()):
            uf.union(a, b)

    clusters: dict[int, list[Path]] = {}
    for i, p in enumerate(paths):
        if hash_list[i] is None:
            continue
        clusters.setdefault(uf.find(i), []).append(p)

    return [members for members in clusters.values() if len(members) > 1]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_group(results: list[dict]) -> None:
    """Attach a 0-1 'quality_score' to each result dict, min-max normalized
    within this group only (raw metric ranges aren't comparable across
    unrelated images, but are meaningful when comparing duplicates of the
    same photo)."""
    if not results:
        return
    keys = [
        k for k in METRIC_WEIGHTS
        if all(r.get(k) is not None and math.isfinite(r[k]) for r in results)
    ]
    total_weight = sum(abs(METRIC_WEIGHTS[k]) for k in keys) or 1.0

    ranges = {}
    for k in keys:
        vals = [r[k] for r in results]
        lo, hi = min(vals), max(vals)
        ranges[k] = (lo, hi if hi > lo else lo + 1e-9)

    for r in results:
        score = 0.0
        for k in keys:
            lo, hi = ranges[k]
            norm = (r[k] - lo) / (hi - lo)
            weight = METRIC_WEIGHTS[k]
            score += norm * weight if weight > 0 else (1 - norm) * abs(weight)
        r["quality_score"] = score / total_weight


def humansize(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if unit == "B":
            if n < 1024:
                return f"{n:.0f}{unit}"
        elif n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


THUMBNAIL_FAILURE_COLOR = (60, 60, 60)  # neutral gray placeholder, visually distinct from real photos


def make_thumbnail(path: Path) -> PILImage.Image:
    try:
        img = PILImage.open(path)
        img = img.convert("RGB")
        img.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
        return img
    except Exception:
        return PILImage.new("RGB", (PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE), THUMBNAIL_FAILURE_COLOR)


@dataclass
class Group:
    paths: list[Path]
    results: list[dict]
    thumbnails: list[PILImage.Image] | None  # lazily generated on first view, see refresh_detail
    suggested_idx: int
    current_pick: int
    is_close_call: bool
    status: str = "pending"  # pending | confirmed | skipped


def _compute_dest(
    path: Path,
    dest_dir: Path,
    dry_run: bool,
    recursive: bool = False,
    scan_root: Path | None = None,
) -> Path:
    """Where *path* should land under *dest_dir* if moved as a non-kept
    duplicate. Preserves *path*'s position relative to *scan_root* when
    *recursive* (so two same-named files from different subdirectories don't
    collide into one flat name); just the filename otherwise. A collision
    suffix keeps the same relative parent directory -- dropping it would
    silently flatten the file into dest_dir's root, defeating the point of
    preserving structure in the first place."""
    rel = path.relative_to(scan_root) if (recursive and scan_root is not None) else Path(path.name)
    dest = dest_dir / rel
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while dest.exists():
        dest = dest_dir / rel.parent / f"{rel.stem}_dup{n}{rel.suffix}"
        n += 1
    return dest


def apply_group(
    group: Group,
    group_index: int,
    keep_idx: int,
    dest_dir: Path,
    dry_run: bool = False,
    manifest: list[dict] | None = None,
    recursive: bool = False,
    scan_root: Path | None = None,
) -> dict:
    """Move every non-kept file in *group* to *dest_dir*. Shared by the
    interactive confirm path (apply_pick) and --auto mode. Files that are
    symlinks to the kept file's real target are left alone (moving the
    target would leave the kept path dangling). Records whatever was
    actually moved into a manifest entry from a `finally` -- even a failure
    partway through the loop (disk full, permission error) leaves an
    accurate, reversible record instead of silently losing track of files
    already relocated to disk. Does not set group.status; callers decide
    (kept "pending" on a raised exception is load-bearing for the retry
    path, see test_web_api.py's test_confirm_partial_failure_leaves_group_pending).
    Appends the entry to *manifest* if given, and always returns it."""
    group.current_pick = keep_idx
    kept_path = group.paths[keep_idx]
    moved = []
    try:
        # is_symlink()/resolve() can themselves raise (e.g. EACCES on a
        # parent directory) -- kept inside the try so a failure here still
        # goes through the finally below instead of leaving *manifest*
        # without an entry for this group at all.
        kept_real = kept_path.resolve() if kept_path.is_symlink() else None
        for idx, path in enumerate(group.paths):
            if idx == keep_idx:
                continue
            if kept_real is not None and path.resolve() == kept_real:
                continue
            dest = _compute_dest(path, dest_dir, dry_run, recursive=recursive, scan_root=scan_root)
            if not dry_run:
                shutil.move(str(path), str(dest))
            moved.append({"from": str(path), "to": str(dest)})
    finally:
        entry = {"group": group_index, "kept": str(kept_path), "moved": moved, "dry_run": dry_run}
        if manifest is not None:
            manifest.append(entry)
    return entry


def auto_apply_groups(
    groups: list[Group],
    dest_dir: Path,
    dry_run: bool = False,
    recursive: bool = False,
    scan_root: Path | None = None,
) -> dict:
    """Apply every pending group's suggested (top-scored) pick, no review UI
    -- used by --auto. Doesn't second-guess close calls: the suggested pick
    is applied exactly as it would default to in interactive review.

    A group whose apply_group() raises partway through is recorded as a
    failure (its status stays "pending", the same state a partial move
    failure leaves it in during interactive review) and the run continues
    with the remaining groups -- one bad group (disk full, permission
    error) must not kill an unattended run and leave earlier groups'
    successfully-moved files unreported.

    bytes_reclaimed sums each moved file's size from group.results[idx]
    ["file_size"] (already populated by analyze_paths) *before* the move --
    the source path is gone from disk by the time apply_group returns, so
    re-stat()'ing it afterward would silently always read 0.

    Returns {"confirmed": int, "failed": int, "files_moved": int,
    "bytes_reclaimed": int, "failures": [{"group", "error", "files_moved",
    "bytes_moved"}, ...]}."""
    confirmed = 0
    failed = 0
    files_moved = 0
    bytes_reclaimed = 0
    failures = []
    manifest: list[dict] = []

    for i, group in enumerate(groups):
        if group.status != "pending":
            continue
        keep_idx = group.suggested_idx
        size_by_path = {str(p): r.get("file_size", 0) for p, r in zip(group.paths, group.results)}

        error = None
        pre_len = len(manifest)
        try:
            apply_group(group, i, keep_idx, dest_dir, dry_run, manifest, recursive=recursive, scan_root=scan_root)
        except Exception as exc:  # noqa: BLE001 -- one group's failure must not abort the rest
            error = exc

        # apply_group's finally appends an entry in the normal case, but
        # defend against a future change reintroducing a raise that happens
        # before that finally is reached (it must not be possible today --
        # see the comment in apply_group -- but indexing manifest[-1]
        # unconditionally would otherwise misattribute a previous group's
        # entry, or IndexError on group 0).
        moved = manifest[-1]["moved"] if len(manifest) > pre_len else []
        moved_bytes = sum(size_by_path.get(m["from"], 0) for m in moved)
        n_moved = len(moved)
        files_moved += n_moved

        if error is not None:
            failed += 1
            failures.append({"group": i, "error": str(error), "files_moved": n_moved, "bytes_moved": moved_bytes})
            continue

        group.status = "confirmed"
        confirmed += 1
        if not dry_run:
            bytes_reclaimed += moved_bytes

    return {
        "confirmed": confirmed,
        "failed": failed,
        "files_moved": files_moved,
        "bytes_reclaimed": bytes_reclaimed,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Interactive confirm/skip/re-pick primitives
# ---------------------------------------------------------------------------
# Take the manifest/group explicitly rather than reading state off a caller
# object, so the same manifest invariants (stays-pending-on-partial-failure,
# unapply-on-skip, clear-stale-entries-before-reapply, re-pick-after-confirm)
# live in exactly one place -- the one genuinely destructive, stateful path
# in the app.

def unapply(manifest: list[dict], group_index: int) -> None:
    """Reverse file moves for group *group_index* using *manifest* (the
    record apply_pick/apply_group left behind). Does NOT change the group's
    status -- the caller decides. Purely data-driven: doesn't need the Group
    object itself, only the manifest entry apply_group recorded for it."""
    entry = next((m for m in manifest if m["group"] == group_index), None)
    if not entry:
        return
    if entry["dry_run"]:
        # Dry-run moves never touch the filesystem (see apply_group), so
        # there's nothing to check for on disk -- the manifest entry itself
        # is the only state a dry-run "move" left behind, and reversing it
        # is just dropping that entry.
        manifest.remove(entry)
        return
    restored = []
    try:
        for moved in entry["moved"]:
            src = Path(moved["from"])
            dst = Path(moved["to"])
            if dst.exists() and not src.exists():
                # shutil.move, not Path.rename: dest_dir may point at a
                # different filesystem than the scanned directory, and a
                # plain rename raises OSError (EXDEV) cross-device where
                # shutil.move falls back to copy+remove -- the same reason
                # apply_group uses shutil.move rather than rename for the
                # forward move.
                shutil.move(str(dst), str(src))
                restored.append(moved)
    finally:
        # Keep tracking whatever wasn't restored (never just the fact that
        # *something* was restored) even if a move raised partway through,
        # so *manifest* always reflects real filesystem state for the rest
        # of this session -- the same invariant apply_group preserves on
        # the forward move (see test_manifest_crash_safety.py). Dropping
        # the whole entry here regardless of partial failure would
        # silently lose track of files still sitting in dest_dir/.
        remaining = [m for m in entry["moved"] if m not in restored]
        if remaining:
            entry["moved"] = remaining
        else:
            manifest.remove(entry)


def pick_needs_reapply(manifest: list[dict], group_index: int, group: Group) -> bool:
    """True for a confirmed group whose current_pick has since diverged from
    what's actually on disk (manifest's record of what got applied) -- i.e.
    re-applying would really re-move files, rather than being a no-op.
    Re-picking on an already confirmed group only stages current_pick;
    nothing moves until the caller explicitly re-applies."""
    entry = next((m for m in manifest if m["group"] == group_index), None)
    return entry is None or entry["kept"] != str(group.paths[group.current_pick])


def apply_pick(
    group: Group,
    group_index: int,
    keep_idx: int,
    dest_dir: Path,
    dry_run: bool,
    manifest: list[dict],
    recursive: bool = False,
    scan_root: Path | None = None,
) -> None:
    """Confirm *keep_idx* for *group*: clear any stale manifest entries left
    over from a previous partial failure for this group (unapply only
    processes the first match, so a sequence of partial failures can orphan
    entries), then apply_group the move, then mark the group confirmed.
    Unlike apply_group (which deliberately leaves group.status alone -- see
    its docstring), this DOES set it, since it's the caller apply_group
    expects to make that decision: only reached after every move in
    apply_group completed without raising, so the group is genuinely
    confirmed. If apply_group raises, group.status is left untouched
    (stays "pending"), same invariant apply_group's own callers rely on."""
    manifest[:] = [m for m in manifest if m["group"] != group_index]
    apply_group(
        group, group_index, keep_idx, dest_dir, dry_run, manifest,
        recursive=recursive, scan_root=scan_root,
    )
    group.status = "confirmed"


def cached_result(cache: dict, p: Path, st: os.stat_result) -> dict | None:
    entry = cache.get(str(p.resolve()))
    if entry is None or entry.get("mtime") != st.st_mtime_ns or entry.get("size") != st.st_size:
        return None
    try:
        result = dict(entry["result"])
        result["dimensions"] = tuple(result["dimensions"])
        return result
    except (KeyError, TypeError):
        return None


def store_result(cache: dict, p: Path, st: os.stat_result, result: dict) -> None:
    cache[str(p.resolve())] = {"mtime": st.st_mtime_ns, "size": st.st_size, "result": dict(result)}


def _analyze_one(path_str: str) -> dict | None:
    try:
        return analyze(path_str)
    except Exception:
        return None


def analyze_paths(paths: list[Path], cache: dict,
                  precomputed_stats: dict[Path, os.stat_result] | None = None,
                  progress_callback=None) -> dict[Path, dict]:
    """analyze() every path, reusing `cache` for files whose (mtime, size)
    haven't changed and running the rest through a thread pool (analyze()'s
    cv2/numpy calls release the GIL -- see the comments at
    THREAD_POOL_WORKERS's definition).

    If *precomputed_stats* is provided it is used instead of calling stat()
    again, and any path it doesn't cover is dropped (an unreadable file is
    tolerated here, as everywhere else in the scan).

    *progress_callback*, if given, is called as progress_callback(label,
    done, total) as each uncached item completes, instead of the default
    TTY-aware print via _print_progress -- see group_duplicates's matching
    parameter."""
    results: dict[Path, dict] = {}
    if precomputed_stats is not None:
        stats = precomputed_stats
        paths = [p for p in paths if p in stats]
    else:
        paths, stats = _stat_paths(paths)
    to_compute = []
    for p in paths:
        hit = cached_result(cache, p, stats[p])
        if hit is not None:
            results[p] = hit
        else:
            to_compute.append(p)

    if to_compute:
        total = len(to_compute)
        tty = sys.stdout.isatty()
        original_cv2_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
                for done, (p, r) in enumerate(
                    zip(to_compute, executor.map(_analyze_one, [str(p) for p in to_compute])), start=1
                ):
                    if r is not None:
                        store_result(cache, p, stats[p], r)
                        results[p] = r
                    if progress_callback is not None:
                        progress_callback("Analyzing", done, total)
                    else:
                        _print_progress("Analyzing", done, total, tty)
        finally:
            cv2.setNumThreads(original_cv2_threads)
        if progress_callback is None and tty:
            print()

    for p in results:
        results[p]["file_size"] = stats[p].st_size
    return results


def build_groups(
    directory: Path, threshold: int, recursive: bool = False, dest_dir: Path | None = None,
    progress_callback=None, hash_cache: dict | None = None, analyze_cache: dict | None = None,
) -> list[Group]:
    """*progress_callback*, if given, is passed straight through to
    group_duplicates/analyze_paths -- see their matching parameter. None
    (the default) preserves the CLI's existing TTY-aware stdout printing
    unchanged.

    *hash_cache*/*analyze_cache* are caller-owned and in-memory only: a scan
    writes nothing to *directory*. Passing dicts that outlive this call is
    what makes a rescan of an already-scanned library cheap (the web UI's
    Session does exactly that); omitting them means every path is recomputed
    from scratch, which is the right default for one-shot --auto runs. Both
    key on path + mtime + size, so entries stay valid across rescans with a
    different threshold or recursive setting -- neither affects a file's hash
    or its quality metrics."""
    paths = find_images(directory, recursive=recursive, exclude_dir=dest_dir)

    if hash_cache is None:
        hash_cache = {}
    if analyze_cache is None:
        analyze_cache = {}

    raw_groups = group_duplicates(paths, threshold, hash_cache, progress_callback=progress_callback)

    # Compute stats for the grouped files once and pass to analyze_paths,
    # rather than letting it call stat() again on files already stat()'d
    # during the hash phase (the same Path objects are reused).
    grouped_paths = [p for members in raw_groups for p in members]
    grouped_paths, grouped_stats = _stat_paths(grouped_paths)
    analyzed = analyze_paths(
        grouped_paths, analyze_cache, precomputed_stats=grouped_stats,
        progress_callback=progress_callback,
    )

    groups = []
    for members in raw_groups:
        # Skip files that failed analysis (not in analyzed dict).
        valid = [(p, analyzed[p]) for p in members if p in analyzed]
        if len(valid) < 2:
            continue  # no longer a duplicate group
        members, results = zip(*valid)
        members = list(members)
        results = list(results)
        score_group(results)

        # Reorder the group best-first rather than just recording which index
        # won, so the suggested file is always [1] -- leftmost column in the
        # web table, first preview on the stage. Reviewing is a
        # high-volume loop and the eye shouldn't have to hunt for the ★ in a
        # different position every group. Both lists are permuted by the same
        # order, so every index downstream (current_pick, the manifest,
        # /api/thumb's j, the digit-key shortcuts) stays consistent -- there
        # is no "original index" left to translate back to.
        #
        # sorted() is stable and find_images() returns sorted paths, so files
        # that tie on quality_score keep filename order relative to each
        # other; identical copies don't shuffle unpredictably between scans.
        order = sorted(range(len(results)), key=lambda i: -results[i]["quality_score"])
        members = [members[i] for i in order]
        results = [results[i] for i in order]
        suggested_idx = 0
        # bool(...): quality_score can be a numpy float64 (propagated from
        # analyze()'s metrics), and `<` against one produces numpy.bool_,
        # not Python bool -- `and` returns its second operand as-is rather
        # than coercing it, so close_call would silently end up numpy.bool_
        # too. That's fine for a plain truthy "if g.is_close_call" check,
        # but numpy.bool_ (unlike numpy.float64) isn't a subclass of its
        # Python equivalent and isn't JSON-serializable -- the web front
        # end's /api/state was the first consumer to actually hit this.
        close_call = bool(
            len(results) > 1
            and results[0]["quality_score"] - results[1]["quality_score"] < CLOSE_CALL_MARGIN
        )
        groups.append(
            Group(
                paths=members,
                results=results,
                # Not generated here: decoding+downscaling every group's images
                # up front stalls scan completion on large scans, and groups the
                # user never navigates to (e.g. quits early) would pay that
                # cost for nothing. Generated lazily on first request instead
                # (see duplicates_web._cached_render).
                thumbnails=None,
                suggested_idx=suggested_idx,
                current_pick=suggested_idx,
                is_close_call=close_call,
            )
        )
    return groups
