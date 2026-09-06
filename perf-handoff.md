# Perf handoff: vectorizing the O(n²) hash sweep

Status: shipped in `e7cd9a0`. This note is for whoever picks up scan performance next.

## Problem

`group_duplicates` (`duplicates_core.py`) proposes duplicate pairs with an
all-pairs Hamming comparison of 64-bit perceptual hashes. It was a pure-Python
double loop:

```python
for i in range(n):
    for j in range(i + 1, n):
        if hamming(h[i][0], h[j][0]) <= threshold and (not confirm or ...):
            uf.union(i, j)
```

Single-threaded, ~8M pairs/sec. Costs measured / extrapolated on this machine:

| n | pairs | old sweep |
|------|-------|-----------|
| 5k | 12.5M | ~1.5s |
| 20k | 200M | ~24s |
| 50k | 1.25B | ~150s |

It is paid **in full on every rescan**: `hash_cache` and `analyze_cache`
deliberately survive a rescan (CLAUDE.md documents this), so on the web UI's
rescan-from-control-panel flow the sweep is ~100% of wall-clock while
everything else is near-instant.

## What shipped

Replaced only the double loop (`duplicates_core.py`, the `uf = UnionFind(...)`
block). Everything above (stat, cache, threaded hashing, the
`confirm = threshold <= DEFAULT_HASH_THRESHOLD` line) and below (cluster dict +
`len > 1` filter) is untouched.

- Pack the 64-bit hashes into an `(n,)` `uint64` array, 256-bit confirm hashes
  into `(n, 4)` `uint64`. Big-endian bytes viewed as `uint64` — lane order and
  endianness are irrelevant to XOR and popcount.
- Blocked over rows (`_SWEEP_XOR_BYTE_BUDGET`, 32 MB) to bound the XOR temp.
- Per block: broadcast XOR → `_popcount` → `np.nonzero(d <= threshold)` →
  mask to the upper triangle and drop `None` rows → run the 256-bit confirm on
  the survivors only → `uf.union` each surviving pair.
- `_popcount` uses `np.bitwise_count` (numpy ≥ 2.0) with a byte-lookup-table
  fallback (`_POPCOUNT8`) for older numpy.

~35 net lines in one file. No new dependencies. `install.sh` unchanged.

## Why this approach

Three candidates were designed and judged (numpy popcount / LSH banding /
BK-tree). Popcount won on this codebase, not just on diff size:

- **Exact.** Still all-pairs, same integer Hamming metric, same edge
  predicate, same confirm gate. Output is provably bit-identical to the old
  code at every threshold. Verified: diffed old vs new `build_groups` on
  `tests/Test-image/` (identical), plus a brute-force cross-check in
  `tests/test_vectorized_sweep.py` over 40 randomized trials.
- **LSH banding** (bucket by sub-hash bands, compare only bucket-mates) is
  truly sub-quadratic but its pathological case is real here: near-black
  frames, solid backgrounds and flat screenshots share low-frequency DCT
  structure, so they land in the same band bucket in bulk → `O(k²)` Python
  set operations per bucket, plausibly *slower* than vectorized popcount on
  that same library. It also degrades exactly when the user raises
  `--threshold` (the documented "find more" knob) and needs careful
  pigeonhole reasoning to stay exact.
- **BK-tree** is exact and sublinear per query for small radius, but it is
  Python-object-heavy — no numpy vectorization — and a bigger, trap-prone
  diff.
- **Hybrid** (popcount small-n, LSH large-n) adds a second code path and a
  cutoff constant to tune, for a ceiling this tool's use case (personal
  near-dup photo library on a NAS) does not reach. YAGNI.

## Results (warm cache, this machine)

| n | old | new | speedup |
|------|-------|-------|---------|
| 3k | ~0.4s | 0.15s | ~3x |
| 10k | ~4s | 1.6s | ~2.5x |
| 25k | ~40s | 1.4s | ~7x (grows with n) |

`np.bitwise_count` scales far better than the fallback. The XOR-temp working
set peaks ~80–100 MB regardless of n with the 32 MB byte budget.

## Limitations / ceiling

- **Still O(n²).** This is a ~7–30× constant-factor win, not an algorithmic
  one. Practical ceiling ~80–120k images (~2–5 min at n=100k).
- **`ii`/`jj` candidate arrays** scale with matches found — tiny for a normal
  library. A user who raises `--threshold` very high *and* has a huge library
  (confirm off, most pairs pass the 64-bit gate) can push these toward n²
  `int64`. Per-block iteration bounds it partially; the old code only avoided
  the blowup by being slow.
- **Does not help cold first scans below ~25k images.** Those are bound by
  native image decode in the hashing `ThreadPoolExecutor` (`_hash_one`),
  already parallel across cores. Crossover where the sweep starts to dominate
  a cold scan is ~n=20–30k.

## Handoff: what to analyze next

Ranked by expected value:

1. **Is O(n²) actually a problem for the real user?** The whole use case is a
   personal library on a NAS/headless box. If libraries are consistently
   < 30k images, this change already removes the pain and nothing more is
   needed. Get a real library-size distribution before investing further.
2. **If libraries routinely exceed ~100k:** implement pigeonhole-exact LSH
   banding as a prefilter *in front of* the popcount sweep (not replacing it).
   With `B = threshold + 1` contiguous bands, any matching pair must share ≥ 1
   identical band → zero false negatives. Feed bucket-mates into the existing
   vectorized path. Watch the giant-shared-bucket case (flat images) — cap
   bucket size and fall back to the full popcount sweep for oversized buckets.
3. **Cold-scan decode throughput** (`_hash_one` / `load_hash_gray`): the other
   bottleneck. Process pool is explicitly rejected in CLAUDE.md (macOS spawn
   crash). Headroom would come from a coarser reduced-decode
   (`MIN_REDUCED_DECODE_SIDE`) — but that constant is empirically tuned
   against `tests/Test-image/` and touches the 64-bit-hash agreement invariant
   between decode paths. High risk, measure first.
4. **`analyze_paths`** only matters if a large fraction of the library is
   duplicated (`g ≈ n`). Not on the critical path for typical libraries.

## Implementation traps for the next editor

- The 256-bit confirm hash must **never** be used to bucket or prune
  candidates — only to confirm pairs the 64-bit gate already proposed.
  Using it to prune changes semantics silently when `confirm` is off
  (`threshold > 10`).
- Any index structure (LSH bands, BK-tree radius) must be parameterized to
  **this scan's** `threshold`, not a value baked for 10/64. The UI tells users
  to raise it.
- Grouping is the transitive closure (UnionFind) of the edge set. Dropping one
  bridging edge can split a component — near-miss edges carry disproportionate
  weight. Any candidate-generation optimization must be a **superset** of the
  exhaustive edge set.
- `uf.union` args come from `.tolist()` so they are Python ints — keep it that
  way (numpy scalars leak into the JSON API elsewhere; see CLAUDE.md).
- `tests/test_vectorized_sweep.py` shrinks `_SWEEP_XOR_BYTE_BUDGET` to force
  `block == 3` and exercise the `ii + start` offset. Keep that knob.
