# Perf analysis: answering perf-handoff.md's open questions

Follow-up to `perf-handoff.md` (the vectorized hash sweep, `e7cd9a0`). That note
ranked four next steps. Measured on this machine (Apple M-series, 10 cores,
numpy 2.4.5, cv2 5.0.0), the ranking is **inverted**: the sweep is finished, and
`analyze()` — ranked last — is the actual bottleneck.

No code changed. Benchmarks used `tests/Test-image` (42 real photos, mostly
1200x1500) and 20 synthetic 4032x3024 JPEGs re-encoded from them.

## Headline

Per-image cost, threaded exactly as the pipeline runs it (10 workers,
`cv2.setNumThreads(1)`):

| phase | test-image (~1.8 MP) | 12 MP |
|---|---|---|
| hash (`_hash_one`: reduced decode + `phash_pair`) | 4.17 ms | 1.87 ms |
| `analyze()` (`_analyze_one`) | 30.6 ms | 36.3 ms |

**analyze is 8–19x the per-image cost of hashing.** So it overtakes the whole
hash phase once more than **5–14 %** of the library lands in duplicate groups
(5.2 % at 12 MP, 13.6 % at 1.8 MP). For a tool whose entire purpose is finding
duplicates, that fraction is the normal case, not the `g ≈ n` edge the handoff
assumed.

End-to-end on `tests/Test-image` (43 files, 42 grouped), warm process:

```
total 1.24s   Hashing 0.17s (4.2 ms/file)   Analyzing 0.99s (23.6 ms/file)
```

Analyze is 80 % of scan wall-clock there.

Composed for a plausible library (n = 20k, 15 % grouped, 12 MP photos):

```
hashing  37 s   sweep 0.9 s   analyze 108 s   -> analyze is 74% of the scan
```

## The sweep is done — close the LSH branch

Vectorized sweep, measured on synthetic random hashes:

| n | sweep |
|---|---|
| 5k | 0.05 s |
| 20k | 0.88 s |
| 50k | 5.49 s |

Against a 12 MP library the hash phase costs 1.87 ms/file, so the sweep only
matches hashing's cost at **n ≈ 130k**, and never approaches analyze's. Two
corrections to the handoff:

- Item 2 (pigeonhole LSH banding "if libraries routinely exceed ~100k") — its
  trigger never fires. At 100k the sweep is ~22 s against ~190 s of hashing and
  far more analyze. Not worth the exactness risk. **Drop it.**
- Item 3's claim that the sweep starts to dominate a cold scan at n = 20–30k is
  off by roughly 5x. It doesn't dominate anywhere in this tool's range.

## Where analyze's time actually goes

Single-threaded per image, which is how each worker runs it:

| step | test-image | 12 MP |
|---|---|---|
| `load_gray` (COLOR decode + cvtColor + float32) | 33.4 ms | 24.1 ms |
| `laplacian_sharpness` (INTER_AREA → 512² + Laplacian) | 27.4 ms | 49.6 ms |
| `effective_resolution` (INTER_AREA → 2048 + FFT + binning) | 54.1 ms | 75.3 ms |
| `noise_estimate` | 5.4 ms | 9.3 ms |
| `blockiness_score` | 5.2 ms | 7.8 ms |

The two expensive steps are the two that immediately downscale, and they each
pay their own full-resolution `INTER_AREA` pass (~42 ms of `effective_resolution`
at 12 MP is that resize alone). Inside `effective_resolution`, post-downscale:
`_power_spectrum` 24 ms, radius grid 6.5 ms, `bincount` 5.3 ms.

## Recommendations, ranked

### 1. Publish groups incrementally instead of analyzing everything up front

Biggest win, and the only one that needs a design decision.

Today `build_groups` analyzes every grouped file before the UI shows anything.
At n = 20k / 15 % that's ~108 s of dead time before the first decision. The
reviewer only ever looks at one group at a time.

Analyze **per group**, and push each group into the session as it finishes.
Time-to-first-group drops from ~146 s to ~38 s (hash phase + one group); the
rest streams in while the human reviews. `static/app.js` and `/api/state` already
handle a growing list via SSE progress, and the codebase already does exactly
this for thumbnails — `build_groups`' own comment ("decoding+downscaling every
group's images up front stalls scan completion") applies verbatim to analyze.

Constraint that decides the shape: `build_groups` doesn't just score, it
**permutes** `paths`/`results` best-first, sets `suggested_idx`/`is_close_call`,
and drops groups that fall below 2 valid members. CLAUDE.md: "there is no
'original index' to translate back to." So keep analyze → `score_group` →
permute → `len < 2` filter **atomic per group, before that group becomes
visible**. Never publish a group and reorder it afterwards; `?g=<generation>`
covers rescans, not in-place permutation. `--auto` still analyzes every group —
same total work, no behavior change.

### 2. Bring back the on-disk cache, fixing what got it removed

Both caches die with the process, so restarting the server re-pays the entire
scan. `daf7d65` ("Move the scan caches in-memory") removed the persistent
version for three specific reasons — all fixable:

| why it was removed | fix |
|---|---|
| littered the scanned library | write to `~/.cache/find_duplicates/<hash of root>.json`, never into the library |
| went stale across code changes (key is path+mtime+size, no code version) | stamp the file with a version covering `phash_pair`, `load_hash_gray` and `compare_image_quality`'s metrics; mismatch = ignore the file |
| unconditional write failed on a read-only / no-write NAS share | best-effort write in a `try`, a failure logs and scans normally |

Turns a repeat scan of an unchanged library from ~150 s into ~2 s. That is the
real "NAS box reviewed over several sessions" workflow. ~30 lines.

Do not ship this without the version stamp — `phash_pair`'s docstring
deliberately relies on "no cache file to hunt down and delete" when someone edits
the hash.

### 3. Two micro-wins inside analyze (~15–20 % combined, tiny diffs)

**Bit-identical, do it unconditionally.** `effective_resolution`'s radius grid
builds two full h×w `int64` arrays via `np.indices`. Broadcasting two 1-D arrays
gives a provably identical result 4x faster:

```python
dx = (np.arange(w) - cx).astype(np.float32)
dy = (np.arange(h) - cy).astype(np.float32)
r = np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2).astype(np.int32)
```

Verified `np.array_equal` against the current expression at three shapes;
6.6 ms → 1.7 ms at 2048x1536. Saves ~4 % of analyze.

**Not bit-exact, verify before shipping.** `load_gray` decodes color solely so
`brisque_score` can have BGR — and `brisque` is an optional import that stays
unresolved by design. Decoding `IMREAD_GRAYSCALE` when it's unavailable saves
~4 ms/image (11–15 % of analyze). It is a metric change: libjpeg's Y channel
isn't identical to `BGR2GRAY` after chroma upsampling. Measured across all 11
groups in `tests/Test-image`:

```
sharpness_normalized            median 0.2%   max  1.5%
effective_resolution_fraction   median 0.0%   max  0.3%
noise_sigma                     median 0.1%   max  1.4%
blockiness                      median 0.6%   max  8.8%
suggested-pick flips: 0/11 groups
```

### 4. Rejected: share one downscale between the two resize-heavy metrics

Feeding `effective_resolution`'s 2048-capped image into `laplacian_sharpness`
saves ~8 ms/image, but drifts sharpness up to 6.5 % **asymmetrically within a
group** — a >2048 px member gets the chained two-stage resize, a smaller one
doesn't. `score_group` min-max normalizes within the group, so exactly that
asymmetry can flip a ranking. Not worth 5 %.

### 5. Unmeasured on the real target: `Path.resolve()` on a network mount

`cached_hash`/`store_hash`/`cached_result`/`store_result` each call
`str(p.resolve())`, and recursive `find_images` calls it once more per file — 2–4
calls per file per scan. On local APFS that's 10.2 µs/path (~1 s at n = 20k,
noise). Over SMB/NFS `resolve()` is a per-path-component readlink round-trip and
could be far worse; it cannot be measured from here. Cheap to eliminate: resolve
once in `_stat_paths` and carry the string alongside the `stat_result` that
function already returns.

## What is no longer worth investigating

- LSH banding, BK-trees, any further sweep work (see above).
- Decode throughput in `_hash_one` (handoff item 3). At 1.87–4.17 ms/image
  threaded it is already a small slice of any scan with duplicates in it, and the
  risk sits on `MIN_REDUCED_DECODE_SIDE`'s hash-agreement invariant.
