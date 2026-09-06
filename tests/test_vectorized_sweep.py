"""Locks in that the vectorized popcount sweep in group_duplicates emits
exactly the clusters the old O(n^2) Python double loop did.

The sweep was replaced with a blocked numpy XOR + byte-popcount. This checks
the replacement is bit-identical: same edge predicate, same confirm gate,
same transitive closure, None rows excluded, block-offset arithmetic correct.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duplicates_core as dc


def brute(hash_list, threshold):
    """Reference: the original nested-loop sweep."""
    confirm = threshold <= dc.DEFAULT_HASH_THRESHOLD
    uf = dc.UnionFind(len(hash_list))
    for i, hi in enumerate(hash_list):
        if hi is None:
            continue
        for j in range(i + 1, len(hash_list)):
            hj = hash_list[j]
            if hj is None:
                continue
            if dc.hamming(hi[0], hj[0]) <= threshold and (
                    not confirm or dc.hamming(hi[1], hj[1]) <= dc.CONFIRM_HASH_THRESHOLD):
                uf.union(i, j)
    clusters = {}
    for i, h in enumerate(hash_list):
        if h is None:
            continue
        clusters.setdefault(uf.find(i), []).append(i)
    return sorted(sorted(c) for c in clusters.values() if len(c) > 1)


class _FakeStat:
    st_mtime_ns = 0
    st_size = 0


def sweep_clusters(hash_list, threshold):
    """Run the real group_duplicates sweep with decode/hash monkeypatched away.
    Every path is treated as uncached and 'hashed' straight from hash_list, so
    None entries flow through group_duplicates' own None handling."""
    paths = [Path(f"/x/{i}.jpg") for i in range(len(hash_list))]
    hmap = dict(zip(paths, hash_list))
    orig = (dc._stat_paths, dc.cached_hash, dc._hash_one)
    dc._stat_paths = lambda ps: (list(ps), {p: _FakeStat() for p in ps})
    dc.cached_hash = lambda cache, p, st: None
    dc._hash_one = lambda p: hmap[p]
    try:
        groups = dc.group_duplicates(paths, threshold, cache={})
    finally:
        dc._stat_paths, dc.cached_hash, dc._hash_one = orig
    idx = {p: i for i, p in enumerate(paths)}
    return sorted(sorted(idx[p] for p in g) for g in groups)


def flip(v, positions):
    for p in positions:
        v ^= (1 << p)
    return v


def main():
    rnd = random.Random(1234)

    # 1. popcount lookup vs int.bit_count over full-width XORs
    for _ in range(1000):
        x = rnd.getrandbits(256) ^ rnd.getrandbits(256)
        by = dc.np.frombuffer(x.to_bytes(32, "big"), dtype=dc.np.uint8)
        assert int(dc._POPCOUNT8[by].sum()) == x.bit_count()
    print("ok popcount lookup matches int.bit_count")

    b64, b256 = rnd.getrandbits(64), rnd.getrandbits(256)

    # 2. exact-collision cluster: 4 identical hashes -> one group of 4
    assert sweep_clusters([(b64, b256)] * 4, 10) == [[0, 1, 2, 3]]
    print("ok exact-collision cluster keeps all members")

    # 3. transitive chain A-B=8, B-C=8, A-C=16 (> threshold) -> all three grouped
    a = b64
    b = flip(a, range(0, 8))
    c = flip(b, range(8, 16))
    assert dc.hamming(a, c) == 16
    assert sweep_clusters([(a, b256), (b, b256), (c, b256)], 10) == [[0, 1, 2]]
    print("ok transitive chain merges across a >threshold gap")

    # 4. None rows never group, never pair with each other
    assert sweep_clusters([(b64, b256), None, (b64, b256), None], 10) == [[0, 2]]
    print("ok None hashes excluded")

    # 5. boundary: d64 == threshold groups, threshold+1 does not
    assert sweep_clusters([(b64, b256), (flip(b64, range(10)), b256)], 10) == [[0, 1]]
    assert sweep_clusters([(b64, b256), (flip(b64, range(11)), b256)], 10) == []
    print("ok <= threshold boundary is inclusive")

    # 6. confirm gate: d64=15 with a far confirm hash groups only when threshold>10
    hl = [(b64, b256), (flip(b64, range(15)), flip(b256, range(200)))]
    assert sweep_clusters(hl, 10) == []
    assert sweep_clusters(hl, 18) == [[0, 1]]
    print("ok confirm gate follows threshold")

    # 7. randomized cross-check vs brute force, with the block size forced to 3
    #    so start>0 runs and the `ii + start` offset is exercised.
    orig_budget = dc._SWEEP_XOR_BYTE_BUDGET
    try:
        for trial in range(40):
            n = rnd.randint(20, 60)
            dc._SWEEP_XOR_BYTE_BUDGET = 8 * n * 3  # -> block == 3
            seeds = [rnd.getrandbits(64) for _ in range(rnd.randint(1, 4))]
            hl = []
            for _ in range(n):
                if rnd.random() < 0.1:
                    hl.append(None)
                    continue
                h = flip(rnd.choice(seeds), rnd.sample(range(64), rnd.randint(0, 14)))
                hl.append((h, rnd.getrandbits(256)))
            threshold = rnd.choice([6, 10, 12, 18])
            assert sweep_clusters(hl, threshold) == brute(hl, threshold), f"trial {trial}"
    finally:
        dc._SWEEP_XOR_BYTE_BUDGET = orig_budget
    print("ok vectorized sweep matches brute force over 40 random trials (block=3)")


if __name__ == "__main__":
    main()
