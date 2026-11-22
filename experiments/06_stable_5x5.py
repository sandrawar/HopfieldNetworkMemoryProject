"""
As large a set as possible of length-25 vectors in which every pattern is stable.

Two ±1 vectors of odd length cannot be orthogonal: their dot product is odd.
The search therefore keeps |x·y| <= 1 for every pair. At N = 25 that condition
keeps every pattern stable for at most 13 patterns.
"""

import numpy as np

from common import RESULTS_DIR, save_report, train

N_NEURONS = 25
HEIGHT = 5
WIDTH = 5
TRIALS_PER_SEED = 150_000
SEEDS = range(12)


def _dots_are_small(left, right):
    shared = len(set(left) & set(right))
    distance = (len(left) - shared) + (len(right) - shared)
    return abs(N_NEURONS - 2 * distance) <= 1


def _search_masks(seed):
    """Set of positions of the negative bits. Pairs have dot product ±1."""
    rng = np.random.default_rng(seed)
    chosen = []

    for _ in range(TRIALS_PER_SEED):
        negatives = 12 if rng.random() < 0.5 else 13
        mask = tuple(sorted(rng.choice(N_NEURONS, negatives, replace=False).tolist()))

        if any(not _dots_are_small(mask, previous) for previous in chosen):
            continue

        chosen.append(mask)

    return chosen


def build_patterns():
    """
    An all-ones image plus as many patterns as possible with pairwise dot
    product ±1. At this length that dot product cannot flip a stored sign
    while the set has at most 13 patterns.
    """
    best_masks = max((_search_masks(seed) for seed in SEEDS), key=len)
    patterns = [np.ones(N_NEURONS, dtype=int)]

    for mask in best_masks:
        pattern = np.ones(N_NEURONS, dtype=int)
        pattern[list(mask)] = -1
        patterns.append(pattern)

    return np.vstack(patterns)


def _as_text(pattern):
    bitmap = np.where(pattern.reshape(HEIGHT, WIDTH) > 0, "#", ".")
    return "\n".join("".join(row) for row in bitmap)


def main():
    print("=== ZBIÓR 5x5, WSZYSTKIE WZORCE STABILNE ===")
    patterns = build_patterns()
    network = train(patterns, "hebb")
    stable = [network.is_stable(pattern) for pattern in patterns]
    dots = patterns @ patterns.T
    off_diagonal = dots[~np.eye(len(patterns), dtype=bool)]

    print(f"Liczba wzorców: {len(patterns)}")
    print(f"Stabilnych: {sum(stable)}/{len(patterns)}")
    print(f"Największy |iloczyn| par różnych wzorców: {int(np.max(np.abs(off_diagonal)))}")

    for index, pattern in enumerate(patterns):
        print(f"\nWzorzec {index + 1} ({'stabilny' if stable[index] else 'NIESTABILNY'})")
        print(_as_text(pattern))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "stable-5x5.csv"
    np.savetxt(output, patterns, fmt="%d", delimiter=",")
    print(f"\nZapisano: {output}")


if __name__ == "__main__":
    with save_report("06_stable_5x5.txt"):
        main()
