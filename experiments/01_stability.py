"""
How many patterns in each dataset stay stable after Hebb and after Oja.

A stored pattern counts as learned when one synchronous step leaves it unchanged.
"""

from common import DATASET_NAMES, load_dataset, save_report, train


def main():
    print("=== STABILNOŚĆ WZORCÓW UCZĄCYCH ===")
    print(f"{'zbiór':24} {'P':>4} {'N':>6} {'P/N':>8} {'Hebb':>10} {'Oja':>10}")

    for name in DATASET_NAMES:
        print(f"Liczenie: {name}")
        patterns, _, _ = load_dataset(name)
        n_patterns, n_neurons = patterns.shape
        counts = {}

        for rule in ("hebb", "oja"):
            network = train(patterns, rule)
            counts[rule] = sum(network.is_stable(pattern) for pattern in patterns)

        print(
            f"{name:24} {n_patterns:4d} {n_neurons:6d} "
            f"{n_patterns / n_neurons:8.3f} "
            f"{counts['hebb']:4d}/{n_patterns:<5d} "
            f"{counts['oja']:4d}/{n_patterns:<5d}"
        )


if __name__ == "__main__":
    with save_report("01_stability.txt"):
        main()
