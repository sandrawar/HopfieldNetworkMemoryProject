"""
What a trained network does when the input is a random ±1 vector.

The final state is a stored pattern, its negation, or some other attractor.
"""

import numpy as np

from common import classify_attractor, load_dataset, save_report, train

DATASETS = (
    "large-25x25.csv",
    "animals-14x9.csv",
    "letters-14x20.csv",
)
PROBES = 20
SEED = 0


def main():
    print("=== LOSOWE WEJŚCIE ===")
    print(f"Reguła Hebba, tryb synchroniczny, prób na zbiór: {PROBES}.")

    for name in DATASETS:
        patterns, _, _ = load_dataset(name)
        network = train(patterns, "hebb")
        rng = np.random.default_rng(SEED)
        counts = {"pattern": 0, "negation": 0, "spurious": 0}

        print(f"\n{name}: P={len(patterns)}, N={patterns.shape[1]}")

        for probe_index in range(PROBES):
            probe = rng.choice(np.array([-1, 1]), size=patterns.shape[1])
            states, _, status = network.evolve(probe, mode="synchronous", max_steps=30)
            kind, index, bits = classify_attractor(states[-1], patterns)
            counts[kind] += 1
            n_neurons = patterns.shape[1]

            if kind == "pattern":
                detail = f"dokładnie wzorzec {index + 1}"
            elif kind == "negation":
                detail = f"dokładna negacja wzorca {index + 1}"
            else:
                detail = f"najbliższy wzorzec {index + 1}, zgodnych bitów {bits}/{n_neurons}"

            print(f"  próba {probe_index + 1:2d}: {status:8} -> {detail}")

        print(
            f"  razem: wzorzec {counts['pattern']}, "
            f"negacja {counts['negation']}, "
            f"inny stan {counts['spurious']}"
        )


if __name__ == "__main__":
    with save_report("05_random_input.txt"):
        main()
