"""
A similar pattern set against a diverse one, at the same cardinality
and the same number of neurons.

Both sets have 6 patterns from large-25x25.plus.csv (N = 625).
Among every combination of six patterns, the subsets with the largest and
the smallest mean pairwise pixel agreement are kept.
"""

from itertools import combinations

import numpy as np

from common import NOISE_LEVELS, TRIALS, load_dataset, recovery_rate, save_report, train

SUBSET_SIZE = 6


def pairwise_agreement(patterns):
    """Fraction of agreeing pixels for every pair of distinct patterns."""
    count = patterns.shape[1]
    products = patterns @ patterns.T
    upper = np.triu_indices(len(patterns), 1)
    return (count + products[upper]) / (2 * count)


def extreme_subsets(patterns):
    """
    Returns (indexes of the most similar subset, indexes of the least similar).
    A tie keeps the earlier combination in lexicographic order.
    """
    most_similar = None
    least_similar = None

    for indexes in combinations(range(len(patterns)), SUBSET_SIZE):
        agreement = pairwise_agreement(patterns[list(indexes)])
        mean_agreement = float(agreement.mean())

        if most_similar is None or mean_agreement > most_similar[0]:
            most_similar = (mean_agreement, indexes)

        if least_similar is None or mean_agreement < least_similar[0]:
            least_similar = (mean_agreement, indexes)

    return most_similar[1], least_similar[1]


def describe_recall(patterns):
    """After Hebb: how many distinct final states the training patterns reach."""
    network = train(patterns, "hebb")
    finals = []

    for pattern in patterns:
        states, _, status = network.evolve(pattern, mode="synchronous", max_steps=20)
        finals.append((status, states[-1]))

    keys = []
    groups = []

    for status, final in finals:
        key = final.tobytes()

        if key not in keys:
            keys.append(key)
            groups.append([])

        groups[keys.index(key)].append(status)

    print(f"  Hebb, liczba różnych stanów końcowych: {len(groups)}")

    for index, members in enumerate(groups, start=1):
        count = len(members)
        noun = "wzorzec" if count == 1 else "wzorce" if count < 5 else "wzorców"
        print(f"    stan {index}: {count} {noun}")


def main():
    patterns, _, _ = load_dataset("large-25x25.plus.csv")
    similar_index, diverse_index = extreme_subsets(patterns)
    groups = (
        ("podobne", similar_index),
        ("zróżnicowane", diverse_index),
    )

    print("=== PODOBNE KONTRA ZRÓŻNICOWANE ===")
    print(
        "Plik large-25x25.plus.csv. Oba podzbiory mają "
        f"P={SUBSET_SIZE}, N={patterns.shape[1]}. Prób na wzorzec: {TRIALS}."
    )

    for label, indexes in groups:
        subset = patterns[list(indexes)]
        agreement = pairwise_agreement(subset)
        numbers = ", ".join(str(index + 1) for index in indexes)
        print(
            f"\n{label}: wzorce {numbers}; "
            f"zgodność średnia {agreement.mean():.1%}, "
            f"mediana {np.median(agreement):.1%}, "
            f"zakres {agreement.min():.1%}–{agreement.max():.1%}"
        )
        describe_recall(subset)

        for rule in ("hebb", "oja"):
            print(f"  {rule}")

            for noise_index, noise in enumerate(NOISE_LEVELS):
                result = recovery_rate(
                    subset,
                    rule,
                    noise,
                    seed=200 + noise_index,
                )
                stable_only = result["stable_only"]
                stable_text = (
                    "nie ma wzorców stabilnych"
                    if stable_only is None
                    else f"{stable_only:.0%}"
                )
                print(
                    f"    szum {int(noise * 100):2d}%: "
                    f"wszystkie {result['all']:.0%}, "
                    f"wśród stabilnych {stable_text}, "
                    f"stabilnych wzorców {result['n_stable']}/{SUBSET_SIZE}"
                )


if __name__ == "__main__":
    with save_report("04_similar_vs_diverse.txt"):
        main()
