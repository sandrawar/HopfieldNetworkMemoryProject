"""
Compares recall on large-25x25.csv and large-25x50.csv.

These are the same drawings at two sizes. Noise, trial count, and learning
rules match, so a difference in the table comes from the bitmap size.
"""

from common import NOISE_LEVELS, TRIALS, load_dataset, recovery_rate, save_report

DATASETS = ("large-25x25.csv", "large-25x50.csv")


def main():
    print("=== 25x25 KONTRA 25x50 ===")
    print(f"Prób na wzorzec: {TRIALS}. Tryb synchroniczny.")
    print("Liczby to udane odzyskania: wszystkie wzorce / tylko stabilne.")

    for name in DATASETS:
        print(f"Liczenie: {name}")
        patterns, height, width = load_dataset(name)
        print(
            f"\n{name}: {len(patterns)} wzorców, "
            f"bitmapa {width}x{height}, neuronów {patterns.shape[1]}"
        )

        for rule in ("hebb", "oja"):
            print(f"  {rule}")

            for noise_index, noise in enumerate(NOISE_LEVELS):
                result = recovery_rate(
                    patterns,
                    rule,
                    noise,
                    seed=100 + noise_index,
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
                    f"średnie podobieństwo {result['mean_similarity']:.0%}, "
                    f"stabilnych wzorców {result['n_stable']}/{len(patterns)}"
                )


if __name__ == "__main__":
    with save_report("03_size_25x25_vs_25x50.txt"):
        main()
