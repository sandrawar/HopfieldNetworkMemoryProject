"""
How often the original pattern is recovered from an input with flipped bits.

For every dataset, both learning rules, and noise levels 10%, 20%, and 30%,
counts the fraction of trials that return exactly to the pattern.
"""

from common import DATASET_NAMES, NOISE_LEVELS, TRIALS, load_dataset, recovery_rate, save_report


def main():
    print("=== ODZYSK Z USZKODZONEGO WEJŚCIA ===")
    print(f"Prób na wzorzec: {TRIALS}. Tryb synchroniczny.")
    print("all = wszystkie wzorce, stable = tylko te, które są stanami stabilnymi.")
    header = f"{'zbiór':24} {'reguła':6} {'P':>4} {'N':>6}"

    for noise in NOISE_LEVELS:
        header += f" {int(noise * 100):>14}%"

    print(header)

    for name in DATASET_NAMES:
        print(f"Liczenie: {name}")
        patterns, _, _ = load_dataset(name)

        for rule in ("hebb", "oja"):
            row = f"{name:24} {rule:6} {len(patterns):4d} {patterns.shape[1]:6d}"

            for noise_index, noise in enumerate(NOISE_LEVELS):
                result = recovery_rate(
                    patterns,
                    rule,
                    noise,
                    seed=noise_index,
                )
                stable_only = result["stable_only"]
                stable_text = "  -- " if stable_only is None else f"{stable_only:5.0%}"
                row += f" {result['all']:5.0%}/{stable_text}"

            print(row)


if __name__ == "__main__":
    with save_report("02_damage_recovery.txt"):
        main()
