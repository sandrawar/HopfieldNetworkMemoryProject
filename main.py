
import re

import numpy as np
from pathlib import Path

from hopfield import HopfieldNetwork

from visualization import finish_figures, show_convergence, show_patterns



def load_patterns(file_path):
    """Loads patterns from a CSV file."""
    patterns = np.loadtxt(file_path, delimiter=",", dtype=int)
    patterns = np.atleast_2d(patterns)
    return patterns


def bitmap_shape(file_path, n_neurons):
    """
    Reads the bitmap size from the file name. The order is width x height,
    so animals-14x9.csv means 14 columns and 9 rows.
    """
    match = re.search(r"(\d+)x(\d+)", Path(file_path).name)

    if match is None:
        raise ValueError(
            f"W nazwie {Path(file_path).name} nie ma rozmiaru w postaci 14x9."
        )

    width = int(match.group(1))
    height = int(match.group(2))

    if height * width != n_neurons:
        raise ValueError(
            f"Nazwa podaje {width}x{height} = {height * width} pikseli, "
            f"a wzorzec ma {n_neurons} neuronów."
        )

    return height, width


def main():
    project_dir = Path(__file__).parent
    dataset_path = project_dir / "datasets" / "large-25x25.csv"

    patterns = load_patterns(dataset_path)
    height, width = bitmap_shape(dataset_path, patterns.shape[1])

    print("=== PAMIĘĆ HOPFIELDA ===")
    print(f"Plik: {dataset_path.name}")
    print(f"Liczba wzorców: {len(patterns)}")
    print(f"Bitmapa: {width}x{height} (szerokość x wysokość)")
    print(f"Liczba neuronów: {patterns.shape[1]}")

    show_patterns(
        patterns,
        height=height,
        width=width,
        title=f"Wszystkie wzorce — {dataset_path.name}",
    )

    network = HopfieldNetwork()
    network.train_hebb(patterns)

    print("\n=== TEST STABILNOŚCI (Hebb) ===")

    for i, pattern in enumerate(patterns):
        stable = network.is_stable(pattern)
        print(f"Wzorzec {i + 1}: {'STABILNY' if stable else 'NIESTABILNY'}")

    print("\n=== TEST STABILNOŚCI (Oja) ===")

    oja_network = HopfieldNetwork()
    oja_network.train_oja(patterns, learning_rate=0.01, epochs=200)

    for i, pattern in enumerate(patterns):
        stable = oja_network.is_stable(pattern)
        print(f"Wzorzec {i + 1}: {'STABILNY' if stable else 'NIESTABILNY'}")

    print("\n=== TEST USZKODZENIA ===")

    original = patterns[0].copy()
    corrupted = original.copy()

    # Flip 20% of the neurons.
    number_to_flip = int(len(corrupted) * 0.20)
    indices = np.random.choice(
        len(corrupted),
        number_to_flip,
        replace=False
    )

    corrupted[indices] *= -1

    print(f"Uszkodzono neuronów: {number_to_flip}")

    for mode in ("synchronous", "asynchronous"):
        states, energies, status = network.evolve(
            corrupted,
            mode=mode,
            max_steps=20,
            seed=0,
        )
        recovered = states[-1]
        similarity = np.mean(recovered == original) * 100
        energy_text = " -> ".join(f"{value:.1f}" for value in energies)

        print(f"\n{mode}")
        print(f"status: {status}, kroki: {len(states) - 1}")
        print(f"energia: {energy_text}")
        print(f"podobieństwo do oryginału: {similarity:.2f}%")

        show_convergence(
            states,
            height=height,
            width=width,
            title=f"Uszkodzony wzorzec 1 — {mode} ({status})",
        )

    print("\n=== KROKI DLA KAŻDEGO WZORCA ===")

    for index, pattern in enumerate(patterns):
        states, energies, status = network.evolve(
            pattern,
            mode="synchronous",
            max_steps=20,
        )
        final = states[-1]
        overlaps = patterns @ final
        nearest = int(np.argmax(overlaps))
        matching_bits = int((patterns.shape[1] + overlaps[nearest]) / 2)
        shown = states

        if len(states) == 1:
            shown = np.vstack([states[0], states[0]])

        print(
            f"Wzorzec {index + 1}: status={status}, kroki={len(states) - 1}, "
            f"najbliższy zapisany wzorzec={nearest + 1} "
            f"(zgodnych bitów: {matching_bits}/{patterns.shape[1]})"
        )

        show_convergence(
            shown,
            height=height,
            width=width,
            title=f"Wzorzec {index + 1} — synchronicznie ({status})",
        )

    finish_figures()


if __name__ == "__main__":
    main()
