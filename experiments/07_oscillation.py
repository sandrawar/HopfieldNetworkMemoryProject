"""
A training set and an input for which synchronous mode oscillates between two states.

The network learns one pattern: a white 4x4 square. The input is a checkerboard.
With an even number of neurons the checkerboard bits sum to zero, so the local
field has the opposite sign to the state. The network jumps to the other
checkerboard and then back.

Asynchronous mode does not increase the energy, so the same input settles at a fixed point.
"""

import numpy as np

from common import save_report, train

HEIGHT = 4
WIDTH = 4


def white_square():
    return np.ones(HEIGHT * WIDTH, dtype=int)


def checkerboard():
    bitmap = np.indices((HEIGHT, WIDTH)).sum(axis=0) % 2
    return np.where(bitmap == 0, 1, -1).reshape(-1)


def _as_text(pattern):
    bitmap = np.where(pattern.reshape(HEIGHT, WIDTH) > 0, "#", ".")
    return "\n".join("".join(row) for row in bitmap)


def main():
    print("=== OSCYLACJA SYNCHRONICZNA ===")
    patterns = np.atleast_2d(white_square())
    probe = checkerboard()
    network = train(patterns, "hebb")

    print("Zbiór uczący, jeden wzorzec 4x4:")
    print(_as_text(patterns[0]))
    print("\nWejście:")
    print(_as_text(probe))

    for mode in ("synchronous", "asynchronous"):
        states, energies, status = network.evolve(
            probe,
            mode=mode,
            max_steps=20,
            seed=0,
        )
        print(f"\n{mode}: status={status}, kroki={len(states) - 1}")
        print("energie:", " -> ".join(f"{value:.2f}" for value in energies))

        for index, state in enumerate(states):
            label = "start" if index == 0 else f"krok {index}"
            print(f"\n{label}")
            print(_as_text(state))


if __name__ == "__main__":
    with save_report("07_oscillation.txt"):
        main()
