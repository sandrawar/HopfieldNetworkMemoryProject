"""
Hopfield memory on bitmaps of at least 200x300.

The weight matrix would be 60,000 by 60,000, so the local field is computed
directly from the patterns:

    h = (1/N) X^T (X s) - (P/N) s

This is Hebb's rule with a zero diagonal, without building W.
Cat photos are downloaded into experiments/cats, converted to ±1, and resized to 200x300.
"""

import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import RESULTS_DIR, matching_bits, save_report

CATS_DIR = Path(__file__).resolve().parent / "cats"
HEIGHT = 200
WIDTH = 300
NOISE = 0.1
MAX_STEPS = 8

CAT_URLS = (
    (
        "cat-1.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Cat03.jpg?width=640",
    ),
    (
        "cat-2.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Kittyply_edit1.jpg?width=640",
    ),
    (
        "cat-3.jpg",
        "https://commons.wikimedia.org/wiki/Special:FilePath/Cat_November_2010-1a.jpg?width=640",
    ),
)


def hebb_field(patterns, state):
    """Hebb field with a zero diagonal, computed without the weight matrix."""
    n_patterns, n_neurons = patterns.shape
    overlaps = patterns @ state
    return (patterns.T @ overlaps) / n_neurons - (n_patterns / n_neurons) * state


def synchronous_step(patterns, state):
    field = hebb_field(patterns, state)
    return np.where(field > 0, 1, np.where(field < 0, -1, state))


def evolve(patterns, state, max_steps=MAX_STEPS):
    state = np.asarray(state, dtype=int).copy()
    states = [state.copy()]

    for _ in range(max_steps):
        new_state = synchronous_step(patterns, state)

        if np.array_equal(new_state, state):
            return np.vstack(states), "stable"

        states.append(new_state.copy())
        state = new_state

    return np.vstack(states), "max_steps"


def download_cats():
    CATS_DIR.mkdir(parents=True, exist_ok=True)

    for file_name, url in CAT_URLS:
        destination = CATS_DIR / file_name

        if destination.exists():
            continue

        print(f"Pobieram {file_name}")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "HopfieldStudentProject/1.0"},
        )
        with urllib.request.urlopen(request) as response:
            destination.write_bytes(response.read())


def image_to_pattern(path):
    """Grayscale photo to a ±1 bitmap with height 200 and width 300."""
    image = Image.open(path).convert("L")
    image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float64)
    return np.where(pixels >= np.median(pixels), 1, -1).astype(int).reshape(-1)


def load_cat_patterns():
    download_cats()
    paths = sorted(CATS_DIR.glob("cat-*.jpg"))

    if len(paths) < 2:
        raise RuntimeError(f"W {CATS_DIR} nie ma zdjęć kotów.")

    patterns = np.vstack([image_to_pattern(path) for path in paths])
    return patterns, [path.name for path in paths]


def save_figure(patterns, names, original, corrupted, recovered):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    panels = list(patterns) + [original, corrupted, recovered]
    titles = [f"uczący: {name}" for name in names]
    titles += ["oryginał", f"uszkodzony {int(NOISE * 100)}%", "po zbieganiu"]
    columns = len(panels)
    fig, axes = plt.subplots(1, columns, figsize=(2.4 * columns, 3.2), constrained_layout=True)

    for axis, panel, title in zip(axes, panels, titles):
        axis.imshow(panel.reshape(HEIGHT, WIDTH), cmap="gray", vmin=-1, vmax=1)
        axis.set_title(title, fontsize=8)
        axis.axis("off")

    output = RESULTS_DIR / "cats-recovery.png"
    fig.savefig(output, dpi=80)
    plt.close(fig)
    return output


def main():
    print("=== DUŻE BITMAPY (KOTY) ===")
    patterns, names = load_cat_patterns()
    n_patterns, n_neurons = patterns.shape
    print(f"Wzorców: {n_patterns}, bitmapa {WIDTH}x{HEIGHT}, neuronów: {n_neurons}")

    for index, name in enumerate(names):
        restored = synchronous_step(patterns, patterns[index])
        bits = matching_bits(restored, patterns[index])
        print(f"{name}: po jednym kroku {bits}/{n_neurons} pikseli zgodnych z oryginałem")

    rng = np.random.default_rng(0)
    original = patterns[0].copy()
    corrupted = original.copy()
    number_to_flip = int(len(corrupted) * NOISE)
    corrupted[rng.choice(len(corrupted), number_to_flip, replace=False)] *= -1

    states, status = evolve(patterns, corrupted)
    recovered = states[-1]
    print(
        f"\nUszkodzono {number_to_flip} pikseli wzorca 1. "
        f"status={status}, kroki={len(states) - 1}"
    )
    print(f"Podobieństwo do oryginału: {matching_bits(recovered, original) / n_neurons:.1%}")

    figure_path = save_figure(patterns, names, original, corrupted, recovered)
    print(f"Rysunek: {figure_path}")


if __name__ == "__main__":
    with save_report("08_large_bitmaps.txt"):
        main()
