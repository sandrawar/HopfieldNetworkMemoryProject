"""Shared dataset loading and recall measurement for the experiments."""

import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hopfield import HopfieldNetwork

DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DATASET_NAMES = (
    "small-7x7.csv",
    "animals-14x9.csv",
    "letters-abc-8x12.csv",
    "letters-14x20.csv",
    "OCRA-12x30-cut.csv",
    "large-25x25.csv",
    "large-25x25.plus.csv",
    "large-25x50.csv",
)

class _Tee:
    """Writes text to the console and immediately to a file."""

    def __init__(self, file, stdout):
        self.file = file
        self.stdout = stdout

    def write(self, data):
        self.stdout.write(data)
        self.stdout.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


@contextmanager
def save_report(filename):
    """Saves the whole experiment printout under experiments/results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    previous = sys.stdout
    report_file = path.open("w", encoding="utf-8")
    sys.stdout = _Tee(report_file, previous)
    print(f"Wyniki: {path}")

    try:
        yield path
    finally:
        sys.stdout = previous
        report_file.close()
        print(f"Zapisano: {path}", flush=True)


NOISE_LEVELS = (0.1, 0.2, 0.3)
TRIALS = 10
OJA_LEARNING_RATE = 0.01
OJA_EPOCHS = 200


def load_patterns(file_path):
    """Loads patterns from CSV. One row is one pattern with values ±1."""
    patterns = np.loadtxt(file_path, delimiter=",", dtype=int)
    return np.atleast_2d(patterns)


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


def load_dataset(name):
    """Returns (patterns, height, width) for a file in datasets."""
    path = DATASETS_DIR / name
    patterns = load_patterns(path)
    height, width = bitmap_shape(path, patterns.shape[1])
    return patterns, height, width


def train(patterns, rule):
    """Trains a new network with Hebb's rule or Oja's rule."""
    network = HopfieldNetwork()

    if rule == "hebb":
        network.train_hebb(patterns)
    elif rule == "oja":
        network.train_oja(
            patterns,
            learning_rate=OJA_LEARNING_RATE,
            epochs=OJA_EPOCHS,
        )
    else:
        raise ValueError("rule musi być 'hebb' albo 'oja'.")

    return network


def corrupt(pattern, fraction, rng):
    """Flips a fraction of the bits. The count is the integer part of the length."""
    corrupted = np.array(pattern, copy=True)
    number_to_flip = int(len(corrupted) * fraction)

    if number_to_flip == 0:
        return corrupted

    indices = rng.choice(len(corrupted), number_to_flip, replace=False)
    corrupted[indices] *= -1
    return corrupted


def matching_bits(left, right):
    """Number of matching coordinates of two ±1 vectors."""
    return int(np.sum(np.asarray(left) == np.asarray(right)))


def classify_attractor(state, patterns):
    """
    Classifies a state as a stored pattern, its negation, or another attractor.
    The negation of a stable pattern is stable too, because the local field
    changes sign together with the state.
    """
    dots = patterns @ state
    n_neurons = patterns.shape[1]
    exact = np.where(dots == n_neurons)[0]
    negated = np.where(dots == -n_neurons)[0]

    if len(exact):
        return "pattern", int(exact[0]), n_neurons

    if len(negated):
        return "negation", int(negated[0]), n_neurons

    nearest = int(np.argmax(dots))
    bits = int((n_neurons + dots[nearest]) / 2)
    return "spurious", nearest, bits


def recovery_rate(patterns, rule, noise, trials=TRIALS, mode="synchronous", seed=0):
    """
    Fraction of trials in which a damaged pattern returns exactly to the original.

    Counts trials over all patterns and, separately, over the stable ones only.
    Returns a dict with keys all, stable_only, mean_similarity, n_stable.
    """
    network = train(patterns, rule)
    stable_mask = np.array([network.is_stable(pattern) for pattern in patterns])
    rng = np.random.default_rng(seed)

    hits_all = 0
    hits_stable = 0
    trials_stable = 0
    similarities = []

    for pattern, is_stable in zip(patterns, stable_mask):
        for _ in range(trials):
            corrupted = corrupt(pattern, noise, rng)
            states, _, _ = network.evolve(
                corrupted,
                mode=mode,
                max_steps=20,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            final = states[-1]
            hit = bool(np.array_equal(final, pattern))
            hits_all += int(hit)
            similarities.append(matching_bits(final, pattern) / len(pattern))

            if is_stable:
                hits_stable += int(hit)
                trials_stable += 1

    total = len(patterns) * trials
    stable_only = None if trials_stable == 0 else hits_stable / trials_stable

    return {
        "all": hits_all / total,
        "stable_only": stable_only,
        "mean_similarity": float(np.mean(similarities)),
        "n_stable": int(np.sum(stable_mask)),
    }
