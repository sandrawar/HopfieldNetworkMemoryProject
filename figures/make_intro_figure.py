import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from hopfield import HopfieldNetwork

patterns = np.loadtxt(root / "datasets" / "large-25x25.csv", delimiter=",", dtype=int)
network = HopfieldNetwork()
network.train_hebb(patterns)

house = patterns[1].copy()
damaged = house.copy()
rng = np.random.default_rng(0)
flip = rng.choice(len(damaged), int(0.20 * len(damaged)), replace=False)
damaged[flip] *= -1
states, _, status = network.evolve(damaged, mode="synchronous", max_steps=20, seed=0)
if status != "stable" or not np.array_equal(states[-1], house):
    raise SystemExit(f"nieoczekiwany przebieg: {status}")

names = ["Słońce", "Dom", "Szachownica", "Strzałka", "Paski", "Litery"]
recall = [
    (house, "Dom"),
    (damaged, "Wejście, 20% uszkodzeń"),
    (states[-1], "Po 1 kroku"),
]

fig = plt.figure(figsize=(10.2, 4.6))
grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.15], hspace=0.35, wspace=0.18)

for index, (pattern, name) in enumerate(zip(patterns, names)):
    axis = fig.add_subplot(grid[0, index])
    axis.imshow(pattern.reshape(25, 25), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    axis.set_title(name, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])

for index, (pattern, name) in enumerate(recall):
    axis = fig.add_subplot(grid[1, 2 * index : 2 * index + 2])
    axis.imshow(pattern.reshape(25, 25), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    axis.set_title(name, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])

fig.savefig(root / "figures" / "dzialanie.png", dpi=140, bbox_inches="tight")
print("zapisano", len(states) - 1, "krok")
