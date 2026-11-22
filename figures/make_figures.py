import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

names = ["7×7", "zwierzęta", "abc", "alfabet", "OCRA", "25×25", "25×25+", "25×50"]
hebb = [3, 2, 0, 0, 1, 5, 1, 0]
oja = [3, 2, 0, 0, 0, 5, 1, 0]
total = [5, 6, 3, 26, 7, 6, 15, 6]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(10, 4.2))
width = 0.38
ax.bar(x - width / 2, [h / t for h, t in zip(hebb, total)], width, label="Hebb", color="#3d5a80")
ax.bar(x + width / 2, [h / t for h, t in zip(oja, total)], width, label="Oja", color="#98c1d9")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("Udział stabilnych wzorców")
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("Stabilne wzorce uczące")
fig.tight_layout()
fig.savefig(out / "stabilnosc.png")
plt.close()

rows = ["7×7", "zwierzęta", "abc", "alfabet", "OCRA", "25×25", "25×25+", "25×50"]
data = np.array([
    [42, 46, 32],
    [27, 20, 22],
    [0, 0, 0],
    [0, 0, 0],
    [7, 7, 4],
    [72, 65, 63],
    [7, 7, 7],
    [0, 0, 0],
], dtype=float)
fig, ax = plt.subplots(figsize=(8.2, 4.4))
image = ax.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["10%", "20%", "30%"])
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows)
ax.set_xlabel("Szum")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{int(data[i, j])}%", ha="center", va="center", fontsize=9)
fig.colorbar(image, ax=ax, label="Odzysk [%]")
ax.set_title("Dokładny odzysk, reguła Hebba, wszystkie wzorce")
fig.tight_layout()
fig.savefig(out / "odzysk.png")
plt.close()

noise = [10, 20, 30]
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(noise, [72, 68, 55], "o-", color="#3d5a80", label="25×25, Hebb")
ax.plot(noise, [67, 72, 57], "o--", color="#98c1d9", label="25×25, Oja")
ax.plot(noise, [0, 0, 0], "s-", color="#c44536", label="25×50, Hebb i Oja")
ax.set_xticks(noise)
ax.set_xlabel("Szum [%]")
ax.set_ylabel("Dokładny odzysk [%]")
ax.set_ylim(-3, 100)
ax.legend()
ax.set_title("Te same rysunki, dwa rozmiary")
fig.tight_layout()
fig.savefig(out / "rozmiar_25.png")
plt.close()

fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(noise, [0, 0, 0], "o-", color="#c44536", label="podobne, 69% zgodności")
ax.plot(noise, [100, 100, 100], "o-", color="#3d5a80", label="zróżnicowane, 51% zgodności")
ax.set_xticks(noise)
ax.set_xlabel("Szum [%]")
ax.set_ylabel("Dokładny odzysk [%]")
ax.set_ylim(-3, 105)
ax.legend()
ax.set_title("Sześć wzorców, N = 625, reguła Hebba")
fig.tight_layout()
fig.savefig(out / "podobne_zroznicowane.png")
plt.close()

labels = ["25×25", "zwierzęta", "alfabet"]
pattern = np.array([1, 1, 0])
negation = np.array([1, 2, 0])
spurious = np.array([18, 17, 20])
fig, ax = plt.subplots(figsize=(6.4, 4))
xx = np.arange(3)
ax.bar(xx, pattern, label="wzorzec", color="#3d5a80")
ax.bar(xx, negation, bottom=pattern, label="negacja", color="#ee6c4d")
ax.bar(xx, spurious, bottom=pattern + negation, label="inny stan", color="#98c1d9")
ax.set_xticks(xx)
ax.set_xticklabels(labels)
ax.set_ylabel("Liczba prób (z 20)")
ax.legend()
ax.set_title("Losowe wejście, reguła Hebba")
fig.tight_layout()
fig.savefig(out / "losowe_wejscie.png")
plt.close()


def grid(text):
    rows = text.strip().splitlines()
    return np.array([[1 if char == "#" else -1 for char in row] for row in rows])


panels = [
    ("Wejście", "#.#.\n.#.#\n#.#.\n.#.#"),
    ("Krok synchroniczny", ".#.#\n#.#.\n.#.#\n#.#."),
    ("Koniec asynchroniczny", "....\n....\n....\n...."),
]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
for axis, (title, text) in zip(axes, panels):
    axis.imshow(grid(text), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
fig.tight_layout()
fig.savefig(out / "oscylacja.png")
plt.close()

patterns = np.loadtxt(out.parent / "experiments" / "results" / "stable-5x5.csv", delimiter=",")
fig, axes = plt.subplots(3, 4, figsize=(8, 6.2))
for index, axis in enumerate(axes.ravel()):
    axis.imshow(patterns[index].reshape(5, 5), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    axis.set_title(str(index + 1), fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
fig.suptitle("Dwanaście stabilnych wzorców 5×5")
fig.tight_layout()
fig.savefig(out / "zbior_5x5.png")
plt.close()
print("zapisano")
