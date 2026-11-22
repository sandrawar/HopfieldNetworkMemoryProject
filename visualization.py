
import numpy as np
import matplotlib.pyplot as plt


def finish_figures():
    """Opens every prepared figure and waits until the user closes them."""
    print("Rysunki gotowe. Zamknij okna, żeby zakończyć program.")
    plt.show()


def _show_image(axis, pattern, height, width):
    axis.imshow(
        np.asarray(pattern).reshape(height, width),
        cmap="gray",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axis.set_aspect("equal")
    axis.axis("off")


def show_bitmap(pattern, height, width, title="Bitmapa"):
    """Shows one pattern as a black-and-white bitmap."""
    fig, axis = plt.subplots(figsize=(4, 4))
    _show_image(axis, pattern, height, width)
    fig.suptitle(title)
    finish_figures()


def show_patterns(patterns, height, width, title="Wzorce uczące"):
    """
    Shows every pattern in the set.
    Wide images wrap onto extra rows so the whole set fits in the window.
    """
    number_of_patterns = len(patterns)
    if height > width:
        columns = min(number_of_patterns, 6)
    else:
        columns = min(number_of_patterns, 3)
    rows = int(np.ceil(number_of_patterns / columns))

    figure_width = 3.1 * columns
    figure_height = 3.1 * rows
    scale = min(1.0, 12 / figure_width, 8 / figure_height)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(figure_width * scale, figure_height * scale),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, pattern in enumerate(patterns):
        _show_image(axes[i], pattern, height, width)
        axes[i].set_title(f"Wzorzec {i + 1}")

    for axis in axes[number_of_patterns:]:
        axis.axis("off")

    fig.suptitle(title)


def show_convergence(states, height, width, title="Zbieganie sieci"):
    """
    Shows successive network states as rectangular bitmaps.
    states[0] is the initial state; each later frame is one simulation step.
    """
    states = np.asarray(states)

    if states.ndim == 1:
        states = states.reshape(1, -1)

    if states.shape[1] != height * width:
        raise ValueError(
            f"Stan ma {states.shape[1]} neuronów, a bitmapa ma {height * width}."
        )

    number_of_states = len(states)
    columns = min(number_of_states, 8)
    rows = int(np.ceil(number_of_states / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.3 * columns, 2.5 * rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, state in enumerate(states):
        _show_image(axes[i], state, height, width)
        axes[i].set_title("start" if i == 0 else f"krok {i}")

    for axis in axes[number_of_states:]:
        axis.axis("off")

    fig.suptitle(title)
