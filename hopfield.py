
import numpy as np


class HopfieldNetwork:
    def __init__(self):
        self.weights = None
        self.patterns = None
        self.n_neurons = None

    def _as_patterns(self, patterns):
        """Checks that patterns are a non-empty 2D array of values ±1."""
        patterns = np.asarray(patterns, dtype=int)

        if patterns.ndim != 2:
            raise ValueError("Wzorce muszą być tablicą 2D.")

        if patterns.shape[0] == 0 or patterns.shape[1] == 0:
            raise ValueError("Zbiór wzorców nie może być pusty.")

        if not np.all(np.isin(patterns, [-1, 1])):
            raise ValueError("Wzorce muszą zawierać wyłącznie -1 i 1.")

        return patterns

    def train_hebb(self, patterns):
        """Trains the network with Hebb's rule."""
        patterns = self._as_patterns(patterns)

        self.patterns = patterns
        self.n_neurons = patterns.shape[1]

        # Classical Hebb rule: W = (1/N) * X^T X
        self.weights = (patterns.T @ patterns) / self.n_neurons

        # A neuron is not connected to itself.
        np.fill_diagonal(self.weights, 0)

        return self.weights

    def train_oja(self, patterns, learning_rate=0.01, epochs=200):
        """
        Trains the network with Erkki Oja's rule.

        Patterns are presented one at a time. For autoassociation the neuron
        output is clamped to the pattern bit (y_i = x_i). With values ±1 the
        update

            Δw_ij = η * y_i * (x_j - y_i * w_ij)

        simplifies to

            Δw_ij = η * (x_i * x_j - w_ij),  i ≠ j.

        learning_rate is η in (0, 1], epochs is the number of passes over the
        set. Weights start at zero.
        """
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate musi być w przedziale (0, 1].")

        if not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError("epochs musi być dodatnią liczbą całkowitą.")

        patterns = self._as_patterns(patterns)

        self.patterns = patterns
        self.n_neurons = patterns.shape[1]
        self.weights = np.zeros((self.n_neurons, self.n_neurons), dtype=float)

        for _ in range(epochs):
            for pattern in patterns:
                outer = np.outer(pattern, pattern)
                self.weights += learning_rate * (outer - self.weights)
                np.fill_diagonal(self.weights, 0)

        return self.weights

    def _check_trained(self):
        if self.weights is None or self.n_neurons is None:
            raise RuntimeError("Sieć nie została jeszcze nauczona.")

    def evolve(self, state, mode="synchronous", max_steps=20, seed=None):
        """
        Runs the network from the given state and records each step.

        mode="synchronous": one step updates every neuron from the same state.
        mode="asynchronous": one step is a full pass over the neurons in random
        order. Later neurons see updates made earlier in the same pass.

        status:
        - "stable" — the next step changes nothing,
        - "cycle" — a previous state repeated,
        - "max_steps" — the step limit was reached.

        Returns (states, energies, status). The first state is the input.
        """
        self._check_trained()

        if mode not in ("synchronous", "asynchronous"):
            raise ValueError("mode musi być 'synchronous' albo 'asynchronous'.")

        if not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
            raise ValueError("max_steps musi być dodatnią liczbą całkowitą.")

        state = np.asarray(state, dtype=int).copy()

        if state.shape != (self.n_neurons,):
            raise ValueError(
                f"Stan ma długość {state.size}, a sieć ma {self.n_neurons} neuronów."
            )

        if not np.all(np.isin(state, [-1, 1])):
            raise ValueError("Stan musi zawierać wyłącznie -1 i 1.")

        rng = np.random.default_rng(seed)
        states = [state.copy()]
        energies = [float(self.energy(state))]
        seen = {state.tobytes(): 0}

        for _ in range(max_steps):
            if mode == "synchronous":
                new_state = self.update_synchronous(state)
            else:
                new_state = self._asynchronous_sweep(state, rng)

            if np.array_equal(new_state, state):
                return np.vstack(states), np.asarray(energies), "stable"

            states.append(new_state.copy())
            energies.append(float(self.energy(new_state)))

            if new_state.tobytes() in seen:
                return np.vstack(states), np.asarray(energies), "cycle"

            seen[new_state.tobytes()] = len(states) - 1
            state = new_state

        return np.vstack(states), np.asarray(energies), "max_steps"

    def _asynchronous_sweep(self, state, rng):
        """One asynchronous pass: each neuron is visited at most once."""
        state = np.array(state, copy=True)
        order = rng.permutation(self.n_neurons)

        for index in order:
            activation = self.weights[index] @ state

            if activation > 0:
                state[index] = 1
            elif activation < 0:
                state[index] = -1

        return state

    def update_synchronous(self, state):
        """Updates every neuron at the same time."""
        self._check_trained()
        activations = self.weights @ state

        new_state = np.where(
            activations > 0, 1,
            np.where(activations < 0, -1, state)
        )

        return new_state

    def update_asynchronous(self, state, neuron_index=None):
        """Updates a single neuron."""
        self._check_trained()
        state = state.copy()

        if neuron_index is None:
            neuron_index = np.random.randint(len(state))

        activation = self.weights[neuron_index] @ state

        if activation > 0:
            state[neuron_index] = 1
        elif activation < 0:
            state[neuron_index] = -1

        return state

    def is_stable(self, pattern):
        """Returns whether one synchronous step leaves the pattern unchanged."""
        result = self.update_synchronous(pattern)
        return np.array_equal(result, pattern)

    def energy(self, state):
        """Energy of a network state."""
        self._check_trained()
        return -0.5 * state @ self.weights @ state
