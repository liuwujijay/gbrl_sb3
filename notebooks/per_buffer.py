"""
per_buffer.py
─────────────
Prioritized Experience Replay (PER) buffer built on top of AWRReplayBuffer.

Provides:
  - SumTree         : O(log n) priority insert + stratified sampling
  - PrioritizedAWRReplayBuffer : drop-in replacement for AWRReplayBuffer
"""

from __future__ import annotations

import numpy as np

from buffers.replay_buffer import AWRReplayBuffer, AWRReplayBufferSamples


# ──────────────────────────────────────────────────────────────────────────────
# SumTree
# ──────────────────────────────────────────────────────────────────────────────

class SumTree:
    """
    Binary sum-tree for O(log n) priority-weighted sampling.

    Leaves store individual transition priorities; internal nodes store
    subtree sums.  The root (index 0) holds the total priority mass.

    Capacity must be a positive integer.  The tree is circular: once full,
    the oldest leaf is overwritten (matching the replay-buffer's FIFO policy).
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"SumTree capacity must be > 0, got {capacity}")
        self.capacity: int = capacity
        # size = 2*capacity: indices [0, capacity-1) are internal nodes,
        # [capacity-1, 2*capacity-1) are leaves
        self.tree: np.ndarray = np.zeros(2 * capacity, dtype=np.float64)
        self._write: int = 0  # next circular write position (data index)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _propagate(self, leaf_idx: int, delta: float) -> None:
        """Walk up the tree from leaf_idx, adding delta to every ancestor."""
        idx = leaf_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def _retrieve(self, idx: int, s: float) -> int:
        """Return the leaf index whose cumulative range contains s."""
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx          # reached a leaf
            right = left + 1
            if right >= len(self.tree) or s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    # ── public API ───────────────────────────────────────────────────────────

    def update(self, data_idx: int, priority: float) -> None:
        """Set the priority of transition at *data_idx* (0-based data index)."""
        leaf_idx = data_idx + self.capacity - 1
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def add(self, priority: float) -> int:
        """
        Write *priority* to the next circular slot.

        Returns the data index that was written (matches replay buffer ``pos``
        at the time of the corresponding ``add()`` call).
        """
        data_idx = self._write
        self.update(data_idx, priority)
        self._write = (self._write + 1) % self.capacity
        return data_idx

    def get(self, s: float) -> tuple[int, float, int]:
        """
        Stratified lookup for cumulative value *s*.

        Returns
        -------
        leaf_idx  : absolute tree index of the chosen leaf
        priority  : stored priority at that leaf
        data_idx  : 0-based replay-buffer index
        """
        s = float(np.clip(s, 0.0, self.total - 1e-8))
        leaf_idx = self._retrieve(0, s)
        # guard: keep inside the valid leaf range
        leaf_idx = int(np.clip(leaf_idx, self.capacity - 1, 2 * self.capacity - 2))
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), data_idx

    @property
    def total(self) -> float:
        """Total priority mass (root of the tree)."""
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Largest priority among all leaves, defaulting to 1.0 if empty."""
        leaves = self.tree[self.capacity - 1:]
        peak = float(leaves.max())
        return peak if peak > 0.0 else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# PrioritizedAWRReplayBuffer
# ──────────────────────────────────────────────────────────────────────────────

class PrioritizedAWRReplayBuffer(AWRReplayBuffer):
    """
    Drop-in replacement for ``AWRReplayBuffer`` with SumTree-based PER.

    New transitions are inserted with ``max_priority`` so they are sampled
    at least once.  After each gradient step, call ``update_priorities()``
    with the absolute TD errors to adjust per-transition weights.

    Importance-sampling weights (stored in ``self.is_weights``) correct for
    the sampling bias introduced by prioritisation.  ``beta`` is annealed
    linearly from ``beta_start`` to 1.0 over ``beta_steps`` sample calls.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to ``AWRReplayBuffer.__init__``.
    alpha : float
        Priority exponent.  0 = uniform sampling, 1 = full prioritisation.
    beta : float
        IS-weight exponent start value (annealed → 1.0).
    beta_steps : int
        Number of ``sample()`` calls over which beta reaches 1.0.
    eps : float
        Small constant added to every priority to prevent zero-probability
        transitions.
    """

    def __init__(
        self,
        *args,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_steps: int = 1_000,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if beta_steps <= 0:
            raise ValueError(f"beta_steps must be > 0, got {beta_steps}")

        self.alpha: float = alpha
        self.beta: float = beta
        self._beta_start: float = beta
        self._beta_steps: int = beta_steps
        self.eps: float = eps
        self._sample_calls: int = 0

        self.sum_tree: SumTree = SumTree(self.buffer_size)
        self.is_weights: np.ndarray = np.ones(1, dtype=np.float32)
        self._last_indices: np.ndarray = np.zeros(1, dtype=np.int64)

    # ── overrides ────────────────────────────────────────────────────────────

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        """Insert transition with max-current-priority into the SumTree."""
        priority = self.sum_tree.max_priority ** self.alpha
        self.sum_tree.add(priority)   # circular write index matches self.pos
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(
        self,
        batch_size: int,
        env=None,
    ) -> AWRReplayBufferSamples:
        """
        Stratified priority-weighted sampling with IS correction.

        Divides the total priority into ``batch_size`` equal segments and
        draws one sample uniformly from each segment — this reduces variance
        compared to purely random priority-weighted sampling.
        """
        self._sample_calls += 1
        self.beta = min(
            1.0,
            self._beta_start
            + (1.0 - self._beta_start) * self._sample_calls / self._beta_steps,
        )

        valid = self.buffer_size if self.full else self.pos
        if valid == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        segment = self.sum_tree.total / batch_size
        indices: list[int] = []
        raw_priorities: list[float] = []

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            _, p, data_idx = self.sum_tree.get(s)
            data_idx = int(np.clip(data_idx, 0, valid - 1))
            indices.append(data_idx)
            raw_priorities.append(p + self.eps)

        self._last_indices = np.array(indices, dtype=np.int64)
        priorities = np.array(raw_priorities, dtype=np.float64)

        # IS weights:  w_i = (N · P(i))^{-β}  normalised by max(w)
        probs = priorities / (self.sum_tree.total + self.eps)
        raw_w = (valid * probs) ** (-self.beta)
        self.is_weights = (raw_w / raw_w.max()).astype(np.float32)

        return self._get_samples(self._last_indices, env=env)

    def update_priorities(self, td_errors: np.ndarray) -> None:
        """
        Recompute per-transition priorities from absolute TD errors.

        Should be called once after each gradient step with the TD errors
        of the last sampled batch.

        Parameters
        ----------
        td_errors : np.ndarray, shape (batch_size,)
            Absolute TD errors |r + γV(s') - V(s)| for the last batch.
        """
        if len(td_errors) != len(self._last_indices):
            raise ValueError(
                f"td_errors length {len(td_errors)} does not match "
                f"last batch size {len(self._last_indices)}"
            )
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, p in zip(self._last_indices, priorities):
            self.sum_tree.update(int(idx), float(p))

    # ── diagnostics ──────────────────────────────────────────────────────────

    def per_stats(self) -> dict:
        """Return a dict of PER health metrics for logging."""
        return {
            "per/beta":           round(self.beta, 4),
            "per/tree_total":     round(self.sum_tree.total, 4),
            "per/max_priority":   round(self.sum_tree.max_priority, 4),
            "per/mean_is_weight": round(float(self.is_weights.mean()), 4),
            "per/min_is_weight":  round(float(self.is_weights.min()), 4),
        }
