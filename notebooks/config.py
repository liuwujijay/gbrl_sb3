"""
config.py
─────────
Typed configuration dataclasses for the AWR-GBRL offline classifier.

All hyper-parameters live here so the training script stays free of
magic numbers.  Override individual fields via ``dataclasses.replace``
or load from a dict with ``TrainConfig.from_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional


@dataclass(frozen=True)
class PERConfig:
    """Prioritized Experience Replay hyper-parameters."""

    alpha: float = 0.6
    """Priority exponent.  0 = uniform, 1 = full prioritisation."""

    beta_start: float = 0.4
    """IS-weight correction start value (annealed → 1.0)."""

    eps: float = 1e-6
    """Floor added to every priority to prevent zero-probability transitions."""


@dataclass(frozen=True)
class TreeStructConfig:
    """GBRL tree structure hyper-parameters."""

    max_depth: int = 4
    n_bins: int = 64
    min_data_in_leaf: int = 0
    par_th: int = 2
    grow_policy: str = "oblivious"


@dataclass(frozen=True)
class OptimizerConfig:
    """Per-head optimizer settings."""

    algo: str
    lr: float
    shrinkage: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """Top-level training configuration."""

    # ── data / experiment ─────────────────────────────────────────────────
    seed: int = 42
    test_size: float = 0.2

    # ── replay buffer ─────────────────────────────────────────────────────
    buffer_size: int = 5_000
    batch_size: int = 64

    # ── training loop ─────────────────────────────────────────────────────
    num_epochs: int = 50
    gradient_steps: int = 10
    eval_every: int = 5

    # ── AWR hyper-parameters ──────────────────────────────────────────────
    beta: float = 0.1
    """AWR temperature.  Lower = greedier imitation of high-advantage actions."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    weights_max: float = 20.0
    normalize_advantage: bool = True
    reward_mode: str = "gae"

    # ── PER ───────────────────────────────────────────────────────────────
    per: PERConfig = field(default_factory=PERConfig)

    # ── tree structure ────────────────────────────────────────────────────
    tree_struct: TreeStructConfig = field(default_factory=TreeStructConfig)

    # ── optimizers ────────────────────────────────────────────────────────
    policy_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(algo="SGD", lr=0.05)
    )
    value_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(algo="SGD", lr=0.005)
    )

    # ── runtime ───────────────────────────────────────────────────────────
    device: str = "cpu"

    # ── LightGBM baseline ─────────────────────────────────────────────────
    lgbm_n_estimators: int = 100
    lgbm_max_depth: int = 4
    lgbm_learning_rate: float = 0.1
    lgbm_num_leaves: int = 15

    # ── EMA logging ───────────────────────────────────────────────────────────
    ema_alpha: float = 0.1

    # ── data source ───────────────────────────────────────────────────────────
    feature_cols: List[str] = field(default_factory=list)
    """Column names to use as features.  Empty = auto-detect (all but label)."""

    label_col: str = "label"
    """Column name containing the integer class label."""

    chunk_size: int = 4_096
    """Rows per streaming chunk.  Keep ≤ buffer_size to avoid overflow."""

    scaler_fit_rows: int = 50_000
    """Max rows used to fit the StandardScaler when streaming large sources."""

    # S3 daily partitions  (sorted lexicographically → oldest first)
    s3_uris: List[str] = field(default_factory=list)
    """List of S3 URIs, one per day/partition.
    Accepts both ``s3://bucket/prefix`` and ``s3:bucket/prefix`` forms."""

    s3_anon: bool = False
    """Use anonymous (unsigned) S3 access for public buckets."""

    s3_profile: Optional[str] = None
    """AWS credentials profile (from ~/.aws/credentials)."""

    # local file paths (glob patterns accepted)
    local_paths: List[str] = field(default_factory=list)
    """Local .parquet or .csv file paths (glob patterns accepted)."""

    # ─────────────────────────────────────────────────────────────────────

    @property
    def per_beta_steps(self) -> int:
        """Total sample() calls over which PER β anneals to 1.0."""
        return self.num_epochs * self.gradient_steps

    def as_policy_kwargs(self) -> dict[str, Any]:
        """Build the policy_kwargs dict expected by AWR_GBRL."""
        ts = self.tree_struct
        po = self.policy_optimizer
        vo = self.value_optimizer
        return {
            "tree_struct": {
                "max_depth": ts.max_depth,
                "n_bins": ts.n_bins,
                "min_data_in_leaf": ts.min_data_in_leaf,
                "par_th": ts.par_th,
                "grow_policy": ts.grow_policy,
            },
            "tree_optimizer": {
                "params": {
                    "split_score_func": "cosine",
                    "control_variates": False,
                    "generator_type": "Quantile",
                    "feature_weights": None,
                },
                "policy_optimizer": {
                    "policy_algo": po.algo,
                    "policy_lr": po.lr,
                    "policy_shrinkage": po.shrinkage,
                },
                "value_optimizer": {
                    "value_algo": vo.algo,
                    "value_lr": vo.lr,
                    "value_shrinkage": vo.shrinkage,
                },
            },
            "shared_tree_struct": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        """Reconstruct from a plain dict (e.g. loaded from JSON/YAML)."""
        per = PERConfig(**d.pop("per", {}))
        ts  = TreeStructConfig(**d.pop("tree_struct", {}))
        po  = OptimizerConfig(**d.pop("policy_optimizer", {"algo": "SGD", "lr": 0.05}))
        vo  = OptimizerConfig(**d.pop("value_optimizer", {"algo": "SGD", "lr": 0.005}))
        return cls(per=per, tree_struct=ts, policy_optimizer=po, value_optimizer=vo, **d)
