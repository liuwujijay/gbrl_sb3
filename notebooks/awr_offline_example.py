"""
awr_offline_example.py
──────────────────────
AWR-GBRL offline classification.  Accepts three data source types:

  1. Built-in Iris dataset  (default, for demo / smoke-test)
  2. pandas DataFrame       (in-memory, small data)
  3. Local files            (CSV or Parquet, glob patterns OK)
  4. S3 daily partitions    (s3://bucket/prefix/YYYY-MM-DD  or
                             s3:bucket/prefix/folder-N)
     Files are listed and streamed folder-by-folder, chunk-by-chunk —
     the full dataset is never loaded into memory at once.

Each sample is treated as a length-1 RL episode:
  state   = feature vector
  action  = expert class label
  reward  = 1.0  (expert is always correct)
  done    = True  (no temporal credit — pure bandit)

Usage
-----
    # built-in Iris demo
    python -m notebooks.awr_offline_example

    # local parquet files
    python -m notebooks.awr_offline_example \\
        --local-paths "data/train_*.parquet" \\
        --feature-cols sepal_length sepal_width petal_length petal_width \\
        --label-col species

    # S3 daily partitions
    python -m notebooks.awr_offline_example \\
        --s3-uris s3://mybucket/events/2026-03-06 \\
                  s3://mybucket/events/2026-03-07 \\
                  s3://mybucket/events/2026-03-08 \\
        --feature-cols f1 f2 f3 f4 --label-col label \\
        --chunk-size 8192

    # quiet (suppress per-step ticker)
    python -m notebooks.awr_offline_example --quiet
"""

from __future__ import annotations

import os

# ── Thread-count caps: must be set before any C-extension is imported ─────────
# Prevents an OpenMP fork-lock deadlock on macOS when LightGBM and GBRL
# both try to initialise their own thread pools in the same process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import logging
import sys
import time
from collections import deque
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd
import torch as th
import gymnasium as gym
import lightgbm as lgb
from gymnasium import spaces
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common import utils as sb3_utils
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algos.awr import AWR_GBRL
from notebooks.config import TrainConfig
from notebooks.data_loader import DataSource
from notebooks.per_buffer import PrioritizedAWRReplayBuffer


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("stable_baselines3").setLevel(logging.WARNING)
    logging.getLogger("gymnasium").setLevel(logging.WARNING)
    return logging.getLogger("awr_iris")


log = _setup_logging()


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def _iris_as_dataframe() -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, list[str]]:
    """
    Load the built-in Iris dataset and return train/test DataFrames plus
    column metadata.  Used when no external data source is specified.
    """
    from sklearn.datasets import load_iris  # keep import local — not needed for S3 path
    iris = load_iris()
    feature_cols = [n.replace(" (cm)", "").replace(" ", "_")
                    for n in iris.feature_names]
    label_col    = "species"
    target_names = iris.target_names.tolist()

    df = pd.DataFrame(iris.data, columns=feature_cols)
    df[label_col] = iris.target.astype(np.int64)

    train_df = df.sample(frac=0.8, random_state=42)
    test_df  = df.drop(train_df.index)
    return train_df, test_df, feature_cols, label_col, target_names


def load_dataset(
    cfg: TrainConfig,
    df: Optional[pd.DataFrame] = None,
) -> tuple[DataSource, np.ndarray, np.ndarray, StandardScaler, int, int, list[str]]:
    """
    Build a DataSource from whichever input is configured, fit a scaler,
    and return a held-out test set as numpy arrays.

    Priority (first match wins):
      1. ``df``          — caller-supplied pandas DataFrame
      2. ``cfg.s3_uris`` — S3 daily folders
      3. ``cfg.local_paths`` — local files
      4. fallback        — built-in Iris dataset

    Returns
    -------
    source       : DataSource for streaming training data
    X_test       : np.ndarray  (test features, already scaled)
    y_test       : np.ndarray  (test labels)
    scaler       : fitted StandardScaler
    obs_dim      : int
    n_classes    : int
    target_names : list[str]
    """
    log.info("─" * 60)

    if df is not None:
        # ── pandas DataFrame (small data) ────────────────────────────────────
        log.info("Data source: pandas DataFrame")
        feature_cols = cfg.feature_cols or [c for c in df.columns if c != cfg.label_col]
        label_col    = cfg.label_col

        train_df = df.sample(frac=0.8, random_state=cfg.seed)
        test_df  = df.drop(train_df.index)
        target_names = [str(c) for c in sorted(df[label_col].unique())]
        source = DataSource.from_dataframe(train_df, feature_cols, label_col)

    elif cfg.s3_uris:
        # ── S3 daily partitions ───────────────────────────────────────────────
        log.info(f"Data source: S3  ({len(cfg.s3_uris)} daily folder(s))")
        for u in cfg.s3_uris:
            log.info(f"  {u}")
        feature_cols = cfg.feature_cols
        label_col    = cfg.label_col
        if not feature_cols:
            raise ValueError("--feature-cols must be specified for S3 sources.")
        source = DataSource.from_s3(
            cfg.s3_uris, feature_cols, label_col,
            anon=cfg.s3_anon, profile=cfg.s3_profile,
        )
        # For S3 we use the last chunk as a proxy test set (no separate split)
        # Users should supply explicit test data in production.
        test_df = None
        target_names = [str(i) for i in range(100)]  # unknown a-priori

    elif cfg.local_paths:
        # ── local files ───────────────────────────────────────────────────────
        log.info(f"Data source: local files  ({len(cfg.local_paths)} path(s))")
        feature_cols = cfg.feature_cols
        label_col    = cfg.label_col
        if not feature_cols:
            raise ValueError("--feature-cols must be specified for local file sources.")
        source = DataSource.from_local(cfg.local_paths, feature_cols, label_col)
        test_df = None
        target_names = [str(i) for i in range(100)]

    else:
        # ── built-in Iris fallback ────────────────────────────────────────────
        log.info("Data source: built-in Iris dataset (demo mode)")
        train_df, test_df, feature_cols, label_col, target_names = \
            _iris_as_dataframe()
        source = DataSource.from_dataframe(train_df, feature_cols, label_col)

    # ── fit scaler ────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    source.fit_scaler(scaler, max_rows=cfg.scaler_fit_rows,
                      chunk_size=cfg.chunk_size)

    # ── build test arrays ─────────────────────────────────────────────────────
    if test_df is not None:
        assert isinstance(test_df, pd.DataFrame)
        X_test = scaler.transform(
            test_df[feature_cols].to_numpy(dtype=np.float32)
        ).astype(np.float32)
        y_test = test_df[label_col].to_numpy(dtype=np.int64)
    else:
        # S3 / large-local: stream one chunk as a test proxy
        log.warning("No explicit test split — using first chunk as test set.")
        X_test_raw, y_test = next(source.stream(chunk_size=512, scaler=None))
        X_test = scaler.transform(X_test_raw).astype(np.float32)

    # Infer obs_dim and n_classes
    obs_dim   = len(feature_cols)
    n_classes = int(len(np.unique(y_test)))

    # Re-map target_names to actual observed classes
    if len(target_names) > n_classes:
        target_names = target_names[:n_classes]

    log.info(f"  features   : {obs_dim}  {feature_cols}")
    log.info(f"  classes    : {n_classes}  {target_names}")
    log.info(f"  test rows  : {len(y_test)}")
    log.info(f"  chunk size : {cfg.chunk_size}")
    log.info("─" * 60)

    return source, X_test, y_test, scaler, obs_dim, n_classes, target_names


# ──────────────────────────────────────────────────────────────────────────────
# Buffer population
# ──────────────────────────────────────────────────────────────────────────────

def fill_buffer_streaming(
    model: AWR_GBRL,
    source: DataSource,
    scaler: StandardScaler,
    cfg: TrainConfig,
) -> int:
    """
    Stream all training data from *source* into the replay buffer in chunks.

    Each chunk is at most ``cfg.chunk_size`` rows.  When the buffer is full
    it wraps circularly (oldest transitions are evicted — FIFO).

    Episode definition per row
    --------------------------
    obs      = feature vector   (1 × obs_dim, scaled)
    action   = class label      (expert action)
    reward   = 1.0              (expert always correct)
    next_obs = obs              (no next-state dynamics)
    done     = True             (single-step terminal episode)

    Returns total number of transitions added.
    """
    total = 0
    for X_chunk, y_chunk in source.stream(chunk_size=cfg.chunk_size,
                                           scaler=scaler):
        for obs, label in zip(X_chunk, y_chunk):
            model.replay_buffer.add(
                obs      = obs[np.newaxis, :],
                next_obs = obs[np.newaxis, :],
                action   = np.array([[int(label)]], dtype=np.int64),
                reward   = np.array([1.0],          dtype=np.float32),
                done     = np.array([1.0],          dtype=np.float32),
                infos    = [{"TimeLimit.truncated": False}],
            )
            total += 1
        buf = model.replay_buffer
        log.debug(f"  chunk loaded: +{len(X_chunk)}  "
                  f"buffer pos={buf.pos}  full={buf.full}")

    buf = model.replay_buffer
    log.info(f"Buffer filled: {total:,} transitions  "
             f"(pos={buf.pos}, full={buf.full})")
    return total

def build_model(cfg: TrainConfig, obs_dim: int, n_classes: int) -> AWR_GBRL:
    """
    Construct a DummyVecEnv shim (never stepped) and return a fully
    initialised AWR_GBRL model whose spaces match the dataset.
    """
    obs_space    = spaces.Box(low=-np.inf, high=np.inf,
                              shape=(obs_dim,), dtype=np.float32)
    action_space = spaces.Discrete(n_classes)

    # CartPole-v1 is only used so DummyVecEnv has a valid env to construct;
    # its spaces are immediately overridden below — the env is never stepped.
    vec_env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    vec_env.observation_space         = obs_space
    vec_env.action_space              = action_space
    vec_env.envs[0].observation_space = obs_space
    vec_env.envs[0].action_space      = action_space

    model = AWR_GBRL(
        env               = vec_env,
        beta              = cfg.beta,
        gamma             = cfg.gamma,
        gae_lambda        = cfg.gae_lambda,
        ent_coef          = cfg.ent_coef,
        vf_coef           = cfg.vf_coef,
        weights_max       = cfg.weights_max,
        normalize_advantage = cfg.normalize_advantage,
        reward_mode       = cfg.reward_mode,
        learning_starts   = 0,
        batch_size        = cfg.batch_size,
        buffer_size       = cfg.buffer_size,
        gradient_steps    = cfg.gradient_steps,
        policy_kwargs     = cfg.as_policy_kwargs(),
        device            = cfg.device,
        verbose           = 0,
        seed              = cfg.seed,
    )

    # Replace the default buffer with the PER variant
    model.replay_buffer = PrioritizedAWRReplayBuffer(
        buffer_size       = cfg.buffer_size,
        observation_space = obs_space,
        action_space      = action_space,
        gamma             = model.gamma,
        gae_lambda        = model.gae_lambda,
        device            = cfg.device,
        n_envs            = 1,
        return_type       = cfg.reward_mode,
        alpha             = cfg.per.alpha,
        beta              = cfg.per.beta_start,
        beta_steps        = cfg.per_beta_steps,
        eps               = cfg.per.eps,
        env               = None,
    )

    # Initialise SB3 internal bookkeeping (normally done inside learn())
    model._logger           = sb3_utils.configure_logger(0, None, "awr_iris", True)
    model.ep_info_buffer    = deque(maxlen=100)
    model.ep_success_buffer = deque(maxlen=100)

    log.info("AWR-GBRL model built")
    log.info(f"  PER α={cfg.per.alpha}  β₀={cfg.per.beta_start}"
             f"  β_steps={cfg.per_beta_steps}")
    log.info(f"  obs_dim={obs_dim}  n_classes={n_classes}"
             f"  buffer={cfg.buffer_size}  batch={cfg.batch_size}")
    return model



# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: AWR_GBRL,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> tuple[float, float, float, np.ndarray]:
    """
    Run deterministic inference on the held-out test set.

    ``model.predict(obs, deterministic=True)`` contract:
    - type    : np.ndarray  (SB3 always wraps in numpy)
    - dtype   : int64       (Discrete space → integer class index, NOT a prob)
    - size    : 1           (single obs → single action)
    - state   : None        (non-recurrent policy)

    Returns
    -------
    accuracy, precision (macro), recall (macro), y_pred
    """
    y_pred: list[int] = []

    for obs in X_test:
        action, state = model.predict(obs[np.newaxis, :], deterministic=True)

        # contract assertions — kept in prod to catch silent regressions
        assert isinstance(action, np.ndarray), \
            f"predict() must return np.ndarray, got {type(action)}"
        assert action.size == 1, \
            f"Expected scalar action, got size {action.size}"
        assert np.issubdtype(action.dtype, np.integer), \
            f"Discrete action must be integer dtype, got {action.dtype}"
        assert state is None, \
            f"Non-recurrent policy must return state=None, got {state}"

        action_val = int(action.flat[0])
        assert 0 <= action_val < n_classes, \
            f"Action {action_val} outside valid range [0, {n_classes - 1}]"

        y_pred.append(action_val)

    y_arr     = np.array(y_pred, dtype=np.int64)
    accuracy  = float(np.mean(y_arr == y_test))
    precision = float(precision_score(y_test, y_arr,
                                      average="macro", zero_division=0))
    recall    = float(recall_score(y_test, y_arr,
                                   average="macro", zero_division=0))
    return accuracy, precision, recall, y_arr


# ──────────────────────────────────────────────────────────────────────────────
# LightGBM baseline
# ──────────────────────────────────────────────────────────────────────────────

def run_lgbm_baseline(
    cfg: TrainConfig,
    source: DataSource,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
) -> tuple[float, float, float]:
    """
    Fit a LightGBM multiclass classifier as a static reference baseline.

    For large datasets the training data is streamed chunk-by-chunk and
    accumulated up to ``cfg.scaler_fit_rows`` rows so the baseline fit
    stays memory-bounded.

    n_jobs=1 is mandatory — running LightGBM multithreaded before GBRL
    training causes an OpenMP fork-lock deadlock on macOS.
    ``del clf`` explicitly releases the LightGBM thread pool.
    """
    log.info("─" * 60)
    log.info("LightGBM baseline")
    t0 = time.perf_counter()

    # Accumulate up to scaler_fit_rows rows for baseline training
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    total = 0
    for X_chunk, y_chunk in source.stream(chunk_size=cfg.chunk_size,
                                           scaler=scaler):
        X_parts.append(X_chunk)
        y_parts.append(y_chunk)
        total += len(X_chunk)
        if total >= cfg.scaler_fit_rows:
            break

    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    log.info(f"  training on {len(X_train):,} rows (capped at {cfg.scaler_fit_rows:,})")

    clf = lgb.LGBMClassifier(
        n_estimators  = cfg.lgbm_n_estimators,
        max_depth     = cfg.lgbm_max_depth,
        learning_rate = cfg.lgbm_learning_rate,
        num_leaves    = cfg.lgbm_num_leaves,
        random_state  = cfg.seed,
        n_jobs        = 1,   # single-threaded — avoids OpenMP deadlock
        verbose       = -1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    del clf, X_train, y_train  # release memory + thread pool

    elapsed   = time.perf_counter() - t0
    accuracy  = float(np.mean(y_pred == y_test))
    precision = float(precision_score(y_test, y_pred,
                                      average="macro", zero_division=0))
    recall    = float(recall_score(y_test, y_pred,
                                   average="macro", zero_division=0))

    log.info(f"  fit time  : {elapsed*1000:.0f}ms")
    log.info(f"  accuracy  : {accuracy*100:.1f}%")
    log.info(f"  precision : {precision*100:.1f}%  (macro)")
    log.info(f"  recall    : {recall*100:.1f}%  (macro)")
    log.info("  per-class report:")
    for line in classification_report(
            y_test, y_pred, target_names=target_names).splitlines():
        log.info("    " + line)
    log.info("─" * 60)

    return accuracy, precision, recall


# ──────────────────────────────────────────────────────────────────────────────
# TD-error helper
# ──────────────────────────────────────────────────────────────────────────────

def _compute_and_update_td_errors(
    model: AWR_GBRL,
    per_buf: PrioritizedAWRReplayBuffer,
) -> np.ndarray:
    """
    Compute |return - V(s)| for the last sampled batch and update PER.

    Because every episode is length-1 with done=True:
        TD error = |reward − V(obs)| = |1.0 − V(obs)|

    Must be called once per gradient step, immediately after model.train().
    """
    with th.no_grad():
        last_idx = per_buf._last_indices
        env_idx  = np.zeros(len(last_idx), dtype=np.int64)

        obs_t = th.tensor(
            per_buf.observations[last_idx, env_idx],
            dtype=th.float32,
        )
        ret_t = th.tensor(
            per_buf.returns[last_idx, env_idx],
            dtype=th.float32,
        )
        val_t     = model.policy.critic(obs_t).squeeze(-1)
        td_errors = (ret_t - val_t).abs().cpu().numpy()

    per_buf.update_priorities(td_errors)
    return td_errors


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    cfg: TrainConfig,
    model: AWR_GBRL,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    target_names: list[str],
    lgbm_results: tuple[float, float, float],
    quiet: bool = False,
) -> None:
    """
    Main offline training loop.

    Per epoch:
      1. Run ``gradient_steps`` gradient updates one-at-a-time for
         fine-grained per-step PER priority refresh and loss logging.
      2. Every ``eval_every`` epochs evaluate on the held-out test set.
      3. Track EMA-smoothed losses and rolling avg_last3 accuracy.
    """
    log.info("─" * 60)
    log.info(f"Training  {cfg.num_epochs} epochs × {cfg.gradient_steps} grad steps")
    log.info(f"  buffer={cfg.buffer_size}  batch={cfg.batch_size}"
             f"  eval_every={cfg.eval_every}")
    log.info("─" * 60)

    per_buf: PrioritizedAWRReplayBuffer = model.replay_buffer  # type: ignore[assignment]

    best_acc  = 0.0
    best_pre  = 0.0
    best_rec  = 0.0
    best_pred = np.zeros(len(y_test), dtype=np.int64)
    recent_accs: list[float] = []

    ema_policy: Optional[float] = None
    ema_value:  Optional[float] = None

    HDR = (
        f"{'Epoch':>6}  {'ms':>5}  {'policy':>8}  {'value':>7}  "
        f"{'entropy':>8}  {'trees':>5}  {'iter':>5}  "
        f"{'β_PER':>6}  {'IS_w̄':>5}  {'max_pri':>8}  "
        f"{'td_err̄':>7}  {'EMA_pol':>8}  {'EMA_val':>7}"
    )
    SEP = "─" * len(HDR)
    log.info(HDR)
    log.info(SEP)

    t_total = time.perf_counter()

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.perf_counter()

        step_policy:  list[float] = []
        step_value:   list[float] = []
        step_entropy: list[float] = []
        step_td:      list[float] = []

        if not quiet:
            print(f"  epoch {epoch:3d}/{cfg.num_epochs}  steps: ",
                  end="", flush=True)

        for _ in range(cfg.gradient_steps):
            model.train(gradient_steps=1, batch_size=cfg.batch_size)

            if not quiet:
                print(".", end="", flush=True)

            td_errors = _compute_and_update_td_errors(model, per_buf)

            kv = model.logger.name_to_value
            step_policy.append( kv.get("train/policy_loss",  float("nan")))
            step_value.append(  kv.get("train/value_loss",   float("nan")))
            step_entropy.append(kv.get("train/entropy_loss", float("nan")))
            step_td.append(float(np.mean(td_errors)))

        epoch_ms = (time.perf_counter() - t0) * 1000
        if not quiet:
            print(f"  {epoch_ms:.0f}ms", flush=True)

        # ── epoch aggregates ─────────────────────────────────────────────────
        pol     = float(np.nanmean(step_policy))
        val     = float(np.nanmean(step_value))
        ent     = float(np.nanmean(step_entropy))
        mean_td = float(np.nanmean(step_td))

        a = cfg.ema_alpha
        if ema_policy is None:
            ema_policy, ema_value = pol, val
        else:
            ema_policy = a * pol + (1 - a) * ema_policy
            ema_value  = a * val + (1 - a) * ema_value  # type: ignore[operator]

        kv         = model.logger.name_to_value
        n_trees    = kv.get("train/policy_num_trees",           0)
        boost_iter = kv.get("train/policy_boosting_iterations", 0)
        ps         = per_buf.per_stats()

        log.info(
            f"{epoch:>6d}  {epoch_ms:>5.0f}  "
            f"{pol:>+8.4f}  {val:>7.4f}  {ent:>+8.4f}  "
            f"{n_trees:>5d}  {boost_iter:>5d}  "
            f"{ps['per/beta']:>6.3f}  {ps['per/mean_is_weight']:>5.3f}  "
            f"{ps['per/max_priority']:>8.4f}  {mean_td:>7.4f}  "
            f"{ema_policy:>+8.4f}  {ema_value:>7.4f}"
        )
        log.info(
            f"         spread → "
            f"pol[{min(step_policy):+.3f}…{max(step_policy):+.3f}]  "
            f"val[{min(step_value):.3f}…{max(step_value):.3f}]  "
            f"td[{min(step_td):.3f}…{max(step_td):.3f}]"
        )

        # ── evaluation ───────────────────────────────────────────────────────
        if epoch % cfg.eval_every == 0 or epoch == cfg.num_epochs:
            acc, pre, rec, y_pred = evaluate(model, X_test, y_test, n_classes)
            recent_accs.append(acc)
            avg_last3 = float(np.mean(recent_accs[-3:])) * 100

            flag = ""
            if acc > best_acc:
                best_acc  = acc
                best_pre  = pre
                best_rec  = rec
                best_pred = y_pred
                flag = "  ★ new best"

            log.info(
                f"  ↳ TEST  acc={acc*100:.1f}%  "
                f"prec={pre*100:.1f}%  rec={rec*100:.1f}%  "
                f"avg_last3={avg_last3:.1f}%{flag}"
            )
            log.info(SEP)

    # ── final summary ─────────────────────────────────────────────────────────
    total_s = time.perf_counter() - t_total
    lgbm_acc, lgbm_pre, lgbm_rec = lgbm_results

    log.info("─" * 60)
    log.info("Training complete")
    log.info(f"  wall time   : {total_s:.2f}s")
    log.info(f"  trees grown : {model.policy.model.get_num_trees()}")
    log.info(f"  boost iters : {model.policy.model.get_iteration()}")
    log.info("")
    log.info("  ┌─────────────────────┬──────────┬──────────┬──────────┐")
    log.info("  │ Model               │ Accuracy │  Prec    │  Recall  │")
    log.info("  ├─────────────────────┼──────────┼──────────┼──────────┤")
    log.info(f"  │ LightGBM (baseline) │  {lgbm_acc*100:5.1f}%  │  {lgbm_pre*100:5.1f}%  │  {lgbm_rec*100:5.1f}%  │")
    log.info(f"  │ AWR-GBRL  (best)    │  {best_acc*100:5.1f}%  │  {best_pre*100:5.1f}%  │  {best_rec*100:5.1f}%  │")
    log.info("  └─────────────────────┴──────────┴──────────┴──────────┘")
    log.info("")
    log.info("  AWR-GBRL per-class report (best epoch):")
    for line in classification_report(
            y_test, best_pred, target_names=target_names).splitlines():
        log.info("    " + line)
    log.info("─" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AWR-GBRL offline classifier (DataFrame / local / S3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",         type=int,  default=50,    help="Training epochs")
    p.add_argument("--gradient-steps", type=int,  default=10,    help="Grad steps per epoch")
    p.add_argument("--batch-size",     type=int,  default=64,    help="Mini-batch size")
    p.add_argument("--buffer-size",    type=int,  default=5_000, help="Replay buffer capacity")
    p.add_argument("--eval-every",     type=int,  default=5,     help="Evaluate every N epochs")
    p.add_argument("--seed",           type=int,  default=42,    help="Random seed")
    p.add_argument("--device",         type=str,  default="cpu", help="Torch device")
    p.add_argument("--quiet",          action="store_true",      help="Suppress step ticker")

    # ── data source ───────────────────────────────────────────────────────────
    grp = p.add_argument_group("data source (mutually exclusive; default = built-in Iris)")
    grp.add_argument(
        "--s3-uris", nargs="+", metavar="URI", default=[],
        help="S3 daily-partition URIs  e.g. s3://bucket/data/2026-03-06 …")
    grp.add_argument(
        "--local-paths", nargs="+", metavar="PATH", default=[],
        help="Local .parquet/.csv files or glob patterns")
    grp.add_argument(
        "--feature-cols", nargs="+", metavar="COL", default=[],
        help="Feature column names (required for S3 / local)")
    grp.add_argument(
        "--label-col", type=str, default="label",
        help="Label column name")
    grp.add_argument(
        "--chunk-size", type=int, default=4_096,
        help="Rows per streaming chunk (keep ≤ --buffer-size)")

    # ── S3 auth ───────────────────────────────────────────────────────────────
    grp.add_argument("--s3-anon",    action="store_true",    help="Anonymous S3 access")
    grp.add_argument("--s3-profile", type=str, default=None, help="AWS credentials profile")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(df: Optional[pd.DataFrame] = None) -> None:
    """
    Entry point.

    Parameters
    ----------
    df : optional pandas DataFrame
        When supplied, used as the data source directly (small-data / API mode).
        CLI flags ``--s3-uris`` / ``--local-paths`` are ignored.
    """
    args = _parse_args()

    cfg = replace(
        TrainConfig(),
        num_epochs     = args.epochs,
        gradient_steps = args.gradient_steps,
        batch_size     = args.batch_size,
        buffer_size    = args.buffer_size,
        eval_every     = args.eval_every,
        seed           = args.seed,
        device         = args.device,
        chunk_size     = args.chunk_size,
        feature_cols   = args.feature_cols,
        label_col      = args.label_col,
        s3_uris        = args.s3_uris,
        local_paths    = args.local_paths,
        s3_anon        = args.s3_anon,
        s3_profile     = args.s3_profile,
    )

    np.random.seed(cfg.seed)

    # 1. Resolve data source, fit scaler, get test set
    source, X_test, y_test, scaler, obs_dim, n_classes, target_names = \
        load_dataset(cfg, df=df)

    # 2. LightGBM baseline (streams data, stays memory-bounded)
    lgbm_results = run_lgbm_baseline(
        cfg, source, scaler, X_test, y_test, target_names
    )

    # 3. Build AWR-GBRL model + PER buffer
    model = build_model(cfg, obs_dim, n_classes)

    # 4. Stream all training data into the replay buffer chunk-by-chunk
    fill_buffer_streaming(model, source, scaler, cfg)

    # 5. Train and evaluate
    train(
        cfg, model,
        X_test, y_test, n_classes, target_names,
        lgbm_results,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
