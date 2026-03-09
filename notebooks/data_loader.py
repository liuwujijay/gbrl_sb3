"""
data_loader.py
──────────────
Unified streaming data layer for AWR-GBRL offline training.

Supports three source types, all producing the same chunked iterator:

  1. pandas DataFrame  — in-memory, for small / unit-test data
  2. Local files       — CSV or Parquet, optionally a glob pattern
  3. S3 daily folders  — format  s3://bucket/prefix/folder-YYYY-MM-DD
                         or the short form  s3:bucket/key/folder-N

All sources yield (X_chunk, y_chunk) numpy arrays of at most
``chunk_size`` rows so the replay buffer is never overwhelmed.

S3 folder discovery
-------------------
Given a list of S3 URI prefixes (one per day), the loader:
  1. Lists all .parquet / .csv files under each prefix using s3fs.
  2. Sorts the prefix list lexicographically — folders named by date
     (YYYY-MM-DD) or sequence number naturally sort oldest → newest.
  3. Streams files one at a time, yielding chunks without loading the
     entire day into memory.

Usage
-----
    # pandas (small data)
    src = DataSource.from_dataframe(df, feature_cols, label_col)

    # local files
    src = DataSource.from_local("data/train.parquet", feature_cols, label_col)

    # S3 daily partitions
    src = DataSource.from_s3(
        uris=["s3://bucket/data/2026-03-06",
              "s3://bucket/data/2026-03-07",
              "s3://bucket/data/2026-03-08"],
        feature_cols=feature_cols,
        label_col=label_col,
    )

    for X, y in src.stream(chunk_size=1024, scaler=fitted_scaler):
        fill_buffer(model, X, y)
"""

from __future__ import annotations

import io
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

log = logging.getLogger("awr_iris")

# ── optional heavy imports (only needed for S3 / Parquet) ────────────────────
try:
    import s3fs                          # type: ignore[import]
    _S3FS_AVAILABLE = True
except ImportError:
    _S3FS_AVAILABLE = False

try:
    import pyarrow.parquet as pq         # type: ignore[import]
    _PARQUET_AVAILABLE = True
except ImportError:
    _PARQUET_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_s3_uri(uri: str) -> str:
    """
    Accept both the standard ``s3://`` scheme and the short ``s3:`` form
    used in the project (e.g. ``s3:bucket/key/folder``).

    Returns a canonical ``s3://bucket/key/folder`` string.
    """
    if uri.startswith("s3://"):
        return uri
    if uri.startswith("s3:"):
        return "s3://" + uri[3:].lstrip("/")
    raise ValueError(f"Not a valid S3 URI: {uri!r}")


def _df_to_arrays(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and cast feature / label arrays from a DataFrame chunk."""
    missing = [c for c in feature_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int64)
    return X, y


def _iter_parquet_batches(
    path_or_buf,
    feature_cols: List[str],
    label_col: str,
    chunk_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Stream a single Parquet file in row-group batches without loading it
    fully into memory.  Each yielded chunk is ≤ chunk_size rows.
    """
    if not _PARQUET_AVAILABLE:
        raise ImportError("pyarrow is required for Parquet support: pip install pyarrow")

    all_cols = list(dict.fromkeys(feature_cols + [label_col]))
    pf = pq.ParquetFile(path_or_buf)

    buf_X: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []
    buf_len = 0

    for batch in pf.iter_batches(batch_size=chunk_size, columns=all_cols):
        df = batch.to_pandas()
        X, y = _df_to_arrays(df, feature_cols, label_col)
        buf_X.append(X)
        buf_y.append(y)
        buf_len += len(X)

        while buf_len >= chunk_size:
            Xc = np.concatenate(buf_X)
            yc = np.concatenate(buf_y)
            yield Xc[:chunk_size], yc[:chunk_size]
            # keep remainder
            buf_X = [Xc[chunk_size:]]
            buf_y = [yc[chunk_size:]]
            buf_len = len(buf_X[0])

    # flush tail
    if buf_len > 0:
        yield np.concatenate(buf_X), np.concatenate(buf_y)


def _iter_csv_batches(
    path_or_buf,
    feature_cols: List[str],
    label_col: str,
    chunk_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stream a CSV file in chunks using pandas chunked reader."""
    all_cols = list(dict.fromkeys(feature_cols + [label_col]))
    reader = pd.read_csv(path_or_buf, usecols=all_cols, chunksize=chunk_size)
    for chunk in reader:
        yield _df_to_arrays(chunk, feature_cols, label_col)


# ──────────────────────────────────────────────────────────────────────────────
# S3 file listing
# ──────────────────────────────────────────────────────────────────────────────

def _list_s3_files(
    fs: "s3fs.S3FileSystem",
    prefix: str,
    extensions: tuple[str, ...] = (".parquet", ".csv"),
) -> list[str]:
    """
    Return sorted list of all files under *prefix* with matching extension.
    Uses s3fs.glob for efficient listing.
    """
    # strip leading s3:// for s3fs (it works with or without, but be explicit)
    bare = prefix.replace("s3://", "")
    found: list[str] = []
    for ext in extensions:
        found += fs.glob(f"{bare}/**/*{ext}") + fs.glob(f"{bare}/*{ext}")
    # deduplicate and sort (date-named files sort chronologically)
    return sorted(set(found))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class _SourceKind(Enum):
    DATAFRAME = auto()
    LOCAL     = auto()
    S3        = auto()


class DataSource:
    """
    Unified streaming data source.

    Do not construct directly — use the class-method factories:
      - ``DataSource.from_dataframe``
      - ``DataSource.from_local``
      - ``DataSource.from_s3``
    """

    def __init__(
        self,
        kind: _SourceKind,
        feature_cols: List[str],
        label_col: str,
        *,
        # dataframe
        df: Optional[pd.DataFrame] = None,
        # local
        local_paths: Optional[List[Path]] = None,
        # s3
        s3_uris: Optional[List[str]] = None,
        s3_anon: bool = False,
        s3_profile: Optional[str] = None,
    ) -> None:
        self._kind         = kind
        self.feature_cols  = feature_cols
        self.label_col     = label_col
        self._df           = df
        self._local_paths  = local_paths or []
        self._s3_uris      = s3_uris or []
        self._s3_anon      = s3_anon
        self._s3_profile   = s3_profile

    # ── factories ────────────────────────────────────────────────────────────

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
    ) -> "DataSource":
        """
        Wrap an in-memory pandas DataFrame.

        Intended for small datasets (unit tests, Iris-scale).
        The full DataFrame is held in memory; chunking is applied lazily
        during ``stream()``.
        """
        if df.empty:
            raise ValueError("DataFrame is empty.")
        return cls(_SourceKind.DATAFRAME, feature_cols, label_col, df=df)

    @classmethod
    def from_local(
        cls,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        feature_cols: List[str],
        label_col: str,
    ) -> "DataSource":
        """
        Stream one or more local CSV or Parquet files.

        *paths* may be:
          - a single file path (str or Path)
          - a list of paths
          - a glob string (e.g. ``"data/train_*.parquet"``)
        """
        import glob as _glob

        if isinstance(paths, (str, Path)):
            paths = [paths]

        resolved: list[Path] = []
        for p in paths:
            matched = _glob.glob(str(p))
            if matched:
                resolved.extend(Path(m) for m in sorted(matched))
            else:
                resolved.append(Path(p))

        missing = [p for p in resolved if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Files not found: {missing}")

        return cls(_SourceKind.LOCAL, feature_cols, label_col,
                   local_paths=resolved)

    @classmethod
    def from_s3(
        cls,
        uris: List[str],
        feature_cols: List[str],
        label_col: str,
        *,
        anon: bool = False,
        profile: Optional[str] = None,
    ) -> "DataSource":
        """
        Stream Parquet/CSV files from S3 daily-partitioned folders.

        *uris* is a list of S3 prefixes — one per day — in either form::

            s3://bucket/data/2026-03-06
            s3:bucket/data/2026-03-06        # short form, also accepted

        The list is sorted lexicographically so date-named folders are
        processed oldest → newest automatically.

        Parameters
        ----------
        uris     : List of S3 prefix URIs (one per day / partition).
        anon     : Use anonymous (unsigned) S3 access (public buckets).
        profile  : AWS credentials profile name (from ~/.aws/credentials).
        """
        if not _S3FS_AVAILABLE:
            raise ImportError("s3fs is required: pip install s3fs")

        canonical = sorted(_normalise_s3_uri(u) for u in uris)
        return cls(_SourceKind.S3, feature_cols, label_col,
                   s3_uris=canonical, s3_anon=anon, s3_profile=profile)

    # ── streaming ─────────────────────────────────────────────────────────────

    def stream(
        self,
        chunk_size: int = 4_096,
        scaler=None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield ``(X_chunk, y_chunk)`` pairs of at most *chunk_size* rows.

        Parameters
        ----------
        chunk_size : int
            Maximum rows per yielded chunk.  Keep this ≤ replay-buffer
            size so a single chunk never overflows the buffer.
        scaler     : optional sklearn-compatible scaler with a
                     ``transform(X)`` method.  Applied to X before yield.
                     Pass ``None`` to skip scaling.

        Yields
        ------
        X : np.ndarray  shape (≤chunk_size, n_features)  dtype float32
        y : np.ndarray  shape (≤chunk_size,)              dtype int64
        """
        if self._kind == _SourceKind.DATAFRAME:
            yield from self._stream_dataframe(chunk_size, scaler)
        elif self._kind == _SourceKind.LOCAL:
            yield from self._stream_local(chunk_size, scaler)
        elif self._kind == _SourceKind.S3:
            yield from self._stream_s3(chunk_size, scaler)

    def _apply_scaler(
        self,
        X: np.ndarray,
        scaler,
    ) -> np.ndarray:
        if scaler is None:
            return X
        return scaler.transform(X).astype(np.float32)

    # ── dataframe backend ─────────────────────────────────────────────────────

    def _stream_dataframe(
        self,
        chunk_size: int,
        scaler,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        df = self._df
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start: start + chunk_size]
            X, y  = _df_to_arrays(chunk, self.feature_cols, self.label_col)
            yield self._apply_scaler(X, scaler), y

    # ── local file backend ────────────────────────────────────────────────────

    def _stream_local(
        self,
        chunk_size: int,
        scaler,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        for path in self._local_paths:
            suffix = path.suffix.lower()
            log.debug(f"  streaming local file: {path}")
            if suffix == ".parquet":
                it = _iter_parquet_batches(
                    str(path), self.feature_cols, self.label_col, chunk_size)
            elif suffix == ".csv":
                it = _iter_csv_batches(
                    str(path), self.feature_cols, self.label_col, chunk_size)
            else:
                raise ValueError(
                    f"Unsupported file format: {suffix!r}. Use .parquet or .csv")
            for X, y in it:
                yield self._apply_scaler(X, scaler), y

    # ── S3 backend ────────────────────────────────────────────────────────────

    def _stream_s3(
        self,
        chunk_size: int,
        scaler,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        if not _S3FS_AVAILABLE:
            raise ImportError("s3fs is required: pip install s3fs")

        fs_kwargs: dict = {"anon": self._s3_anon}
        if self._s3_profile:
            fs_kwargs["profile"] = self._s3_profile
        fs = s3fs.S3FileSystem(**fs_kwargs)

        total_folders = len(self._s3_uris)
        for folder_idx, uri in enumerate(self._s3_uris, 1):
            files = _list_s3_files(fs, uri)
            if not files:
                log.warning(f"  S3 folder {uri!r}: no .parquet/.csv files found, skipping")
                continue

            log.info(f"  S3 folder [{folder_idx}/{total_folders}] "
                     f"{uri}  ({len(files)} file(s))")

            for s3_path in files:
                suffix = s3_path.rsplit(".", 1)[-1].lower()
                log.debug(f"    reading s3://{s3_path}")

                with fs.open(f"s3://{s3_path}", "rb") as fh:
                    raw = io.BytesIO(fh.read())  # buffer → avoids repeated network seeks

                if suffix == "parquet":
                    it = _iter_parquet_batches(
                        raw, self.feature_cols, self.label_col, chunk_size)
                elif suffix == "csv":
                    it = _iter_csv_batches(
                        raw, self.feature_cols, self.label_col, chunk_size)
                else:
                    log.warning(f"    skipping unsupported file: s3://{s3_path}")
                    continue

                for X, y in it:
                    yield self._apply_scaler(X, scaler), y

    # ── convenience: fit scaler on a sample ──────────────────────────────────

    def fit_scaler(
        self,
        scaler,
        max_rows: int = 50_000,
        chunk_size: int = 4_096,
    ):
        """
        Fit *scaler* on the first *max_rows* rows of this source (no labels
        needed).  Useful when the full dataset does not fit in memory.

        Returns the fitted scaler.
        """
        collected: list[np.ndarray] = []
        total = 0
        for X, _ in self.stream(chunk_size=chunk_size, scaler=None):
            collected.append(X)
            total += len(X)
            if total >= max_rows:
                break
        X_sample = np.concatenate(collected)[:max_rows]
        scaler.fit(X_sample)
        log.info(f"Scaler fitted on {len(X_sample):,} rows "
                 f"({X_sample.shape[1]} features)")
        return scaler

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        kind = self._kind.name
        if self._kind == _SourceKind.DATAFRAME:
            detail = f"df rows={len(self._df)}"
        elif self._kind == _SourceKind.LOCAL:
            detail = f"files={len(self._local_paths)}"
        else:
            detail = f"s3_folders={len(self._s3_uris)}"
        return (f"DataSource({kind}  {detail}  "
                f"features={self.feature_cols}  label={self.label_col!r})")
