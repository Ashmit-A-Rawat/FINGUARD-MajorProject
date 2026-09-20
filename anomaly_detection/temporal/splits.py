"""Time-aware train/validation/test splitting with episode purging."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalSplit:
    train: np.ndarray  # positions (indices into the feature table)
    val: np.ndarray
    test: np.ndarray
    boundaries: tuple[np.datetime64, np.datetime64]
    purged_rows: int
    purged_episodes: int

    def check_no_future_leakage(self, timestamps: np.ndarray) -> None:
        if not (
            timestamps[self.train].max() < timestamps[self.val].min()
            and timestamps[self.val].max() < timestamps[self.test].min()
        ):
            raise AssertionError("time ordering violated between splits")


def temporal_split(
    timestamps: np.ndarray,
    episode_ids: np.ndarray,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> TemporalSplit:
    """Chronological split. Rows of an anomaly *episode* that straddles a boundary are dropped
    from every split, so no episode is partly in training and partly in evaluation.

    Episode ids are used only to place the split, never as features.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    ts = timestamps.astype("datetime64[ns]").astype("int64")
    b1 = np.quantile(ts, fractions[0])
    b2 = np.quantile(ts, fractions[0] + fractions[1])
    keep = np.ones(len(ts), dtype=bool)
    straddling = 0
    for episode in np.unique(episode_ids[episode_ids != ""]):
        rows = np.flatnonzero(episode_ids == episode)
        lo, hi = ts[rows].min(), ts[rows].max()
        if any(lo < b <= hi for b in (b1, b2)):
            keep[rows] = False
            straddling += 1
    positions = np.arange(len(ts))
    return TemporalSplit(
        train=positions[keep & (ts < b1)],
        val=positions[keep & (ts >= b1) & (ts < b2)],
        test=positions[keep & (ts >= b2)],
        boundaries=(np.datetime64(int(b1), "ns"), np.datetime64(int(b2), "ns")),
        purged_rows=int((~keep).sum()),
        purged_episodes=straddling,
    )
