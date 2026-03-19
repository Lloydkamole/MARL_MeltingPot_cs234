"""Checkpoint 4 metrics pipeline utilities.

Computes and logs:
  - U: Utilitarian metric (sum of cumulative rewards across players)
  - S: Sustainability proxy (average utilitarian reward per episode step)
  - E: Gini coefficient over cumulative rewards (inequality)
"""

import csv
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np


def gini_coefficient(values: Iterable[float]) -> float:
    """Returns Gini coefficient in [0, 1] for a 1D collection of values.

    If values contain negatives, they are shifted so the minimum is 0.
    If all values are 0 (or empty), returns 0.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    min_val = float(np.min(arr))
    if min_val < 0.0:
        arr = arr - min_val
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    arr.sort()
    n = arr.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * float(np.sum(idx * arr)) / (n * total)) - ((n + 1.0) / n)
    return float(np.clip(gini, 0.0, 1.0))


@dataclass
class MetricsSnapshot:
    utilitarian: float
    sustainability: float
    gini: float

    def to_dashboard_dict(self) -> Dict[str, float]:
        return {
            "U": self.utilitarian,
            "S": self.sustainability,
            "E": self.gini,
        }


class MetricsPipeline:
    """Streaming checkpoint-4 metrics logger and live metrics provider."""

    def __init__(self, csv_path: str):
        self._csv_path = csv_path
        self._episode_index = 0
        self._episode_step = 0
        self._file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "global_step",
                "episode",
                "episode_step",
                "U",
                "S",
                "E",
            ],
        )
        self._writer.writeheader()
        self._file.flush()

    @property
    def csv_path(self) -> str:
        return self._csv_path

    def on_step(
        self,
        global_step: int,
        rewards: Sequence[float],
        cumulative_rewards: Sequence[float],
    ) -> MetricsSnapshot:
        """Updates metrics for one environment step and writes one CSV row."""
        del rewards  # We currently derive metrics from cumulative episode returns.
        self._episode_step += 1

        utilitarian = float(np.sum(cumulative_rewards))
        sustainability = utilitarian / float(max(1, self._episode_step))
        gini = gini_coefficient(cumulative_rewards)

        snapshot = MetricsSnapshot(
            utilitarian=utilitarian,
            sustainability=sustainability,
            gini=gini,
        )
        self._writer.writerow(
            {
                "global_step": int(global_step),
                "episode": int(self._episode_index),
                "episode_step": int(self._episode_step),
                "U": snapshot.utilitarian,
                "S": snapshot.sustainability,
                "E": snapshot.gini,
            }
        )
        self._file.flush()
        return snapshot

    def on_episode_reset(self) -> None:
        self._episode_index += 1
        self._episode_step = 0

    def close(self) -> None:
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
