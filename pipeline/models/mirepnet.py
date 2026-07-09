import numpy as np
from pipeline.models.base import BaseModel


class MIRepNetModel(BaseModel):
    def __init__(self):
        self._rng = np.random.default_rng(99)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def predict(self, epoch: np.ndarray) -> tuple[int, float]:
        pred = int(self._rng.integers(0, 2))
        conf = float(self._rng.uniform(0.5, 1.0))
        return pred, conf

    def save(self, path: str) -> None:
        np.save(path, np.array([]))

    def load(self, path: str) -> None:
        pass
