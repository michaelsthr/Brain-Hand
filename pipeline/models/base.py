from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, epoch: np.ndarray) -> tuple[int, float]: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
