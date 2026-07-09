import numpy as np
import torch
from pipeline.models.base import BaseModel


class EEGNetModel(BaseModel):
    def __init__(self, n_chans: int = 3, n_times: int = 750, sfreq: float = 250.0):
        from braindecode.models import EEGNet
        from braindecode import EEGClassifier

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.n_chans = n_chans
        self.n_times = n_times

        self.net = EEGNet(
            n_chans=n_chans,
            n_outputs=2,
            n_times=n_times,
            final_conv_length="auto",
        ).to(device)

        self.clf = EEGClassifier(
            self.net,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            optimizer__lr=0.0001,
            batch_size=32,
            max_epochs=150,
            device=device,
            verbose=1,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.clf.fit(X_train, y_train.astype(np.int64))

    def predict(self, epoch: np.ndarray) -> tuple[int, float]:
        self.net.eval()
        x = torch.tensor(epoch[np.newaxis], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.net(x), dim=1)[0].cpu()
        return int(probs.argmax()), float(probs.max())

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()
