import numpy as np
import joblib
from pipeline.models.base import BaseModel


class CSPLDAModel(BaseModel):
    def __init__(self):
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
        self.lda = LinearDiscriminantAnalysis()

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_feat = self.csp.fit_transform(X_train, y_train)
        self.lda.fit(X_feat, y_train)

    def predict(self, epoch: np.ndarray) -> tuple[int, float]:
        feat = self.csp.transform(epoch[np.newaxis])
        pred = int(self.lda.predict(feat)[0])
        conf = float(self.lda.predict_proba(feat)[0].max())
        return pred, conf

    def save(self, path: str) -> None:
        joblib.dump({"csp": self.csp, "lda": self.lda}, path)

    def load(self, path: str) -> None:
        bundle = joblib.load(path)
        self.csp = bundle["csp"]
        self.lda = bundle["lda"]
