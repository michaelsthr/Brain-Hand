import json
import os
import time
from pathlib import Path

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS

from pipeline.data_loader import load_dataset
from pipeline.models.csp_lda import CSPLDAModel
from pipeline.models.eegnet import EEGNetModel
from pipeline.models.atcnet import ATCNetModel
from pipeline.models.mirepnet import MIRepNetModel

app = Flask(__name__)
CORS(app)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "trained_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WEB_MODEL_DIR = Path("static/model")
if not (WEB_MODEL_DIR / "scene.xml").exists():
    from simulation.hand import export_web_assets
    print("[hand] Exporting web scene to static/model …")
    export_web_assets(str(WEB_MODEL_DIR))


@app.after_request
def add_headers(r):
    r.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    r.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return r


X_train, y_train, X_test, y_test = load_dataset(subject=1)

models: dict = {}

def _init_model(name: str, model, ext: str):
    path = MODEL_DIR / f"{name}.{ext}"
    if path.exists():
        print(f"[{name}] Loading from {path}")
        model.load(str(path))
    else:
        print(f"[{name}] Training …")
        model.train(X_train, y_train)
        model.save(str(path))
        print(f"[{name}] Saved to {path}")
    models[name] = model

_init_model("csp_lda", CSPLDAModel(), "pkl")
_init_model("eegnet",  EEGNetModel(n_chans=X_train.shape[1], n_times=X_train.shape[2]), "pt")
_init_model("atcnet",  ATCNetModel(), "npy")
_init_model("mirepnet", MIRepNetModel(), "npy")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/models")
def list_models():
    return jsonify(list(models.keys()))


@app.route("/accuracy")
def accuracy():
    model_name = request.args.get("model", "csp_lda")
    model = models.get(model_name, models["csp_lda"])
    correct = sum(model.predict(ep)[0] == gt for ep, gt in zip(X_test, y_test))
    return jsonify({"model": model_name, "accuracy": correct / len(y_test), "n": len(y_test)})


@app.route("/stream")
def stream():
    model_name = request.args.get("model", "csp_lda")
    model = models.get(model_name, models["csp_lda"])

    def generate():
        correct = 0
        for i, (epoch, true_label) in enumerate(zip(X_test, y_test)):
            pred, conf = model.predict(epoch)

            if pred == true_label:
                correct += 1

            payload = json.dumps({
                "eeg_c3":      epoch[0, -120:].tolist(),
                "eeg_cz":      epoch[1, -120:].tolist(),
                "eeg_c4":      epoch[2, -120:].tolist(),
                "prediction":  pred,
                "true_label":  int(true_label),
                "confidence":  round(conf, 3),
                "correct":     bool(pred == true_label),
                "accuracy":    round(correct / (i + 1), 3),
                "model":       model_name,
                "trial":       i + 1,
                "total":       len(y_test),
            })
            yield f"data: {payload}\n\n"
            time.sleep(2.0)

        yield 'data: {"done": true}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
