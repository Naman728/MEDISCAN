from pathlib import Path

from flask import Flask, jsonify, render_template, request

from models.brain.model_loader import BrainModelLoader
from models.lung_ct.model_loader import LungCTModelLoader
from models.lung_xray.model_loader import LungXrayModelLoader

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

brain_loader = BrainModelLoader()
lung_xray_loader = LungXrayModelLoader()
lung_ct_loader = LungCTModelLoader()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/brain")
def brain_page():
    return render_template("brain.html")


@app.get("/lung-xray")
def lung_xray_page():
    return render_template("lung_xray.html")


@app.get("/lung-ct")
def lung_ct_page():
    return render_template("lung_ct.html")


@app.get("/blood-sample")
def blood_sample_page():
    return render_template("blood_sample.html")


@app.get("/workspace")
def workspace_page():
    return render_template("model_template.html")


def _predict_from_loader(loader):
    if "file" not in request.files:
        return jsonify({"error": "Missing file field. Use form-data with key 'file'."}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    image_bytes = uploaded_file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        result = loader.predict(image_bytes)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


@app.post("/predict/brain")
def predict_brain():
    return _predict_from_loader(brain_loader)


@app.post("/predict/lung-xray")
def predict_lung_xray():
    return _predict_from_loader(lung_xray_loader)


@app.post("/predict/lung-ct")
def predict_lung_ct():
    return _predict_from_loader(lung_ct_loader)


if __name__ == "__main__":
    port = 5000
    app.run(host="0.0.0.0", port=port, debug=True)