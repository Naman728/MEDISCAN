import io
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except Exception:
    tf = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lung_xray_model.h5"


class LungXrayModelLoader:
    def __init__(self) -> None:
        self.model = self._load_model_if_possible()

    def _load_model_if_possible(self) -> Optional[Any]:
        if tf is None:
            return None
        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
            return None
        try:
            return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        except Exception:
            return None

    def _preprocess_for_tf(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        target_size = (224, 224)

        if self.model is not None and hasattr(self.model, "input_shape") and self.model.input_shape is not None:
            shape = self.model.input_shape
            if isinstance(shape, list):
                shape = shape[0]
            if len(shape) >= 3 and shape[1] and shape[2]:
                target_size = (int(shape[2]), int(shape[1]))

        image = image.resize(target_size)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _stub_predict(self, image_bytes: bytes) -> Dict[str, Any]:
        # Deterministic placeholder score from image intensity for hackathon flow.
        image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((128, 128))
        mean_intensity = float(np.asarray(image, dtype=np.float32).mean() / 255.0)
        confidence = round(min(max(mean_intensity, 0.05), 0.95), 4)
        prediction = "possible_abnormality" if confidence >= 0.5 else "likely_normal"
        return {
            "prediction": prediction,
            "confidence": confidence,
            "note": "Using stub predictor (TensorFlow model unavailable or incompatible).",
        }

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        if self.model is None:
            return self._stub_predict(image_bytes)

        input_tensor = self._preprocess_for_tf(image_bytes)
        output = self.model.predict(input_tensor, verbose=0)
        output = np.array(output)

        if output.ndim == 1:
            output = np.expand_dims(output, axis=0)

        if output.shape[-1] == 1:
            prob_abnormal = float(1 / (1 + np.exp(-output[0, 0])))
            prediction = "abnormal" if prob_abnormal >= 0.5 else "normal"
            confidence = prob_abnormal if prediction == "abnormal" else 1 - prob_abnormal
        else:
            probs = output[0]
            probs = np.exp(probs - np.max(probs))
            probs = probs / np.sum(probs)
            idx = int(np.argmax(probs))
            labels = ["normal", "abnormal", "other"]
            prediction = labels[idx] if idx < len(labels) else f"class_{idx}"
            confidence = float(probs[idx])

        return {"prediction": prediction, "confidence": round(confidence, 4)}