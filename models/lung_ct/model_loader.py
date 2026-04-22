import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    import tensorflow as tf
except Exception:
    tf = None

BASE_DIR = Path(__file__).resolve().parent
PT_MODEL_PATH = BASE_DIR / "lung_ct_model.pth"
TF_MODEL_PATH = BASE_DIR / "lung_ct_model.h5"
ARCH_FILE_PATH = BASE_DIR / "model_def.py"
LABELS_PATH = BASE_DIR / "labels.json"

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CT_FEATURES_DEFAULT = [
    "Spiculated Nodule",
    "Ground-Glass Opacity",
    "Cavitary Lesion",
    "Lymph Node Involvement",
    "Local Invasion",
    "Malignant Effusion",
    "Upper Lobe Predominant",
]


class LungCTModelLoader:
    def __init__(self) -> None:
        self.device = "cpu"
        self.torch_model = None
        self.tf_model = None
        self.backend = "stub"
        self.class_names = CT_FEATURES_DEFAULT.copy()

        self._load_labels()
        self._load_torch_model_if_possible()
        if self.torch_model is None:
            self._load_tf_model_if_possible()

    def _load_labels(self) -> None:
        if not LABELS_PATH.exists():
            return
        try:
            labels_data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
            if isinstance(labels_data, list) and labels_data:
                self.class_names = [str(item) for item in labels_data]
        except Exception:
            pass

    def _extract_state_dict(self, loaded_obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(loaded_obj, dict):
            if "model_state_dict" in loaded_obj and isinstance(loaded_obj["model_state_dict"], dict):
                return loaded_obj["model_state_dict"]
            if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
                return loaded_obj["state_dict"]
            if loaded_obj and all(isinstance(k, str) for k in loaded_obj.keys()):
                return loaded_obj
        return None

    def _resolve_model_builder(self):
        if not ARCH_FILE_PATH.exists():
            return None
        import importlib.util

        spec = importlib.util.spec_from_file_location("lung_ct_model_def", ARCH_FILE_PATH)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "build_model", None)

    def _infer_num_heads(self, state_dict: Dict[str, Any]) -> int:
        head_indices: List[int] = []
        for key in state_dict.keys():
            if key.startswith("heads."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    head_indices.append(int(parts[1]))
        if head_indices:
            return max(head_indices) + 1
        return 1

    def _load_torch_model_if_possible(self) -> None:
        if torch is None:
            return
        if not PT_MODEL_PATH.exists() or PT_MODEL_PATH.stat().st_size == 0:
            return

        loaded_obj = torch.load(str(PT_MODEL_PATH), map_location=self.device, weights_only=False)

        if nn is not None and isinstance(loaded_obj, nn.Module):
            self.torch_model = loaded_obj
            self.torch_model.eval()
            self.backend = "torch"
            return

        state_dict = self._extract_state_dict(loaded_obj)
        if state_dict is None:
            return

        model_builder = self._resolve_model_builder()
        if model_builder is None:
            return

        num_heads = self._infer_num_heads(state_dict)
        try:
            model = model_builder(num_heads=num_heads)
        except TypeError:
            model = model_builder()

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.torch_model = model
        self.backend = "torch"

    def _load_tf_model_if_possible(self) -> None:
        if tf is None:
            return
        if not TF_MODEL_PATH.exists() or TF_MODEL_PATH.stat().st_size == 0:
            return
        try:
            self.tf_model = tf.keras.models.load_model(str(TF_MODEL_PATH), compile=False)
            self.backend = "tensorflow"
        except Exception:
            self.tf_model = None

    def _transform_torch(self, image: Image.Image):
        if torch is None:
            raise RuntimeError("PyTorch is not installed.")
        image = image.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    def _ndarray_to_data_uri(self, arr: np.ndarray) -> str:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        buffer = io.BytesIO()
        Image.fromarray(arr).save(buffer, format="PNG")
        enc = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{enc}"

    def _get_normal_ref(self, size: int = 224):
        np.random.seed(42)
        cx, cy = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        ref = np.random.normal(90, 10, (size, size)).astype(np.float32)
        body = ((x - cx) ** 2 / (cx * 0.92) ** 2 + (y - cy) ** 2 / (cy * 0.88) ** 2) <= 1
        ref[~body] = 0
        fat = body & (((x - cx) ** 2 / (cx * 0.80) ** 2 + (y - cy) ** 2 / (cy * 0.76) ** 2) >= 1)
        ref[fat] = np.random.normal(70, 8, ref[fat].shape)
        ll = ((x - cx * 0.62) ** 2 / (cx * 0.26) ** 2 + (y - cy * 0.55) ** 2 / (cy * 0.32) ** 2) <= 1
        ref[ll] = np.random.normal(25, 5, ref[ll].shape)
        rl = ((x - cx * 1.30) ** 2 / (cx * 0.28) ** 2 + (y - cy * 0.55) ** 2 / (cy * 0.34) ** 2) <= 1
        ref[rl] = np.random.normal(25, 5, ref[rl].shape)
        ht = ((x - cx * 0.85) ** 2 / (cx * 0.20) ** 2 + (y - cy * 0.65) ** 2 / (cy * 0.22) ** 2) <= 1
        ref[ht] = np.random.normal(110, 8, ref[ht].shape)
        return np.stack([np.clip(ref, 0, 255).astype(np.uint8)] * 3, axis=-1)

    def _deviation_map(self, patient_np: np.ndarray, normal_ref: np.ndarray):
        p = patient_np.astype(np.float32) / 255.0
        n = normal_ref.astype(np.float32) / 255.0
        diff = np.abs(p - n).mean(axis=2)
        return (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    def _generate_gradcam(self, image: Image.Image):
        if self.torch_model is None or torch is None:
            return None
        grads, acts = [], []

        def fwd(_m, _i, o):
            acts.append(o.detach())

        def bwd(_m, _gi, go):
            grads.append(go[0].detach())

        layer = self.torch_model.features[-1]
        fh = layer.register_forward_hook(fwd)
        # compatibility with newer torch
        try:
            bh = layer.register_full_backward_hook(bwd)
        except Exception:
            bh = layer.register_backward_hook(bwd)

        t = self._transform_torch(image).requires_grad_(True)
        out = self.torch_model(t)
        self.torch_model.zero_grad(set_to_none=True)
        out.mean().backward()

        fh.remove()
        bh.remove()

        if not grads or not acts:
            return None

        w = grads[0][0].mean(dim=(1, 2))
        cam = (w[:, None, None] * acts[0][0]).sum(dim=0)
        cam = torch.relu(cam).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    def _find_regions(self, patient_np: np.ndarray, dev: np.ndarray, gradcam: Optional[np.ndarray]):
        combined = dev * 0.6 + (gradcam if gradcam is not None else dev) * 0.4
        threshold = np.percentile(combined, 80)
        binary = (combined > threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        annotated = patient_np.copy()
        region_details = []
        region_id = 0
        for i in range(1, min(num_labels, 8)):
            x, y, w, h, area = stats[i]
            if area < 100:
                continue
            region_id += 1
            region_mask = labels == i
            conf = int(min(100, combined[region_mask].mean() * 200))
            color = (220, 38, 38) if conf > 70 else (217, 119, 6) if conf > 40 else (22, 163, 74)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, f"R{region_id}", (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            region_details.append(
                {
                    "id": region_id,
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area_pct": round(area / (IMG_SIZE * IMG_SIZE) * 100, 1),
                    "confidence": conf,
                }
            )
        return annotated, region_id, region_details

    def _predict_torch(self, image_bytes: bytes) -> Dict[str, Any]:
        if self.torch_model is None or torch is None:
            raise RuntimeError("Torch model is not loaded")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        patient_np = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
        normal_ref = self._get_normal_ref(IMG_SIZE)
        dev = self._deviation_map(patient_np, normal_ref)

        t = self._transform_torch(image)
        with torch.no_grad():
            out = self.torch_model(t).squeeze(0).detach().cpu().numpy()

        labels = self.class_names[: len(out)] if len(self.class_names) >= len(out) else [f"feature_{i+1}" for i in range(len(out))]

        # model may already output sigmoid probabilities
        if np.min(out) < 0.0 or np.max(out) > 1.0:
            scores = 1.0 / (1.0 + np.exp(-out))
        else:
            scores = out

        head_scores = [{"label": labels[i], "score": round(float(scores[i]), 4)} for i in range(len(scores))]

        gradcam = self._generate_gradcam(image)
        annotated, n_regions, reg_details = self._find_regions(patient_np, dev, gradcam)

        heatmaps = []

        hm_dev = cv2.applyColorMap(np.uint8(255 * dev), cv2.COLORMAP_HOT)
        hm_dev = cv2.cvtColor(hm_dev, cv2.COLOR_BGR2RGB)
        dev_overlay = cv2.addWeighted(patient_np, 0.5, hm_dev, 0.5, 0)
        heatmaps.append({"label": "Deviation Overlay", "score": round(float(np.mean(dev)), 4), "src": self._ndarray_to_data_uri(dev_overlay)})

        heatmaps.append({"label": "Annotated Regions", "score": round(float(min(1.0, n_regions / 7.0)), 4), "src": self._ndarray_to_data_uri(annotated)})

        if gradcam is not None:
            hm_grad = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_INFERNO)
            hm_grad = cv2.cvtColor(hm_grad, cv2.COLOR_BGR2RGB)
            grad_overlay = cv2.addWeighted(patient_np, 0.45, hm_grad, 0.55, 0)
            heatmaps.append({"label": "Grad-CAM Heatmap", "score": round(float(np.max(gradcam)), 4), "src": self._ndarray_to_data_uri(hm_grad)})
            heatmaps.append({"label": "Grad-CAM Overlay", "score": round(float(np.mean(gradcam)), 4), "src": self._ndarray_to_data_uri(grad_overlay)})

        top_idx = int(np.argmax(scores)) if len(scores) else 0
        prediction = labels[top_idx] if labels else "prediction"
        confidence = float(scores[top_idx]) if len(scores) else 0.0

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "head_scores": head_scores,
            "heatmaps": heatmaps,
            "n_regions": n_regions,
            "high_n": int(sum(1 for s in scores if s > 0.6)),
            "mean_s": round(float(np.mean(scores)) * 100, 1),
            "reg_details": reg_details,
            "backend": "torch",
            "input_sha12": hashlib.sha256(image_bytes).hexdigest()[:12],
        }

    def _predict_tf(self, image_bytes: bytes) -> Dict[str, Any]:
        if self.tf_model is None:
            raise RuntimeError("TensorFlow model is not loaded")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        patient_np = np.array(image.resize((IMG_SIZE, IMG_SIZE)))

        arr = np.asarray(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        out = self.tf_model.predict(arr, verbose=0)
        out = np.array(out).reshape(-1)
        if np.min(out) < 0.0 or np.max(out) > 1.0:
            scores = 1.0 / (1.0 + np.exp(-out))
        else:
            scores = out

        labels = self.class_names[: len(scores)] if len(self.class_names) >= len(scores) else [f"feature_{i+1}" for i in range(len(scores))]
        head_scores = [{"label": labels[i], "score": round(float(scores[i]), 4)} for i in range(len(scores))]

        gray = np.asarray(image.resize((IMG_SIZE, IMG_SIZE)).convert("L"), dtype=np.float32) / 255.0
        hm = cv2.applyColorMap(np.uint8(255 * gray), cv2.COLORMAP_INFERNO)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(patient_np, 0.45, hm, 0.55, 0)

        top_idx = int(np.argmax(scores)) if len(scores) else 0
        prediction = labels[top_idx] if labels else "prediction"
        confidence = float(scores[top_idx]) if len(scores) else 0.0

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "head_scores": head_scores,
            "heatmaps": [
                {"label": "Approx Heatmap", "score": round(float(np.max(gray)), 4), "src": self._ndarray_to_data_uri(hm)},
                {"label": "Approx Overlay", "score": round(float(np.mean(gray)), 4), "src": self._ndarray_to_data_uri(overlay)},
            ],
            "backend": "tensorflow",
            "input_sha12": hashlib.sha256(image_bytes).hexdigest()[:12],
            "note": "TensorFlow backend active; using approximate heatmap.",
        }

    def _predict_stub(self, image_bytes: bytes) -> Dict[str, Any]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        patient_np = np.asarray(image, dtype=np.uint8)
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        conf = float(np.clip(0.35 + gray.std(), 0.05, 0.95))
        pred = "suspicious_pattern" if conf >= 0.55 else "likely_normal"
        hm = cv2.applyColorMap(np.uint8(255 * gray), cv2.COLORMAP_HOT)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(patient_np, 0.5, hm, 0.5, 0)

        return {
            "prediction": pred,
            "confidence": round(conf, 4),
            "head_scores": [{"label": pred, "score": round(conf, 4)}],
            "heatmaps": [{"label": "Demo Overlay", "score": round(conf, 4), "src": self._ndarray_to_data_uri(overlay)}],
            "backend": "stub",
            "input_sha12": hashlib.sha256(image_bytes).hexdigest()[:12],
            "note": "Using stub predictor (CT model unavailable or incompatible).",
        }

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        if self.backend == "torch" and self.torch_model is not None:
            return self._predict_torch(image_bytes)
        if self.backend == "tensorflow" and self.tf_model is not None:
            return self._predict_tf(image_bytes)
        return self._predict_stub(image_bytes)
