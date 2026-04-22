import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "brain_model.pth"
LABELS_PATH = BASE_DIR / "labels.json"
ARCH_FILE_PATH = BASE_DIR / "model_def.py"


class BrainModelLoader:
    def __init__(self) -> None:
        self.device = "cpu"
        self.model = None
        self.class_names = [
            "glioma_probability",
            "meningioma_signal",
            "pituitary_lesion",
            "edema_response",
            "hemorrhage_marker",
            "necrosis_pattern",
        ]
        self._load_labels()
        self._load_model_if_possible()

    def _load_labels(self) -> None:
        if not LABELS_PATH.exists():
            return
        try:
            labels_data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
            if isinstance(labels_data, list) and labels_data:
                self.class_names = [str(item) for item in labels_data]
        except Exception:
            pass

    def _load_model_if_possible(self) -> None:
        if torch is None:
            return
        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
            return

        loaded_obj = torch.load(str(MODEL_PATH), map_location=self.device, weights_only=False)

        if isinstance(loaded_obj, nn.Module):
            self.model = loaded_obj
            self.model.eval()
            return

        state_dict = self._extract_state_dict(loaded_obj)
        if state_dict is None:
            raise ValueError("Unsupported .pth format for Brain model.")

        model_builder = self._resolve_model_builder()
        if model_builder is None:
            raise ValueError(
                "Brain model contains only state_dict. Add models/brain/model_def.py "
                "with a build_model() function that returns the model architecture."
            )

        num_heads = self._infer_num_heads(state_dict)
        try:
            model = model_builder(num_heads=num_heads)
        except TypeError:
            model = model_builder()
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model

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

        spec = importlib.util.spec_from_file_location("brain_model_def", ARCH_FILE_PATH)
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

    def _preprocess(self, image_bytes: bytes):
        if torch is None:
            raise RuntimeError("PyTorch is not installed.")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        vis_image = np.asarray(image, dtype=np.uint8)

        image_array = vis_image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std

        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        return tensor, vis_image

    def _ensure_labels(self, n: int) -> List[str]:
        if len(self.class_names) >= n:
            return self.class_names[:n]
        labels = list(self.class_names)
        for i in range(len(labels), n):
            labels.append(f"abnormality_{i + 1}")
        return labels

    def _saliency_map(self, input_tensor, head_idx: int) -> np.ndarray:
        if torch is None or self.model is None:
            return np.zeros((224, 224), dtype=np.float32)

        x = input_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        if output.ndim == 1:
            output = output.unsqueeze(0)

        if output.shape[-1] == 1:
            target = output[0, 0]
        else:
            target = output[0, head_idx]

        target.backward()
        grad = x.grad[0].detach().abs().mean(dim=0).cpu().numpy()
        grad -= grad.min()
        denom = float(grad.max())
        if denom > 1e-8:
            grad /= denom
        return grad.astype(np.float32)

    def _overlay_to_data_uri(self, image_rgb: np.ndarray, heatmap: np.ndarray) -> str:
        h, w = image_rgb.shape[:2]
        if heatmap.shape != (h, w):
            heatmap = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

        heat = np.clip(heatmap, 0.0, 1.0)
        r = heat
        g = np.clip(1.0 - np.abs(heat - 0.5) * 2.0, 0.0, 1.0) * 0.8
        b = np.clip(1.0 - heat * 0.9, 0.0, 1.0)
        cmap = np.stack([r, g, b], axis=-1)

        base = image_rgb.astype(np.float32) / 255.0
        overlay = np.clip(base * 0.55 + cmap * 0.45, 0.0, 1.0)
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

        buffer = io.BytesIO()
        overlay_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        if self.model is None:
            return {
                "prediction": "brain_model_not_ready",
                "confidence": 0.0,
                "head_scores": [],
                "heatmaps": [],
                "backend": "stub",
                "input_sha12": hashlib.sha256(image_bytes).hexdigest()[:12],
                "note": "Place a valid brain_model.pth (full model) or provide model_def.py for state_dict checkpoints.",
            }

        if torch is None:
            raise RuntimeError("PyTorch is required for Brain inference.")

        input_tensor, vis_image = self._preprocess(image_bytes)

        with torch.no_grad():
            output = self.model(input_tensor)
            if output.ndim == 1:
                output = output.unsqueeze(0)

            if output.shape[-1] == 1:
                probs = torch.sigmoid(output)[0]
            else:
                probs = torch.sigmoid(output)[0]

        scores = probs.detach().cpu().numpy().tolist()
        labels = self._ensure_labels(len(scores))
        head_scores = [
            {
                "label": labels[i],
                "score": round(float(scores[i]), 4),
            }
            for i in range(len(scores))
        ]

        top_idx = int(np.argmax(scores)) if scores else 0
        prediction = labels[top_idx] if labels else "prediction"
        confidence = float(scores[top_idx]) if scores else 0.0

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_k = min(len(ranked), 12)
        heatmaps = []
        for i in ranked[:top_k]:
            saliency = self._saliency_map(input_tensor, i)
            heatmaps.append(
                {
                    "label": labels[i],
                    "score": round(float(scores[i]), 4),
                    "src": self._overlay_to_data_uri(vis_image, saliency),
                }
            )
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "head_scores": head_scores,
            "heatmaps": heatmaps,
            "backend": "torch",
            "input_sha12": hashlib.sha256(image_bytes).hexdigest()[:12],
        }
