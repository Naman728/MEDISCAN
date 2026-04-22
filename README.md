# MEDISCAN

Flask app for medical imaging workflows (brain MRI, lung CT, chest X-ray, and related templates).

## Model weights (Google Drive)

Trained model files (`.pth`, `.h5`, etc.) are **not** stored in this repository because they exceed GitHub’s recommended file size limits (large files are rejected or impractical to version here).

Download the model archives from this folder, unzip if needed, and place the artifacts next to the loaders as described in each zip or below:

**[Model files on Google Drive](https://drive.google.com/drive/folders/1gIJoNB1NW5jvDDmwOwts_KL978Xxr3nv?usp=drive_link)**

Typical layout after extraction:

| Model        | Paths under `models/` |
| ------------ | ---------------------- |
| Brain        | `brain/brain_model.pth` (and `labels.json` / `model_def.py` if provided) |
| Lung CT      | `lung_ct/lung_ct_model.pth` and/or `lung_ct/lung_ct_model.h5` |
| Chest X-ray  | `lung_xray/lung_xray_model.h5` |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Ensure the model files from Drive are in place before expecting real predictions; without them, the app may use stub or limited behavior depending on the route.
