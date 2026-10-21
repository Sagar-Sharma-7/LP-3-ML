# PE Screening Prototype (CPU-only)

A compact final-year ML project that demonstrates a pulmonary embolism (PE) **screening workflow** from individual CT slices. It includes DICOM/PNG loading, CT-image preprocessing, a CPU-friendly classifier, sensitivity-map visual aid, evaluation metrics, and downloadable reports.

> **Safety statement:** This ML project is not a medical device. It must not be used to diagnose, exclude, or treat pulmonary embolism. Results require qualified clinical review.

## Why this scope

The project is designed for the available laptop: i7-1360P CPU, integrated graphics and a strict 5 GB project cap. It uses logistic regression with compact image features rather than a GPU-dependent 3D neural network. This makes training repeatable in minutes on a selected subset.

## Features

- DICOM (`.dcm`) and PNG/JPEG image upload
- DICOM metadata and basic quality/error handling
- Normalisation, resizing, and CPU-friendly feature extraction
- Slice-level probability scores and selected-image study summary
- Occlusion-sensitivity visual aid (explicitly not lesion localisation)
- Evaluation metrics: accuracy, precision, recall, F1, and ROC-AUC
- CSV and PDF report downloads
- Model persistence and an on-device Streamlit user interface
- A synthetic demo-data generator so the whole project runs immediately
- A real-data workflow using a storage-capped subset of FUMPE CT scans and expert masks

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_demo_dataset.py
python train_model.py
streamlit run app.py
```

Open the local address that Streamlit prints, normally `http://localhost:8501`.

## Demo dataset

`generate_demo_dataset.py` creates 240 simulated CT-like images (120 per class) in `data/prepared/`. These images exist only to test the pipeline and UI. They are intentionally marked as synthetic in the manifest. Do not describe their metrics as medical results.

## Real-data workflow: FUMPE

This project is configured to work with **FUMPE** (Ferdowsi University of Mashhad's PE dataset): 35 real CT pulmonary-angiography studies with radiologist-created voxel masks. The public repository is released under [CC BY 4.0](https://huggingface.co/datasets/MedOtter/FUMPE), so cite the dataset paper and repository in your report. The original data paper describes 35 studies, 8,792 slices, and expert ground-truth PE delineations. [Dataset paper](https://www.nature.com/articles/sdata2018180)

The full image dataset is roughly 4 GB. To respect the 5 GB cap, the project downloads only the first **12 real studies** by default (around 1.3 GB) plus their masks. It then creates a compact, balanced working dataset of approximately 900 labelled PNG slices. The slice labels are created from the expert masks: a non-empty mask means `PE present on image`.

```powershell
# Download real CTA volumes and their expert segmentation masks.
.\.venv\Scripts\python.exe download_fumpe_subset.py --patients 12

# Convert real NIfTI CTA volumes into compact labelled image slices.
.\.venv\Scripts\python.exe prepare_fumpe_subset.py --fumpe-root data\raw\fumpe --output data\prepared

# Train using patient-level separation: images from one patient never occur in both training and test sets.
.\.venv\Scripts\python.exe train_model.py --manifest data\prepared\manifest.csv

# Restart the app to load the real-data model.
.\.venv\Scripts\streamlit.exe run app.py
```

The real image volumes remain under `data/raw/fumpe/` so you can reproduce the preparation step. The generated `data/prepared/` images are small. Do not redistribute the volumes unless their licence permits it. The older `prepare_rsna_subset.py` remains available only if you independently obtain the much larger RSNA dataset.

## Folder layout

```text
app.py                       Streamlit application
pe_screening.py              Image, training, prediction, report utilities
generate_demo_dataset.py     Synthetic pipeline-demo data generator
download_fumpe_subset.py     Downloads a small real FUMPE subset (~1.3 GB)
prepare_fumpe_subset.py      Converts FUMPE expert masks into real slice labels
prepare_rsna_subset.py       Licensed RSNA DICOM subset converter
train_model.py               Local training command
data/prepared/manifest.csv   Generated local dataset manifest
models/                      Generated model (ignored by Git)
reports/                     Optional local reports (ignored by Git)
```

## Project limitations for viva

- A single CT slice does not contain the context of a whole CTPA study.
- The demonstrated FUMPE subset is small and should be evaluated only as research data.
- Small subset performance may not generalise to other sites/scanners/populations.
- Sensitivity maps are a visual troubleshooting aid, not a clinical explanation.
- The model is a screening proof-of-concept and cannot replace a radiologist.
