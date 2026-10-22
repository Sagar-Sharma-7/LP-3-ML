# Pulmonary Embolism Screening from CT Images

## Complete project documentation and report-writing guide

> **Project type:** Machine Learning Project
> **Application type:** Local Streamlit web application
> **Primary task:** Classify individual CT slices as PE-present or PE-absent
> **Primary dataset:** FUMPE CT Pulmonary Angiography dataset
> **Execution target:** CPU-only Windows laptop

> **Safety statement**
> This ML project is a learning and demonstration system, not a medical device. It must not be used to diagnose, exclude, or guide treatment for pulmonary embolism. The model produces an image score only; qualified clinicians must interpret medical images.

---

## Contents

1. [Abstract](#1-abstract)
2. [Introduction and motivation](#2-introduction-and-motivation)
3. [Problem statement](#3-problem-statement)
4. [Objectives and scope](#4-objectives-and-scope)
5. [Background concepts](#5-background-concepts)
6. [Hardware, software, and requirements](#6-hardware-software-and-requirements)
7. [Dataset](#7-dataset)
8. [System architecture](#8-system-architecture)
9. [Methodology](#9-methodology)
10. [Model design](#10-model-design)
11. [Training and validation](#11-training-and-validation)
12. [Results and discussion](#12-results-and-discussion)
13. [Application workflow](#13-application-workflow)
14. [Installation and commands](#14-installation-and-commands)
15. [Project structure](#15-project-structure)
16. [Testing](#16-testing)
17. [Storage and performance](#17-storage-and-performance)
18. [Limitations, ethics, and future work](#18-limitations-ethics-and-future-work)
19. [20-page report guide](#19-20-page-report-guide)
20. [Viva preparation](#20-viva-preparation)
21. [Troubleshooting](#21-troubleshooting)
22. [References](#22-references)

---

## 1. Abstract

Pulmonary embolism (PE) is a potentially serious condition in which an obstruction, commonly a blood clot, affects one or more arteries in the lungs. Computed Tomography Pulmonary Angiography (CTPA) is a commonly used imaging method because it provides detailed cross-sectional views of pulmonary vessels. A CTPA study may contain hundreds of slices, making image review time-consuming and dependent on clinical expertise.

This project develops a lightweight machine-learning workflow for **educational PE screening from individual CT slices**. It loads prepared CT images or individual DICOM files, performs normalisation and feature extraction, predicts a PE likelihood score using logistic regression, displays a sensitivity map, and creates CSV/PDF reports. The application is implemented in Python and delivered through a local Streamlit interface.

The real-data workflow uses a small subset of FUMPE, a public CTPA dataset containing expert-created PE segmentation masks. A slice with one or more positive mask pixels is assigned label `1` (PE present); an empty mask produces label `0` (PE absent). To avoid overly optimistic results, the project uses patient-level cross-validation: all slices from one patient remain together in either training or evaluation data.

The current experiment uses 12 CTPA studies and 774 prepared slices: 258 PE-positive and 516 PE-negative. Five-fold patient-level cross-validation gives 66.9% accuracy, 50.4% precision, 55.4% recall, 52.8% F1-score, and 68.6% ROC-AUC. These are modest research-baseline results, not clinical performance. The project is intentionally designed to run on a CPU-only laptop within a 5 GB storage budget.

**Keywords:** pulmonary embolism, CTPA, computed tomography, machine learning, logistic regression, medical imaging, Streamlit, FUMPE.

---

## 2. Introduction and motivation

Machine learning can find repeatable patterns in images, organise large image collections, and support human review. Medical imaging is a major use case because a single CT study contains more information than simple clinical measurements can represent. However, medical ML has high stakes: an inaccurate result can be misunderstood as a diagnosis. A good student project must therefore combine technical implementation with honest validation and careful wording.

PE screening is a strong ML project topic because it requires:

- handling medical-image formats;
- converting raw image data into ML features;
- supervised binary classification;
- selecting suitable metrics for imbalanced classes;
- preventing train/test leakage;
- deploying a model as a usable local application;
- documenting safety, privacy, and limitations.

The complete RSNA PE dataset is valuable but is much too large for this project's 5 GB allocation and CPU-only laptop. FUMPE provides a practical real-data alternative because it includes CTPA volumes and expert masks. The masks allow labels to be created reproducibly without manually annotating medical images.

The project chooses a classical ML baseline rather than a large 3D deep-learning network. This is a deliberate engineering decision: the available system has an Intel Core i7-1360P CPU, 15.6 GB RAM, integrated Intel Iris Xe graphics, and no CUDA-capable GPU. A compact feature-based model trains in minutes and can be explained clearly in a final-year viva.

---

## 3. Problem statement

Reviewing CTPA studies for potential emboli is difficult because a finding can be small and a study can contain many slices. Reproducing a hospital-grade diagnostic system would require large multi-centre datasets, clinical validation, specialist review, and substantial computing resources. This project addresses the following narrower problem:

> Can a local and CPU-efficient ML application classify prepared CTPA slices as PE-present or PE-absent using real expert-labelled data, while reporting its limitations transparently?

The application does not diagnose a patient. It predicts a model score associated with patterns in one image slice. The system should be viewed as an educational image-review prototype.

### Inputs

- FUMPE CT angiography volumes (`.nii.gz`) and matching expert masks during preparation.
- Generated 224 × 224 grayscale PNG slices during training.
- Optional individual DICOM (`.dcm`), PNG, and JPEG files in the local application.

### Outputs

- Slice-level PE likelihood score.
- Lower, intermediate, or higher model-score category.
- Occlusion-sensitivity visual aid.
- Aggregate score for multiple uploaded slices.
- CSV and PDF reports.
- Patient-level validation metrics.

---

## 4. Objectives and scope

### 4.1 Primary objective

Create a fully working local ML application that demonstrates PE screening from CTPA slices using real expert-labelled data on CPU-only hardware.

### 4.2 Specific objectives

1. Download and prepare a storage-capped FUMPE subset.
2. Convert CT volumes into standard 224 × 224 image slices.
3. Derive labels automatically from expert segmentation masks.
4. Train a CPU-friendly binary classifier.
5. Evaluate with patient-level rather than random-slice splitting.
6. Display accuracy, precision, recall, F1-score, and ROC-AUC.
7. Provide an interface for image review and report export.
8. Clearly document non-clinical limitations.

### 4.3 In scope

- Slice-level binary classification.
- NIfTI volume and mask preparation.
- DICOM/PNG/JPEG loading in the application.
- Classical ML, feature extraction, model persistence, and reporting.
- Local-only execution and patient-level cross-validation.

### 4.4 Out of scope

- Clinical diagnosis, triage, or treatment recommendation.
- PACS/hospital integration.
- Full-study 3D interpretation.
- GPU-based 3D CNNs or transformers.
- Collection of patient identifiers.
- Regulatory approval.

### 4.5 Success criteria

The project is considered successful when it can prepare real data, train locally, save/load a model, show valid grouped metrics, score an uploaded image, export a report, and stay within the hardware/storage budget.

---

## 5. Background concepts

### 5.1 Pulmonary embolism and CTPA

PE refers to an obstruction in a pulmonary artery. In clinical practice, CTPA uses iodinated contrast to visualise pulmonary vessels. Full interpretation requires the complete study, patient history, scan quality, and clinical expertise. In this project, “PE present” is only a label derived from an expert mask for one image slice.

### 5.2 CT intensity and windowing

CT values are commonly expressed in Hounsfield units (HU). Air is around -1000 HU, water is near 0 HU, and dense materials have higher values. Because raw CT values span a wide range, a **window** selects a useful range for display and learning.

The project uses a fixed CTPA window:

\[
\text{lower}=-250\text{ HU}, \qquad \text{upper}=450\text{ HU}
\]

This is equivalent to centre 100 HU and width 700 HU. The normalised image is:

\[
I_{normalised}=\operatorname{clip}\left(\frac{I_{HU}+250}{700},0,1\right)
\]

A fixed window is important because normalising each slice independently can hide intensity differences between slices.

### 5.3 DICOM and NIfTI

**DICOM** is the standard medical-image format and may include both pixels and metadata. **NIfTI** is a common research format for a three-dimensional image volume. FUMPE uses NIfTI volumes and masks in this workflow; the Streamlit app can also read individual DICOM slices.

### 5.4 Supervised learning

In supervised learning, examples \((x_i,y_i)\) are used to learn a relationship between features \(x_i\) and labels \(y_i\). Here:

\[
y_i = \begin{cases}
1 & \text{if an expert mask has one or more positive pixels on the slice}\\
0 & \text{if the expert mask is empty on the slice}
\end{cases}
\]

### 5.5 Logistic regression

Logistic regression estimates the probability-like value:

\[
P(y=1|x)=\sigma(w^T x+b)
\]

where \(w\) is the learned weight vector, \(b\) is an intercept, and:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

It is a good baseline here because it is fast, transparent, works with numeric feature vectors, and is practical without a GPU.

---

## 6. Hardware, software, and requirements

### 6.1 Hardware

| Component | Available specification | Design implication |
|---|---:|---|
| CPU | Intel Core i7-1360P, 12 cores / 16 logical processors | Adequate for image features and classical ML |
| RAM | 15.6 GB | Allows one NIfTI volume to be processed at a time |
| GPU | Intel Iris Xe integrated graphics | No CUDA; deep 3D training is excluded |
| Project storage cap | 5 GB | Requires a compact real-data subset |
| OS | Windows / PowerShell | Commands use PowerShell syntax |

### 6.2 Software packages

| Package | Purpose |
|---|---|
| Python 3.12+ | Runtime environment |
| Streamlit | Local browser user interface |
| NumPy | Numerical arrays and image values |
| Pandas | Manifests and CSV result tables |
| scikit-learn | Scaling, classifier, metrics, validation |
| Pillow | Image resizing and PNG/JPEG operations |
| pydicom | DICOM image reading |
| NiBabel | NIfTI volume/mask reading |
| Matplotlib | Sensitivity-map display |
| ReportLab | PDF generation |
| Hugging Face Hub | FUMPE subset transfer |

### 6.3 Functional requirements

| ID | Requirement |
|---|---|
| FR-01 | Prepare a local real-data subset from CT volumes and masks. |
| FR-02 | Create binary slice labels from expert masks. |
| FR-03 | Train and save a CPU-compatible model. |
| FR-04 | Accept DICOM, PNG, and JPEG images in the UI. |
| FR-05 | Show score, category, metadata, and sensitivity map. |
| FR-06 | Export CSV and PDF results. |
| FR-07 | Report patient-level validation metrics. |

### 6.4 Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-01 | Run without an NVIDIA GPU. |
| NFR-02 | Remain below the 5 GB project budget. |
| NFR-03 | Be reproducible with fixed random seeds. |
| NFR-04 | Run locally without a cloud service. |
| NFR-05 | Communicate limitations clearly. |

---

## 7. Dataset

### 7.1 FUMPE dataset

FUMPE is the Ferdowsi University of Mashhad Pulmonary Embolism dataset. It provides CTPA volumes and expert-reviewed PE masks. The original dataset paper reports 35 subjects, 8,792 slices, and detailed PE-region annotations. The mask data make it possible to create repeatable labels from real medical images.

### 7.2 Why FUMPE was selected

| Criterion | FUMPE advantage |
|---|---|
| Real images | Uses CTPA studies, not generated examples |
| Reliable labels | Voxel-level expert masks are available |
| Practical scale | A small subset fits a student laptop |
| Reproducibility | Labels are created by code from masks |
| Educational value | Supports preprocessing, classification, and explainability |

The optional `prepare_rsna_subset.py` file remains in the repository for independently obtained RSNA data, but RSNA is not recommended for the 5 GB project limit.

### 7.3 Current real-data experiment

The default script downloads `PAT001` through `PAT012` and their matching masks. After preparation, the current experiment contains:

| Item | Value |
|---|---:|
| Studies | 12 |
| Prepared slices | 774 |
| PE-positive slices | 258 |
| PE-negative slices | 516 |
| Sampling ratio | approximately 1 positive : 2 negative |
| Output size | 224 × 224 grayscale PNG |
| Validation | 5-fold patient-level cross-validation |

The count differs by patient because some studies contain fewer positive slices than the configured cap.

### 7.4 Licence and attribution

The Hugging Face FUMPE repository states CC BY 4.0 at the time this project was prepared. Always review current terms before use, cite both the original data paper and repository, and do not redistribute medical-image files unless the licence permits it.

### 7.5 Demo data versus real data

`generate_demo_dataset.py` creates artificial CT-like images only to verify the pipeline on a new installation. Its results must **never** be presented as medical performance. The final report should use the FUMPE-derived files, which begin with `fumpe_` and have source `FUMPE_expert_mask` in `manifest.csv`.

---

## 8. System architecture

```text
     FUMPE CTA volumes + expert masks (.nii.gz)
                         │
                         ▼
            prepare_fumpe_subset.py
       fixed window → resize → mask labels
                         │
                         ▼
       prepared PNG slices + manifest.csv
                         │
                         ▼
              train_model.py
    features → grouped validation → final fit
                         │
                         ▼
        models/pe_screening_model.joblib
                         │
                         ▼
                Streamlit application
    upload → score → sensitivity map → CSV/PDF
```

### 8.1 Important source files

| File | Responsibility |
|---|---|
| `app.py` | Streamlit pages, upload workflow, visualisation, downloads |
| `pe_screening.py` | Shared image, ML, report, and sensitivity-map functions |
| `download_fumpe_subset.py` | Downloads selected real FUMPE studies/masks |
| `prepare_fumpe_subset.py` | Converts NIfTI volumes/masks into labelled PNG slices |
| `train_model.py` | Trains and saves the final model |
| `benchmark_models.py` | Compares CPU-friendly candidate models fairly |
| `tune_rbf_threshold.py` | SVM threshold diagnostic tool |
| `generate_demo_dataset.py` | Synthetic setup test only |

---

## 9. Methodology

### 9.1 Download data

The FUMPE download utility retrieves a storage-capped set of 12 image volumes and their masks. It uses Hugging Face Hub rather than a raw direct download because the Hub client is more reliable for large Xet-backed files.

### 9.2 Label slices from masks

For every axial index \(z\), the mask is checked:

\[
\text{label}(z)=\begin{cases}
1 & \text{if any pixel in mask}[:,:,z]>0\\
0 & \text{otherwise}
\end{cases}
\]

The script retains a maximum of 25 positive slices per patient and up to two randomly selected negative slices per positive slice. This reduces severe class imbalance while keeping meaningful negative examples.

### 9.3 Apply fixed CTPA window and resize

Each slice is clipped to -250 to 450 HU, scaled to 0–1, resized to 224 × 224, and saved as a grayscale PNG. Each record in `manifest.csv` has `image_path`, `label`, `study_id`, and `source` columns.

### 9.4 Extract features

The system uses a 1,071-value feature vector for each image:

| Feature family | Count | Description |
|---|---:|---|
| Intensity/gradient statistics | 7 | Mean, standard deviation, percentiles, edge summary |
| Intensity histogram | 24 | Distribution of grayscale values |
| Gradient histogram | 16 | Distribution of edge magnitudes |
| Downsampled 32 × 32 image | 1,024 | Compact spatial representation |
| **Total** | **1,071** | Input values per slice |

`StandardScaler` standardises each feature before training so that features with larger raw numeric ranges do not dominate the classifier.

### 9.5 Train the classifier

The deployed classifier is logistic regression using the `liblinear` solver, 1,500 maximum iterations, random seed 42, and `class_weight="balanced"`. Balanced weighting gives additional importance to the less common PE-positive class.

### 9.6 Sensitivity map

The app divides an image into an 8 × 8 grid. It hides one grid region at a time, recomputes the score, and records the score reduction. Warm regions changed the score more. This is an occlusion-sensitivity aid; it is not a segmentation mask and does not identify a confirmed clot boundary.

### 9.7 Multi-slice summary

For several uploaded images, the app calculates:

\[
\text{Study score}=0.70\times\max(p_i)+0.30\times\operatorname{mean}(\text{top 3 }p_i)
\]

This is a display heuristic for prioritisation, not a clinically validated score.

---

## 10. Model design

### Why a classical model?

The model must run repeatably on CPU hardware. Logistic regression is transparent, trains quickly, and is easy to explain. A large 3D model would require considerably more patient data, GPU memory, training time, and validation effort.

### Candidate models investigated

`benchmark_models.py` compares logistic regression, RBF SVM, extra trees, and random forest with the same patient-level split rule. The deployed logistic model gives the most useful balance of recall, F1-score, speed, and explanation simplicity for this small subset. Some candidate models produced higher ROC-AUC but extremely low recall at their default threshold, which is not useful for a screening-oriented demonstration.

### Class imbalance

There are twice as many negative slices as positive slices. A model that predicts every slice as negative could appear to achieve around 66.7% accuracy. This is why accuracy must never be discussed alone; precision, recall, F1-score, and ROC-AUC are all reported.

---

## 11. Training and validation

### 11.1 Why random slice splitting is wrong

Neighbouring CT slices from the same patient are similar. If nearby slices are placed into both train and test sets, the model effectively sees near-duplicates during training. The resulting score can be artificially high.

### 11.2 Patient-level cross-validation

The project uses `StratifiedGroupKFold` with five folds and `study_id` as the group. For every fold, entire patients are held out. Each slice receives an out-of-fold prediction from a model that never trained on that slice's patient.

### 11.3 Final fitting

After validation, the pipeline is fitted one final time on all prepared slices and saved in:

```text
models/pe_screening_model.joblib
```

The artifact contains the fitted scaler/model, metrics, feature count, and image size.

### 11.4 Reproducibility

- Random seed: 42
- Fixed output size: 224 × 224
- Fixed CTPA window: -250 to 450 HU
- Fixed model configuration
- Version-controlled source code

---

## 12. Results and discussion

### 12.1 Current real-data results

| Metric | Result | Meaning |
|---|---:|---|
| Accuracy | 66.9% | All correctly classified slices divided by all slices |
| Precision | 50.4% | Correct among slices predicted PE-positive |
| Recall | 55.4% | PE-positive slices detected by the model |
| F1-score | 52.8% | Balance of precision and recall |
| ROC-AUC | 68.6% | Ranking performance across thresholds |

Metric formulas, using TP, TN, FP, and FN:

\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

\[
\text{Precision}=\frac{TP}{TP+FP}, \qquad
\text{Recall}=\frac{TP}{TP+FN}
\]

\[
F1=2\times\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}
\]

### 12.2 Interpretation

ROC-AUC above 0.5 indicates that the model learns some useful ranking signal, but the performance is moderate. The correct final-report conclusion is:

> The model demonstrates a working end-to-end real-data ML pipeline and learns discriminative information from the selected FUMPE subset. Its small dataset size, two-dimensional input, and moderate patient-level metrics mean that it is not suitable for clinical use.

Never claim that the model “diagnoses PE” or that this accuracy is clinically sufficient.

### 12.3 Suggested report figures

1. System architecture diagram from this README.
2. FUMPE preparation terminal output.
3. Example prepared CT slice.
4. Streamlit Screening workspace screenshot.
5. Example score plus sensitivity map.
6. Performance review screenshot.
7. Metric table and, optionally, a bar chart.

---

## 13. Application workflow

### 13.1 Interface pages

| Page | Use |
|---|---|
| Screening workspace | Upload and score CT images |
| Performance review | Show cross-validation results and context |
| Method and data | Explain the data, preparation, and limitations |

### 13.2 Review an image

1. Start the app at `http://localhost:8501`.
2. Select **Screening workspace**.
3. Upload one or more DICOM, PNG, or JPEG images.
4. Review original image, score, model category, metadata, and sensitivity map.
5. Download CSV/PDF if needed.

### 13.3 Score categories

| Score range | Category | Correct interpretation |
|---|---|---|
| Below 0.40 | Lower model score | Does not rule out PE |
| 0.40 to below 0.65 | Intermediate model score | Model is uncertain |
| 0.65 and above | Higher model score | Prioritise qualified review; not a diagnosis |

### 13.4 Reports

The CSV contains each filename, score, and category. The PDF includes input name, number of analysed images, maximum/mean score, top-scoring image names, model context, and a non-clinical-use warning.

---

## 14. Installation and commands

### 14.1 Setup

```powershell
cd "D:\DATA\aitpune\college study thingy\ml\mini_project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 14.2 Demo workflow only

Use only to test the code/UI before downloading real data. Do not use demo metrics in the final report.

```powershell
python generate_demo_dataset.py
python train_model.py
streamlit run app.py
```

### 14.3 Real FUMPE workflow

```powershell
# 1. Download 12 real CTA studies and expert masks.
.\.venv\Scripts\python.exe download_fumpe_subset.py --patients 12

# 2. Convert volumes/masks into labelled 224 × 224 CT slices.
.\.venv\Scripts\python.exe prepare_fumpe_subset.py --fumpe-root data\raw\fumpe --output data\prepared

# 3. Train, validate, and save the model.
.\.venv\Scripts\python.exe train_model.py --manifest data\prepared\manifest.csv

# 4. Run the local application.
.\.venv\Scripts\streamlit.exe run app.py
```

### 14.4 Verify real data

```powershell
Get-Content data\prepared\manifest.csv -TotalCount 5
```

Real records use filenames beginning `fumpe_` and source `FUMPE_expert_mask`. If records are named `demo_`, rerun FUMPE preparation and training.

---

## 15. Project structure

```text
mini_project/
├── app.py                         # Streamlit interface
├── pe_screening.py                # Shared ML/image/report utilities
├── train_model.py                 # Local model training
├── generate_demo_dataset.py       # Artificial test data only
├── download_fumpe_subset.py       # Real subset downloader
├── prepare_fumpe_subset.py        # FUMPE volume/mask preparation
├── prepare_rsna_subset.py         # Optional RSNA converter
├── benchmark_models.py            # CPU model comparison
├── tune_rbf_threshold.py          # Threshold diagnostic
├── requirements.txt               # Dependencies
├── .streamlit/config.toml         # Dark local theme
├── data/raw/fumpe/                # Downloaded data; Git ignored
├── data/prepared/                 # Generated images and manifest
├── models/                        # Saved model; Git ignored
└── reports/                       # Exported reports; Git ignored
```

Raw medical data, generated images, models, reports, and the virtual environment are excluded from Git to keep the repository small and avoid committing data artifacts.

---

## 16. Testing

| Test ID | Action | Expected result |
|---|---|---|
| T-01 | Generate demo data | PNG files and manifest are created |
| T-02 | Download FUMPE | NIfTI image/mask pairs appear under `data/raw/fumpe` |
| T-03 | Prepare FUMPE | Real `fumpe_` images and manifest are created |
| T-04 | Train model | Model file is saved and metrics print |
| T-05 | Start Streamlit | Local UI opens without error |
| T-06 | Upload PNG | Image, score, and metadata display |
| T-07 | Upload multiple files | Per-slice results and study summary display |
| T-08 | Download CSV | Result table downloads |
| T-09 | Download PDF | Human-readable report downloads |
| T-10 | Open Performance review | Patient-level metrics appear |

The key scientific test is patient-level validation. The terminal output should include `5-fold patient-level cross-validation` and `held_out_studies: 12` for the default real-data experiment.

---

## 17. Storage and performance

### 17.1 Storage budget

| Content | Expected size |
|---|---:|
| Python environment/dependencies | approximately 0.6 GB |
| 12 raw FUMPE studies + masks | approximately 1.3 GB |
| Prepared slices/manifest | below 0.2 GB |
| Model and reports | below 0.5 GB |
| Safety reserve | more than 2 GB |
| **Target total** | **under 5 GB** |

### 17.2 Runtime expectations

| Task | Expected behaviour |
|---|---|
| Demo generation | seconds |
| FUMPE preparation | minutes, depending on disk speed |
| Training | minutes on CPU |
| Starting the UI | seconds |
| Single score | generally seconds |
| Many images | slower because sensitivity maps repeat predictions |

The project intentionally avoids a 3D deep network because that would require much larger datasets, more storage, and GPU resources.

---

## 18. Limitations, ethics, and future work

### 18.1 Limitations

1. The model predicts individual slices, not a complete CTPA study.
2. Only 12 studies are used in the current experiment.
3. Labels are derived from masks and do not replace a full clinical outcome.
4. Handcrafted 2D features cannot represent complete 3D vessel context.
5. The study-summary score is a heuristic.
6. Moderate metrics are not sufficient for clinical deployment.
7. Generalisation can vary across hospitals, scanners, and protocols.

### 18.2 Ethics and privacy

The interface uses score-oriented language instead of diagnosis wording. A low score does not rule out PE and a high score does not confirm it. The application runs locally and does not intentionally collect user accounts, patient identifiers, or cloud telemetry. Dataset users must follow licence terms and must never attempt to re-identify people.

### 18.3 Risk controls

| Risk | Control |
|---|---|
| Train/test leakage | Patient-level grouped folds |
| Class imbalance | Balanced weights and controlled negative sampling |
| Misinterpretation | Visible non-clinical disclaimer |
| Storage overuse | 12-study download cap |
| Weak explainability | Metric page, reports, and sensitivity-map disclaimer |

### 18.4 Future work

1. Add more real studies and external validation.
2. Use consecutive slices or a 3D architecture.
3. Use transfer learning with appropriate hardware.
4. Add confusion matrix, ROC, and precision-recall charts in the UI.
5. Calibrate probability scores and tune thresholds on a separate validation set.
6. Perform expert usability review before making any decision-support claim.

---

## 19. 20-page report guide

This README is source material for a detailed college report. Use your institution's prescribed cover, certificate, declaration, and formatting rules. Rewrite technical explanations in your own words and add screenshots/tables from your actual run.

### 19.1 Suggested page allocation

| Section | Suggested pages |
|---|---:|
| Title page, certificate, declaration, acknowledgement | 3–4 |
| Abstract and keywords | 1 |
| Contents, list of figures, list of tables | 2–3 |
| Introduction and literature review | 3–4 |
| Problem statement, objectives, and scope | 2 |
| Dataset and methodology | 3–4 |
| Implementation and system design | 2–3 |
| Results and discussion | 2–3 |
| Conclusion, limitations, future work | 1–2 |
| References and appendix | 2+ |

### 19.2 Suggested chapter headings

**Chapter 1: Introduction** — Explain PE broadly, CTPA, medical-image ML, motivation, and objectives.

**Chapter 2: Literature review** — Discuss FUMPE, why expert masks are useful, classical ML versus deep learning, and explain the hardware-driven scope. Cite sources rather than copying them.

**Chapter 3: Methodology** — Include dataset details, mask-to-label conversion, CT windowing, feature extraction, logistic regression, and patient-level cross-validation.

**Chapter 4: System design and implementation** — Include architecture diagram, source-file roles, UI screenshots, and folder structure.

**Chapter 5: Results and discussion** — Present the metric table, explain why accuracy alone is misleading, discuss moderate performance honestly, and include example application screenshots.

**Chapter 6: Conclusion and future work** — State that the technical ML workflow was achieved, then describe limitations and improvement paths.

### 19.3 Figures to include

| Figure | Suggested caption |
|---|---|
| 1 | Architecture of the proposed PE screening ML system |
| 2 | FUMPE volume/mask to labelled PNG preparation workflow |
| 3 | Example CT slice after fixed CTPA windowing |
| 4 | Local Streamlit Screening workspace |
| 5 | Example score with occlusion-sensitivity aid |
| 6 | Performance review with patient-level metrics |
| 7 | Project folder structure |

### 19.4 Tables to include

| Table | Suggested content |
|---|---|
| 1 | Functional and non-functional requirements |
| 2 | Hardware/software specifications |
| 3 | FUMPE subset composition |
| 4 | Feature categories and counts |
| 5 | Model hyperparameters |
| 6 | Evaluation metric definitions |
| 7 | Final experimental results |
| 8 | Limitations and risk controls |

### 19.5 Conclusion template

Adapt this in your own words:

> This ML project implemented a CPU-efficient workflow for PE screening from CTPA image slices. Using expert masks from a real FUMPE subset, the system created slice labels, extracted image features, trained a logistic-regression classifier, and evaluated it through patient-level cross-validation. The local application supports image review, score visualisation, and report export. The moderate metrics and limited dataset prevent clinical use, but the work provides a reproducible foundation for future research using larger datasets and three-dimensional models.

---

## 20. Viva preparation

**What is the main aim?**
To demonstrate a local ML pipeline that scores CTPA slices using real expert-labelled data, without claiming diagnosis.

**Why FUMPE?**
It contains real CTPA volumes and expert masks, making reproducible labels possible at a practical size.

**Why not use the full RSNA dataset?**
It is too large for the 5 GB storage budget and CPU-only environment.

**How are labels created?**
Any positive pixel in the expert mask gives that CT slice label `1`; an empty mask gives `0`.

**Why fixed windowing?**
It keeps a consistent CTPA intensity range. Independent per-slice scaling can suppress useful relative intensity patterns.

**Why logistic regression?**
It is fast, explainable, CPU-friendly, and a strong baseline for engineered features.

**Why patient-level cross-validation?**
It prevents similar slices from the same patient appearing in both training and evaluation data.

**Why is accuracy insufficient?**
There are more negative slices. A model can look accurate by predicting mostly negative, so precision, recall, F1, and ROC-AUC are also required.

**What does 68.6% ROC-AUC mean?**
The model has moderate ability to rank positive slices above negative ones. It is better than random but not clinically sufficient.

**What does the sensitivity map mean?**
It shows coarse regions that changed the model score when hidden. It is not a true PE location or mask.

**What is your first future improvement?**
Use more studies and preserve 3D context across neighbouring CT slices.

---

## 21. Troubleshooting

### FUMPE download fails

Use the current Hugging Face based downloader. Re-running skips completed files:

```powershell
.\.venv\Scripts\python.exe download_fumpe_subset.py --patients 12
```

### `No FUMPE image volumes found`

Confirm that image files exist under `data/raw/fumpe/images/` and that the `--fumpe-root` path is correct.

### The app reports demo metrics

Inspect `data/prepared/manifest.csv`. If files begin with `demo_`, rerun real preparation and training.

### Streamlit does not show the latest interface/theme

Stop the terminal process with `Ctrl + C`, then start again:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### `ModuleNotFoundError`

Activate the environment and reinstall dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### A DICOM image does not load

Check that it is a readable image slice and not an unsupported compressed transfer syntax. The app displays a loading error rather than silently returning a score.

---

## 22. References

1. Tajbakhsh, N., et al. (2018). *A new dataset of computed-tomography angiography images for computer-aided detection of pulmonary embolism.* Scientific Data, 5, 180180. [https://www.nature.com/articles/sdata2018180](https://www.nature.com/articles/sdata2018180)
2. MedOtter. *FUMPE dataset repository.* Hugging Face. [https://huggingface.co/datasets/MedOtter/FUMPE](https://huggingface.co/datasets/MedOtter/FUMPE)
3. Colak, E., Kitamura, F. C., Hobbs, S. B., et al. (2021). *The RSNA Pulmonary Embolism CT Dataset.* Radiology: Artificial Intelligence, 3(2). [https://doi.org/10.1148/ryai.2021200254](https://doi.org/10.1148/ryai.2021200254)
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)
5. Streamlit. *Streamlit documentation.* [https://docs.streamlit.io/](https://docs.streamlit.io/)

---

## Appendix: command reference

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Download real subset
python download_fumpe_subset.py --patients 12

# Prepare real labelled slices
python prepare_fumpe_subset.py --fumpe-root data\raw\fumpe --output data\prepared

# Train and validate
python train_model.py --manifest data\prepared\manifest.csv

# Compare CPU models
python benchmark_models.py

# Start application
streamlit run app.py
```

### Responsible wording

Use: “The system predicts a model score for a CT slice”, “The project is a local ML demonstration”, and “The evaluation uses a small research subset.”

Avoid: “The system diagnoses PE”, “The app confirms a clot”, “The model is clinically accurate”, or “The heatmap identifies the clot location.”
