---
title: Color Invariant Saree Design Recognition
emoji: 🧵
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: 1.36.0
app_file: app/streamlit_app.py
pinned: false
---

# Color-Invariant Saree Design Recognition

### Design retrieval for sarees, independent of colour

**Live Demo:** `COMING SOON`

**GitHub:** [patilanuja602](https://github.com/patilanuja602)
**LinkedIn:** [Anuja Patil](https://www.linkedin.com/in/anuja-patil-7649a124)

---

## Project Overview

This project identifies a saree based on its **surface design or pattern**, independent of its colour.

The key challenge is that the **same design can appear in different colours**, while different designs can have similar colours.

For example:

```text
Same design + different colour  →  Match
Different design + same colour  →  Do not match
```

Because of this, the problem is treated as **design retrieval and verification**, rather than simple saree classification.

The saree type shown to the user is obtained from the metadata of the retrieved reference designs.

---

## How It Works

```text
Saree Image
     ↓
Image Preprocessing
     ↓
DINOv2 Feature Extraction
     ↓
256-D Design Embedding
     ↓
Cosine Similarity Search
     ↓
Design-Level Matching
     ↓
Saree Type / Unknown
```

The model converts each saree image into a **256-dimensional embedding**.

Images with similar surface designs should produce similar embeddings. The query embedding is compared with embeddings stored in the reference gallery to retrieve the closest designs.

Multiple images belonging to the same design are aggregated so that the same design is not repeatedly displayed as separate matches.

---

## Model

The system uses **DINOv2-S/14** as the visual feature extractor with a trainable projection head.

### Training

* PyTorch
* Supervised Contrastive Learning
* Two-view training
* Colour augmentation using hue changes, channel permutation, saturation, gamma and grayscale
* Same-colour / different-design hard negatives
* pHash-based false-negative protection
* Frozen warm-up followed by fine-tuning of the last DINOv2 blocks
* AdamW optimizer

### Final Model

* Embedding size: **256**
* Parameters: **23.1M**
* Parameters tuned in final stage: **8.15M**
* Approximate computation: **12.25 GFLOPs/image**

---

## Results

The final model was evaluated using design-disjoint validation and test splits.

| Split      |  R@1 |  R@5 | R@10 |   mAP | ROC-AUC |   EER |
| ---------- | ---: | ---: | ---: | ----: | ------: | ----: |
| Validation | 0.90 | 1.00 | 1.00 | 0.917 |   0.937 | 0.108 |
| Test       | 0.50 | 1.00 | 1.00 | 0.764 |   0.924 | 0.171 |

The test set contains only **8 queries**, so R@1 has high uncertainty. The complete evaluation results and additional metrics are available in the project files.

The verification threshold of **0.7034** was selected using the validation set and was not tuned on the test set.

---

## Training Code

The complete model training and evaluation code is available in the Google Colab notebook:

**[Open Training Notebook](saree_design_retrieval_colab.ipynb)**

The notebook contains:

* Dataset preparation
* Image preprocessing
* Model architecture
* Colour augmentation
* Contrastive training
* Validation
* Retrieval evaluation
* Verification evaluation
* Result generation

---

## Dataset Audit

Before training, the available datasets were inspected for duplicates, label quality, image quality, colour bias and possible data leakage.

The complete audit is available here:

**[View Dataset Audit Report](Dataset%20Audit%20Report.pdf)**

The project uses:

* **DeepLure Saree Corpus**
* **Indian Saree Patterns dataset from Kaggle**

The DeepLure corpus is proprietary and is **not redistributed in this repository**.

The Kaggle dataset provides saree category information. Category labels are kept separate from design identity because the core task is design retrieval.

---

## Live Application

The deployed application will allow a user to:

1. Upload a saree image
2. Generate its design embedding
3. Search the reference gallery
4. View the closest design matches
5. See the associated saree type
6. Verify whether two images represent the same design

### Live Demo

**[Open the Live Application](YOUR_DEPLOYED_APP_LINK)**

*The deployment link will be added once the application is live.*

---

## Repository Structure

```text
Color-Invariant-Saree-Design-Recognition/
│
├── saree_design_retrieval_colab.ipynb
├── Dataset Audit Report.pdf
├── Patil_Anuja.pdf
├── README.md
│
└── deployment/
    └── ...
```

The deployment files and supporting inference code will be added as the application is finalized.

---

## Limitations

The current system has some limitations:

* The available DeepLure data does not contain reliable design IDs
* Some design groups use automatically proposed labels
* The evaluation dataset is relatively small
* Verification precision is lower than recall at the selected threshold
* Saree categories depend on the metadata available in the reference gallery
* Motif names are not available in the current metadata

A manually verified design-level metadata file would be an important improvement.

---

## Future Improvements

* Larger verified design gallery
* More real same-design / different-colour pairs
* Improved hard-negative mining
* Larger evaluation sets
* Additional saree types such as Kanjeevaram and Paithani
* Extension to other garment categories

New designs can be added to the gallery without retraining the complete model.

---

## About Me

### Anuja Patil

Computer Science Engineering | AI / Machine Learning

I work on practical AI and computer vision systems, with experience in machine learning, computer vision, model development and deployment.

**GitHub:** [patilanuja602](https://github.com/patilanuja602)
**LinkedIn:** [anuja-patil-7649a124](https://www.linkedin.com/in/anuja-patil-7649a124)
**Resume:** [View Resume](Patil_Anuja.pdf)

