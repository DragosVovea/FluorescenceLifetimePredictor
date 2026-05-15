Use this corrected and cleaner GitHub `README.md`. I fixed the model filenames, feature count, optional input handling, and added the GitHub/large-model-file note.

````markdown
# Fluorescence Lifetime Predictor

Machine-learning framework for predicting **fluorescence lifetime (ns)** of organic chromophores in specific solvent environments.

The model uses chromophore and solvent SMILES together with molecular descriptors, photophysical inputs, engineered features, and a trained LightGBM regression model.

---

## Overview

Fluorescence lifetime is an important photophysical property related to excited-state relaxation, radiative decay, non-radiative decay, and molecular environment. This repository provides an easy-to-use prediction script for estimating fluorescence lifetime from molecular structure and optional experimental photophysical information.

The prediction pipeline combines:

- RDKit molecular descriptors
- 3D molecular geometry descriptors
- Morgan and MACCS fingerprints
- Predicted pKa values
- Chromophore–solvent interaction features
- KMeans cluster assignment
- Trained LightGBM regression model

---

## Repository

```text
FluorescenceLifetimePredictor/
├── predict_lifetime.py
├── example_usage.py
├── README.md
├── requirements.txt
├── FinalModel.pkl        # trained model pipeline
└── FinalKMeans.pkl       # trained KMeans clustering model
````

If the `.pkl` files are too large for GitHub, download them from the release/Zenodo link and place them in the repository root folder.

---

## Required model files

The script requires two trained files:

```text
FinalModel.pkl
FinalKMeans.pkl
```

`FinalModel.pkl` contains the trained LightGBM model, selected feature list, KNN imputer, and preprocessing objects.

`FinalKMeans.pkl` contains the KMeans clustering model used to assign the chromophore cluster feature.

---

## Installation

Create a fresh environment and install the required packages:

```bash
pip install pandas numpy scikit-learn lightgbm joblib rdkit-pypi pkapredict
```

Alternative with conda for RDKit:

```bash
conda install -c conda-forge rdkit pandas numpy scikit-learn lightgbm joblib
pip install pkapredict
```

---

## Input format

The prediction script accepts a list of dictionaries or a pandas DataFrame.

Required fields:

| Field         | Description        |
| ------------- | ------------------ |
| `Chromophore` | Chromophore SMILES |
| `Solvent`     | Solvent SMILES     |

Optional fields:

| Field                 | Description                |
| --------------------- | -------------------------- |
| `Absorption max (nm)` | Absorption maximum in nm   |
| `Emission max (nm)`   | Emission maximum in nm     |
| `Quantum yield`       | Fluorescence quantum yield |

If optional values are missing, use `None` or omit them. Missing values are handled by the trained KNN imputer stored in `FinalModel.pkl`.

---

## Minimal example

```python
from predict_lifetime import predict_lifetime

data = [
    {
        "Chromophore": "O=C1OC2=CC=CC=C2C=C1",
        "Solvent": "CCO"
    }
]

result = predict_lifetime(data)
print(result)
```

---

## Example with optional photophysical inputs

```python
from predict_lifetime import predict_lifetime

data = [
    {
        "Chromophore": "O=C1OC2=CC=CC=C2C=C1",
        "Solvent": "CCO",
        "Absorption max (nm)": 320,
        "Emission max (nm)": 450,
        "Quantum yield": 0.35
    }
]

result = predict_lifetime(data)
print(result)
```

---

## Output

The output is a pandas DataFrame containing the input molecules and the predicted fluorescence lifetime:

```text
Predicted lifetime (ns)
```

Example output:

```text
Chromophore              Solvent    Predicted lifetime (ns)
O=C1OC2=CC=CC=C2C=C1    CCO        2.84
```

---

## Pipeline workflow

```text
Chromophore SMILES + Solvent SMILES
        ↓
Canonicalization
        ↓
RDKit descriptor calculation
        ↓
3D descriptor generation
        ↓
Morgan + MACCS fingerprints
        ↓
Predicted pKa
        ↓
KMeans cluster assignment
        ↓
Engineered physicochemical and photophysical features
        ↓
Feature selection using trained feature list
        ↓
KNN imputation + transformations + scaling
        ↓
LightGBM prediction
        ↓
Predicted fluorescence lifetime (ns)
```

---

## Features used by the model

The model uses a selected descriptor subset obtained after recursive feature elimination. The final trained feature space contains **639 selected features**, including:

* chromophore RDKit descriptors;
* solvent RDKit descriptors;
* 3D geometry descriptors;
* Morgan fingerprints;
* MACCS fingerprints;
* predicted pKa;
* absorption/emission-based descriptors;
* Stokes shift and energy descriptors;
* chromophore–solvent interaction descriptors;
* cluster label.

---

## Model details

| Item              | Description                                                   |
| ----------------- | ------------------------------------------------------------- |
| Algorithm         | LightGBM regressor                                            |
| Target            | Fluorescence lifetime, log-transformed as `log(1 + lifetime)` |
| Validation        | Cluster-stratified 10-fold cross-validation                   |
| Feature selection | Recursive feature elimination                                 |
| Missing values    | KNN imputation                                                |
| Scaling           | Power transformation, standard scaling, MinMax scaling        |
| Output transform  | `expm1()` to return lifetime in ns                            |

Final cross-validation performance:

```text
MAE = 0.8324 ± 0.0617 ns
R²  = 0.7523 ± 0.0317
```

---

## Important notes

* Invalid SMILES will raise an error.
* The model is intended for organic chromophores and solvent environments similar to the training data.
* Predictions are most reliable in the intermediate lifetime range where training data density is highest.
* Very short lifetimes and very long lifetimes may show larger relative errors.
* The model is intended as a fast screening and prioritization tool, not as a replacement for experimental spectroscopy or detailed TD-DFT calculations.

---

## Large model files

If `FinalModel.pkl` and `FinalKMeans.pkl` are in the Archive Model. Unzip them After downloading the model files, place them in the same folder as `predict_lifetime.py`.

---

## Citation

If you use this repository, please cite the associated manuscript:

```text
Vovea, D.-C.; Chiș, V. Tree-Based Machine Learning Model for Fluorescence Lifetime Prediction in Organic Compounds.
```

If a Zenodo DOI is created for the model files, cite the Zenodo record as well.

---

## Author

Dragoș-Cătălin Vovea
Vasile Chiș

Faculty of Physics, Babeș-Bolyai University, Cluj-Napoca, Romania

---

## License

MIT License

---

## Acknowledgments

This work was developed as part of fluorescence lifetime prediction research at Babeș-Bolyai University under the supervision of Prof. Vasile Chiș.

```
```
