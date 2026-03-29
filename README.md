# Fluorescence Lifetime Prediction

## Overview

This project provides a machine learning pipeline for predicting **fluorescence lifetime (ns)** of molecular chromophores in specific solvents.

The model combines:

* RDKit molecular descriptors
* 3D geometric descriptors
* Engineered physicochemical features
* Predicted pKa values
* Structural clustering (KMeans on Morgan fingerprints)

---

## Features

### 1. Molecular Descriptors

* All RDKit descriptors (chromophore + solvent)
* E-State fingerprint summary

### 2. 3D Geometry

* Bond length statistics (mean, min, max)
* Bond angles
* Dihedral angles
* Planarity (RMSD)
* Radius of gyration

### 3. Engineered Features

Includes:

* Polarity, flexibility, rigidity
* Hydrogen bonding descriptors
* Conjugation and aromaticity metrics
* Chromophore–solvent interaction features
* Photophysical transformations (Stokes shift, energy)

### 4. pKa Prediction

* Uses `pkapredict` model
* Adds `Predicted_pKa` as a feature

### 5. Clustering

* Morgan fingerprints (radius=2, 2048 bits)
* KMeans clustering
* Adds `Cluster` label as feature

---

## Model

* Algorithm: LightGBM (`LGBMRegressor`)
* Target: `Lifetime (ns)` (log-transformed)
* Feature selection: RFE (88 features)
* Preprocessing:

  * KNN Imputation
  * Power Transform
  * Standard Scaling
  * MinMax Scaling

---

## Installation

```bash
pip install pandas numpy scikit-learn lightgbm rdkit-pypi joblib pkapredict
```

---

## Required Files

* `fluoresencePredictor.pkl` → trained model + preprocessing
* `kmeans_model.pkl` → clustering model

---

## Input Format

Provide a pandas DataFrame with:

```python
data = [
    {
        "Chromophore": "SMILES_STRING",
        "Solvent": "SMILES_STRING",
        "Absorption max (nm)": float,
        "Emission max (nm)": float,
        "Quantum yield": float
    }
]
```

## Output

* Predicted fluorescence lifetime in **nanoseconds (ns)**

---

## Pipeline Flow

```
SMILES → RDKit descriptors → 3D descriptors → pKa prediction
→ clustering → feature engineering → preprocessing → model → prediction
```

---

## Notes

* Invalid SMILES will result in NaN descriptors or errors
* All preprocessing steps must match training
* Feature order is critical (handled via pipeline["features"])

---

## Limitations

* Accuracy depends on descriptor quality and training data
* 3D descriptor generation may fail for complex molecules
* Not validated for all chemical spaces

---

## Future Improvements

* Wrap into sklearn Pipeline
* Add FastAPI deployment
* GPU acceleration for LightGBM
* Batch prediction optimization

---

## Author

Vovea Dragoș-Cătălin

---

## License

MIT License

## Acknowledgments

This work was developed as part of research at Babeș-Bolyai University under the supervision of Prof. Vasile Chiș.



