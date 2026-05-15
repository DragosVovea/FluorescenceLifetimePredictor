import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.Chem.EState import Fingerprinter

try:
    from pkapredict import load_model, smiles_to_rdkit_descriptors
    PKA_AVAILABLE = True
except Exception:
    PKA_AVAILABLE = False


MODEL_PATH = "FinalModel.pkl"
KMEANS_PATH = "FinalKMeans.pkl"


RENAME_3D = {
    "Chrom3D_BondLength_Mean": "C3_BL_Mean",
    "Chrom3D_BondLength_Max": "C3_BL_Max",
    "Chrom3D_BondLength_Min": "C3_BL_Min",
    "Chrom3D_Angle_Mean": "C3_Ang_Mean",
    "Chrom3D_Angle_Max": "C3_Ang_Max",
    "Chrom3D_Angle_Min": "C3_Ang_Min",
    "Chrom3D_Dihedral_Mean": "C3_Dih_Mean",
    "Chrom3D_Dihedral_Std": "C3_Dih_Std",
    "Chrom3D_Planarity_RMSD": "C3_Plan_RMSD",
    "Chrom3D_RadiusOfGyration": "C3_Rg",
    "Solv3D_BondLength_Mean": "S3_BL_Mean",
    "Solv3D_BondLength_Max": "S3_BL_Max",
    "Solv3D_BondLength_Min": "S3_BL_Min",
    "Solv3D_Angle_Mean": "S3_Ang_Mean",
    "Solv3D_Angle_Max": "S3_Ang_Max",
    "Solv3D_Angle_Min": "S3_Ang_Min",
    "Solv3D_Dihedral_Mean": "S3_Dih_Mean",
    "Solv3D_Dihedral_Std": "S3_Dih_Std",
    "Solv3D_Planarity_RMSD": "S3_Plan_RMSD",
    "Solv3D_RadiusOfGyration": "S3_Rg",
}


def canonicalize_smiles(smiles):
    """Return canonical SMILES. Return None if invalid."""
    try:
        if smiles is None or pd.isna(smiles):
            return None
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def mol_from_smiles(smiles):
    """Return RDKit molecule. Return None if invalid."""
    try:
        if smiles is None or pd.isna(smiles):
            return None
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def prepare_input(data):
    """
    Accept a list of dictionaries or a pandas DataFrame.

    Required:
        Chromophore
        Solvent

    Optional:
        Absorption max (nm)
        Emission max (nm)
        Quantum yield
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)

    aliases = {
        "chromophore": "Chromophore",
        "chromophore smiles": "Chromophore",
        "chromophore_smiles": "Chromophore",
        "solvent": "Solvent",
        "solvent smiles": "Solvent",
        "solvent_smiles": "Solvent",
        "absorption": "Absorption max (nm)",
        "absorption max": "Absorption max (nm)",
        "absorption max (nm)": "Absorption max (nm)",
        "emission": "Emission max (nm)",
        "emission max": "Emission max (nm)",
        "emission max (nm)": "Emission max (nm)",
        "quantum yield": "Quantum yield",
        "qunatum yield": "Quantum yield",
        "qy": "Quantum yield",
    }

    df = df.rename(columns={c: aliases.get(c.strip().lower(), c) for c in df.columns})

    for col in ["Chromophore", "Solvent"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    for col in ["Absorption max (nm)", "Emission max (nm)", "Quantum yield"]:
        if col not in df.columns:
            df[col] = np.nan

    df["Chromophore"] = df["Chromophore"].apply(canonicalize_smiles)
    df["Solvent"] = df["Solvent"].apply(canonicalize_smiles)

    if df["Chromophore"].isna().any():
        raise ValueError("Invalid chromophore SMILES detected.")

    if df["Solvent"].isna().any():
        raise ValueError("Invalid solvent SMILES detected.")

    return df


def rdkit_descriptors(mol):
    """Calculate all RDKit descriptors plus charge and E-State summary descriptors."""
    values = {}

    if mol is None:
        for name, _ in Descriptors.descList:
            values[name] = np.nan
        values["MaxCharge"] = np.nan
        values["MinCharge"] = np.nan
        values["MeanCharge"] = np.nan
        values["EState_Sum"] = np.nan
        return pd.Series(values)

    for name, func in Descriptors.descList:
        try:
            values[name] = func(mol)
        except Exception:
            values[name] = np.nan

    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            q = atom.GetProp("_GasteigerCharge")
            q = float(q) if q not in ["nan", "-nan", "inf", "-inf"] else np.nan
            charges.append(q)

        values["MaxCharge"] = np.nanmax(charges)
        values["MinCharge"] = np.nanmin(charges)
        values["MeanCharge"] = np.nanmean(charges)
    except Exception:
        values["MaxCharge"] = np.nan
        values["MinCharge"] = np.nan
        values["MeanCharge"] = np.nan

    try:
        estate = Fingerprinter.FingerprintMol(mol)
        estate_values = estate[0] if isinstance(estate, tuple) else estate
        values["EState_Sum"] = float(np.sum(estate_values))
    except Exception:
        values["EState_Sum"] = np.nan

    return pd.Series(values)


def fingerprints(smiles_series, prefix, n_bits=512):
    """
    Calculate Morgan and MACCS fingerprints.

    Example columns:
        Chrom_Morgan_0 ... Chrom_Morgan_511
        Chrom_MACCS_0 ... Chrom_MACCS_166
    """
    rows = []

    for smiles in smiles_series:
        mol = mol_from_smiles(smiles)
        row = {}

        if mol is None:
            for i in range(n_bits):
                row[f"{prefix}_Morgan_{i}"] = np.nan
            for i in range(167):
                row[f"{prefix}_MACCS_{i}"] = np.nan
            rows.append(row)
            continue

        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            arr = np.array(fp, dtype=int)
            for i, value in enumerate(arr):
                row[f"{prefix}_Morgan_{i}"] = int(value)
        except Exception:
            for i in range(n_bits):
                row[f"{prefix}_Morgan_{i}"] = np.nan

        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.array(fp, dtype=int)
            for i, value in enumerate(arr):
                row[f"{prefix}_MACCS_{i}"] = int(value)
        except Exception:
            for i in range(167):
                row[f"{prefix}_MACCS_{i}"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def plane_rmsd(mol, conf, atom_indices):
    """RMSD of selected atoms from the best-fit plane."""
    if len(atom_indices) < 3:
        return np.nan

    coords = np.array([list(conf.GetAtomPosition(i)) for i in atom_indices])
    coords = coords - coords.mean(axis=0)

    _, _, vh = np.linalg.svd(coords)
    normal = vh[2]
    distances = np.dot(coords, normal)

    return float(np.sqrt(np.mean(distances ** 2)))


def descriptors_3d(mol):
    """Calculate simple 3D molecular geometry descriptors."""
    empty = {
        "BondLength_Mean": np.nan,
        "BondLength_Max": np.nan,
        "BondLength_Min": np.nan,
        "Angle_Mean": np.nan,
        "Angle_Max": np.nan,
        "Angle_Min": np.nan,
        "Dihedral_Mean": np.nan,
        "Dihedral_Std": np.nan,
        "Planarity_RMSD": np.nan,
        "RadiusOfGyration": np.nan,
    }

    if mol is None:
        return empty

    mol_h = Chem.AddHs(mol)

    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol_h, params)

        if status != 0:
            return empty

        AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
        conf = mol_h.GetConformer()
    except Exception:
        return empty

    try:
        bond_lengths = [
            AllChem.GetBondLength(conf, b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            for b in mol_h.GetBonds()
        ]
        empty["BondLength_Mean"] = np.mean(bond_lengths) if bond_lengths else np.nan
        empty["BondLength_Max"] = np.max(bond_lengths) if bond_lengths else np.nan
        empty["BondLength_Min"] = np.min(bond_lengths) if bond_lengths else np.nan
    except Exception:
        pass

    try:
        angles = []
        for atom in mol_h.GetAtoms():
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    angles.append(AllChem.GetAngleDeg(conf, neighbors[i], atom.GetIdx(), neighbors[j]))

        empty["Angle_Mean"] = np.mean(angles) if angles else np.nan
        empty["Angle_Max"] = np.max(angles) if angles else np.nan
        empty["Angle_Min"] = np.min(angles) if angles else np.nan
    except Exception:
        pass

    try:
        dihedrals = []
        for bond in mol_h.GetBonds():
            if bond.IsInRing():
                continue

            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()

            neighbors_a = [
                n.GetIdx()
                for n in mol_h.GetAtomWithIdx(a).GetNeighbors()
                if n.GetIdx() != b
            ]

            neighbors_b = [
                n.GetIdx()
                for n in mol_h.GetAtomWithIdx(b).GetNeighbors()
                if n.GetIdx() != a
            ]

            for i in neighbors_a:
                for j in neighbors_b:
                    try:
                        dihedrals.append(AllChem.GetDihedralDeg(conf, i, a, b, j))
                    except Exception:
                        pass

        empty["Dihedral_Mean"] = np.mean(dihedrals) if dihedrals else np.nan
        empty["Dihedral_Std"] = np.std(dihedrals) if dihedrals else np.nan
    except Exception:
        pass

    try:
        aromatic_atoms = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetIsAromatic()]
        empty["Planarity_RMSD"] = plane_rmsd(mol_h, conf, aromatic_atoms)
    except Exception:
        pass

    try:
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol_h.GetNumAtoms())])
        center = coords.mean(axis=0)
        empty["RadiusOfGyration"] = float(
            np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
        )
    except Exception:
        pass

    return empty


def descriptors_3d_for_series(smiles_series):
    """Calculate 3D descriptors for a SMILES Series."""
    rows = [descriptors_3d(mol_from_smiles(s)) for s in smiles_series]
    return pd.DataFrame(rows)


def add_predicted_pka(df):
    """
    Add Predicted_pKa.

    If pkapredict is not installed or prediction fails, Predicted_pKa is NaN.
    The trained imputer from FinalModel.pkl will handle the missing value.
    """
    df = df.copy()

    if not PKA_AVAILABLE:
        df["Predicted_pKa"] = np.nan
        return df

    try:
        model = load_model()
        names = model.feature_name_

        desc = pd.DataFrame([
            smiles_to_rdkit_descriptors(smiles, names)
            for smiles in df["Chromophore"].astype(str)
        ])

        df["Predicted_pKa"] = model.predict(desc)
    except Exception:
        df["Predicted_pKa"] = np.nan

    return df


def engineer_features(df):
    """Create engineered features used during model training."""
    df = df.copy()
    df = df.rename(columns=RENAME_3D)

    eps = 1e-6

    df["RigidityScore"] = 1 / (1 + df.get("Chrom_NumRotatableBonds", 0))
    df["ConjugationIndex"] = df.get("Chrom_NumAromaticRings", 0) * df.get("Chrom_AromaticProportion", 0)
    df["HbondPotential"] = df.get("Chrom_NumHDonors", 0) + df.get("Chrom_NumHAcceptors", 0)
    df["PolarityRatio"] = df.get("Chrom_TPSA", 0) / (df.get("Chrom_MolWt", 1) + eps)

    df["AromaticAliphaticRatio"] = (1 - df.get("Chrom_FractionCSP3", 0)) / (
        df.get("Chrom_FractionCSP3", 0) + eps
    )

    df["PolarityDifference"] = (
        df.get("Chrom_TPSA", 0) / (df.get("Chrom_MolWt", 1) + eps)
        - df.get("Solv_TPSA", 0) / (df.get("Solv_MolWt", 1) + eps)
    )

    df["Chrom_KappaRatio"] = df.get("Chrom_Kappa1", np.nan) / (
        df.get("Chrom_Kappa2", np.nan) + df.get("Chrom_Kappa3", np.nan) + eps
    )

    df["Chrom_Flexibility"] = df.get("Chrom_NumRotatableBonds", 0) / (
        df.get("Chrom_MolWt", 1) + eps
    )

    df["Chrom_HbondDensity"] = (
        df.get("Chrom_NumHDonors", 0) + df.get("Chrom_NumHAcceptors", 0)
    ) / (df.get("Chrom_MolWt", 1) + eps)

    df["Chrom_HbondPolarityScore"] = df.get("Chrom_TPSA", 0) * df["PolarityRatio"]
    df["Chrom_PolarityBalance"] = df.get("Chrom_NumHDonors", 0) - df.get("Chrom_NumHAcceptors", 0)
    df["Chrom_ConjugationQuantumInteraction"] = df["ConjugationIndex"] * df["Quantum yield"]
    df["Chrom_Log_MolWt"] = np.log1p(df.get("Chrom_MolWt", np.nan))
    df["Chrom_Sqrt_TPSA"] = np.sqrt(df.get("Chrom_TPSA", np.nan))

    df["Solv_KappaRatio"] = df.get("Solv_Kappa1", np.nan) / (
        df.get("Solv_Kappa2", np.nan) + df.get("Solv_Kappa3", np.nan) + eps
    )

    df["Solv_Flexibility"] = df.get("Solv_NumRotatableBonds", 0) / (
        df.get("Solv_MolWt", 1) + eps
    )

    df["Solv_HbondDensity"] = (
        df.get("Solv_NumHDonors", 0) + df.get("Solv_NumHAcceptors", 0)
    ) / (df.get("Solv_MolWt", 1) + eps)

    df["Solv_Log_MolWt"] = np.log1p(df.get("Solv_MolWt", np.nan))
    df["Solv_Sqrt_TPSA"] = np.sqrt(df.get("Solv_TPSA", np.nan))

    df["AbsEmiRatio"] = df["Absorption max (nm)"] / (df["Emission max (nm)"] + eps)
    df["AbsEmiRatio_sq"] = df["AbsEmiRatio"] ** 2
    df["AbsEmiRatio_cu"] = df["AbsEmiRatio"] ** 3
    df["HasLongAbsorption"] = (df["Absorption max (nm)"] > 400).astype(float)

    df["StokesShift"] = df["Absorption max (nm)"] - df["Emission max (nm)"]
    df["StokesShift_Ratio"] = df["StokesShift"] / (df["Emission max (nm)"] + eps)

    df["ChromSolv_Mw_Ratio"] = df.get("Chrom_MolWt", np.nan) / (df.get("Solv_MolWt", np.nan) + eps)
    df["ChromSolv_LogP_Diff"] = df.get("Chrom_MolLogP", np.nan) - df.get("Solv_MolLogP", np.nan)
    df["ChromSolv_TPSA_Diff"] = df.get("Chrom_TPSA", np.nan) - df.get("Solv_TPSA", np.nan)

    df["IsPolarSolvent"] = (df.get("Solv_TPSA", 0) > 20).astype(float)
    df["Emission_Energy_eV"] = 1240 / (df["Emission max (nm)"] + eps)
    df["Absorption_Energy_eV"] = 1240 / (df["Absorption max (nm)"] + eps)
    df["StokesShift_eV"] = df["Absorption_Energy_eV"] - df["Emission_Energy_eV"]

    return df


def build_features(data, kmeans_model=None):
    """Build the full descriptor matrix for prediction."""
    df = prepare_input(data)

    chrom_desc = pd.DataFrame(
        [rdkit_descriptors(mol_from_smiles(s)) for s in df["Chromophore"]]
    ).add_prefix("Chrom_")

    solv_desc = pd.DataFrame(
        [rdkit_descriptors(mol_from_smiles(s)) for s in df["Solvent"]]
    ).add_prefix("Solv_")

    chrom_3d = descriptors_3d_for_series(df["Chromophore"]).add_prefix("Chrom3D_")
    solv_3d = descriptors_3d_for_series(df["Solvent"]).add_prefix("Solv3D_")

    chrom_fp = fingerprints(df["Chromophore"], prefix="Chrom", n_bits=512)
    solv_fp = fingerprints(df["Solvent"], prefix="Solv", n_bits=512)

    df = add_predicted_pka(df)

    features = pd.concat(
        [
            df.reset_index(drop=True),
            chrom_desc.reset_index(drop=True),
            solv_desc.reset_index(drop=True),
            chrom_3d.reset_index(drop=True),
            solv_3d.reset_index(drop=True),
            chrom_fp.reset_index(drop=True),
            solv_fp.reset_index(drop=True),
        ],
        axis=1,
    )

    if kmeans_model is not None:
        clusters = []

        for smiles in features["Chromophore"]:
            mol = mol_from_smiles(smiles)

            if mol is None:
                clusters.append(np.nan)
                continue

            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                x = np.array(fp, dtype=int).reshape(1, -1)
                clusters.append(kmeans_model.predict(x)[0])
            except Exception:
                clusters.append(np.nan)

        features["Cluster"] = clusters

    return engineer_features(features)


def predict_lifetime(data, model_path=MODEL_PATH, kmeans_path=KMEANS_PATH):
    """
    Predict fluorescence lifetime in nanoseconds.

    Parameters
    ----------
    data : list of dictionaries or pandas DataFrame

    Returns
    -------
    pandas DataFrame
    """
    pipeline = joblib.load(model_path)
    kmeans_model = joblib.load(kmeans_path)

    features = build_features(data, kmeans_model=kmeans_model)

    # Select exactly the features used during training.
    x = features.reindex(columns=pipeline["features"])

    # Apply trained preprocessing.
    x = pipeline["imputer"].transform(x)

    if pipeline.get("power_transformer") is not None:
        x = pipeline["power_transformer"].transform(x)

    if pipeline.get("scaler") is not None:
        x = pipeline["scaler"].transform(x)

    if pipeline.get("minmax") is not None:
        x = pipeline["minmax"].transform(x)

    # Model was trained on log1p(lifetime), so convert back with expm1.
    pred_log = pipeline["model"].predict(x)
    pred_lifetime = np.expm1(pred_log)

    result = prepare_input(data)
    result["Predicted lifetime (ns)"] = pred_lifetime

    return result


if __name__ == "__main__":
    data = [
        {
            "Chromophore": "O=C1OC2=CC=CC=C2C=C1",
            "Solvent": "CCO",
            "Absorption max (nm)": None,
            "Emission max (nm)": None,
            "Quantum yield": None,
        }
    ]

    result = predict_lifetime(data)

    print("\nPredicted fluorescence lifetime:")
    print(result.to_string(index=False))
