import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.EState import Fingerprinter
from pkapredict import load_model, smiles_to_rdkit_descriptors

rename_map = {
    'Chrom3D_BondLength_Mean': 'C3_BL_Mean',
    'Chrom3D_BondLength_Max': 'C3_BL_Max',
    'Chrom3D_BondLength_Min': 'C3_BL_Min',

    'Chrom3D_Angle_Mean': 'C3_Ang_Mean',
    'Chrom3D_Angle_Max': 'C3_Ang_Max',
    'Chrom3D_Angle_Min': 'C3_Ang_Min',

    'Chrom3D_Dihedral_Mean': 'C3_Dih_Mean',
    'Chrom3D_Dihedral_Std': 'C3_Dih_Std',

    'Chrom3D_Planarity_RMSD': 'C3_Plan_RMSD',
    'Chrom3D_RadiusOfGyration': 'C3_Rg',

    'Solv3D_BondLength_Mean': 'S3_BL_Mean',
    'Solv3D_BondLength_Max': 'S3_BL_Max',
    'Solv3D_BondLength_Min': 'S3_BL_Min',

    'Solv3D_Angle_Mean': 'S3_Ang_Mean',
    'Solv3D_Angle_Max': 'S3_Ang_Max',
    'Solv3D_Angle_Min': 'S3_Ang_Min',

    'Solv3D_Dihedral_Mean': 'S3_Dih_Mean',
    'Solv3D_Dihedral_Std': 'S3_Dih_Std',

    'Solv3D_Planarity_RMSD': 'S3_Plan_RMSD',
    'Solv3D_RadiusOfGyration': 'S3_Rg'
}


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None


# -----------------------------
# Molecular descriptor calculation (all RDKit descriptors)
# -----------------------------
def calculate_all_rdkit_descriptors(mol):
    if mol is None:
        return pd.Series({d_name: np.nan for d_name, _ in Descriptors.descList})

    desc_values = {}
    for d_name, d_func in Descriptors.descList:
        try:
            desc_values[d_name] = d_func(mol)
        except:
            desc_values[d_name] = np.nan

    # Quantum-like / electronic descriptors (Gasteiger charges)
    try:
        Chem.AllChem.ComputeGasteigerCharges(mol)
        charges = [float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()]
        desc_values.update({
            'MaxCharge': max(charges),
            'MinCharge': min(charges),
            'MeanCharge': np.mean(charges)
        })
    except:
        desc_values.update({'MaxCharge': np.nan, 'MinCharge': np.nan, 'MeanCharge': np.nan})

    # E-State fingerprint (sum)
    try:
        estate_fp = Fingerprinter.FingerprintMol(mol)
        desc_values['EState_Sum'] = sum(estate_fp)
    except:
        desc_values['EState_Sum'] = np.nan

    return pd.Series(desc_values)


def calculate_solvent_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return calculate_all_rdkit_descriptors(mol)



def rmsd_from_plane(mol, conf, atom_indices):
    """Compute RMS deviation from the best-fit plane for selected atoms."""
    coords = np.array([list(conf.GetAtomPosition(i)) for i in atom_indices])
    coords_centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(coords_centered)
    normal_vector = vh[2]
    distances = np.dot(coords_centered, normal_vector)
    return np.sqrt(np.mean(distances ** 2))


def compute_3d_descriptors(mol):
    """
    Generate 3D conformer and compute geometric descriptors for a single RDKit molecule.
    Returns a dictionary of descriptors:
        - BondLength_Mean, BondLength_Max, BondLength_Min
        - Angle_Mean, Angle_Max, Angle_Min
        - Dihedral_Mean, Dihedral_Std
        - Planarity_RMSD
        - RadiusOfGyration
    """
    mol_H = Chem.AddHs(mol)
    try:
        # Generate 3D conformer
        AllChem.EmbedMolecule(mol_H, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol_H)
        conf = mol_H.GetConformer()
    except:
        return {
            'BondLength_Mean': np.nan, 'BondLength_Max': np.nan, 'BondLength_Min': np.nan,
            'Angle_Mean': np.nan, 'Angle_Max': np.nan, 'Angle_Min': np.nan,
            'Dihedral_Mean': np.nan, 'Dihedral_Std': np.nan,
            'Planarity_RMSD': np.nan, 'RadiusOfGyration': np.nan
        }

    # -----------------------------
    # Bond lengths
    # -----------------------------
    bond_lengths = [AllChem.GetBondLength(conf, b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_H.GetBonds()]
    bond_length_mean = np.mean(bond_lengths) if bond_lengths else np.nan
    bond_length_max = np.max(bond_lengths) if bond_lengths else np.nan
    bond_length_min = np.min(bond_lengths) if bond_lengths else np.nan

    # -----------------------------
    # Bond angles
    # -----------------------------
    angles = []
    for atom in mol_H.GetAtoms():
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 2:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                angle = AllChem.GetAngleDeg(conf, neighbors[i], atom.GetIdx(), neighbors[j])
                angles.append(angle)
    angle_mean = np.mean(angles) if angles else np.nan
    angle_max = np.max(angles) if angles else np.nan
    angle_min = np.min(angles) if angles else np.nan

    # -----------------------------
    # Dihedral / torsion angles
    # -----------------------------
    dihedrals = []
    for bond in mol_H.GetBonds():
        if bond.IsInRing():
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_neighbors = [n.GetIdx() for n in mol_H.GetAtomWithIdx(begin).GetNeighbors() if n.GetIdx() != end]
        end_neighbors = [n.GetIdx() for n in mol_H.GetAtomWithIdx(end).GetNeighbors() if n.GetIdx() != begin]
        for i in begin_neighbors:
            for j in end_neighbors:
                try:
                    dih = AllChem.GetDihedralDeg(conf, i, begin, end, j)
                    dihedrals.append(dih)
                except:
                    continue
    dihedral_mean = np.mean(dihedrals) if dihedrals else np.nan
    dihedral_std = np.std(dihedrals) if dihedrals else np.nan

    # -----------------------------
    # Planarity (aromatic atoms)
    # -----------------------------
    aromatic_atoms = [a.GetIdx() for a in mol_H.GetAtoms() if a.GetIsAromatic()]
    planarity_rmsd = rmsd_from_plane(mol_H, conf, aromatic_atoms) if aromatic_atoms else np.nan

    # -----------------------------
    # Radius of gyration
    # -----------------------------
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol_H.GetNumAtoms())])
    center = coords.mean(axis=0)
    radius_of_gyration = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))

    return {
        'BondLength_Mean': bond_length_mean,
        'BondLength_Max': bond_length_max,
        'BondLength_Min': bond_length_min,
        'Angle_Mean': angle_mean,
        'Angle_Max': angle_max,
        'Angle_Min': angle_min,
        'Dihedral_Mean': dihedral_mean,
        'Dihedral_Std': dihedral_std,
        'Planarity_RMSD': planarity_rmsd,
        'RadiusOfGyration': radius_of_gyration
    }


def calculate_3d_descriptors_for_smiles_series(smiles_series):
    """
    Compute 3D geometric descriptors for a pandas Series of SMILES strings.
    Returns a DataFrame with one row per molecule.
    """
    geom_desc_list = []

    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            geom_desc_list.append({
                'BondLength_Mean': np.nan, 'BondLength_Max': np.nan, 'BondLength_Min': np.nan,
                'Angle_Mean': np.nan, 'Angle_Max': np.nan, 'Angle_Min': np.nan,
                'Dihedral_Mean': np.nan, 'Dihedral_Std': np.nan,
                'Planarity_RMSD': np.nan, 'RadiusOfGyration': np.nan
            })
            continue

        geom_desc = compute_3d_descriptors(mol)
        geom_desc_list.append(geom_desc)

    return pd.DataFrame(geom_desc_list)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=rename_map)
    eps=1e-6
    eps = 1e-6
    df['RigidityScore'] = 1 / (1 + df.get('Chrom_NumRotatableBonds', 0))
    df['ConjugationIndex'] = df.get('Chrom_NumAromaticRings', 0) * df.get('Chrom_AromaticProportion', 0)
    df['HbondPotential'] = df.get('Chrom_NumHDonors', 0) + df.get('Chrom_NumHAcceptors', 0)
    df['PolarityRatio'] = df.get('Chrom_TPSA', 0) / df.get('Chrom_MolWt', 1)
    df['AromaticAliphaticRatio'] = (1 - df.get('Chrom_FractionCSP3', 0)) / (df.get('Chrom_FractionCSP3', 0) + eps)
    df['PolarityDifference'] = (df.get('Chrom_TPSA', 0) / (df.get('Chrom_MolWt', 1) + eps)) - (
            df.get('Solv_TPSA', 0) / (df.get('Solv_MolWt', 1) + eps))

    # Chromophore features
    df['Chrom_KappaRatio'] = df['Chrom_Kappa1'] / (df['Chrom_Kappa2'] + df['Chrom_Kappa3'] + 1e-6)
    df['Chrom_Flexibility'] = df['Chrom_NumRotatableBonds'] / (df['Chrom_MolWt'] + 1e-6)
    df['Chrom_HbondDensity'] = (df['Chrom_NumHDonors'] + df['Chrom_NumHAcceptors']) / (df['Chrom_MolWt'] + 1e-6)
    df['Chrom_HbondPolarityScore'] = df['Chrom_TPSA'] * df['PolarityRatio']
    df['Chrom_PolarityBalance'] = df['Chrom_NumHDonors'] - df['Chrom_NumHAcceptors']
    df['Chrom_ConjugationQuantumInteraction'] = df['ConjugationIndex'] * df['Quantum yield']
    df['Chrom_Log_MolWt'] = np.log1p(df['Chrom_MolWt'])
    df['Chrom_Sqrt_TPSA'] = np.sqrt(df['Chrom_TPSA'])

    # Solvent features
    df['Solv_KappaRatio'] = df['Solv_Kappa1'] / (df['Solv_Kappa2'] + df['Solv_Kappa3'] + 1e-6)
    df['Solv_Flexibility'] = df['Solv_NumRotatableBonds'] / (df['Solv_MolWt'] + 1e-6)
    df['Solv_HbondDensity'] = (df['Solv_NumHDonors'] + df['Solv_NumHAcceptors']) / (df['Solv_MolWt'] + 1e-6)
    df['Solv_Log_MolWt'] = np.log1p(df['Solv_MolWt'])
    df['Solv_Sqrt_TPSA'] = np.sqrt(df['Solv_TPSA'])

    # Photophysical features
    df['AbsEmiRatio'] = df['Absorption max (nm)'] / (df['Emission max (nm)'] + 1e-6)
    df['AbsEmiRatio_sq'] = df['AbsEmiRatio'] ** 2
    df['AbsEmiRatio_cu'] = df['AbsEmiRatio'] ** 3
    df['HasLongAbsorption'] = (df['Absorption max (nm)'] > 400).astype(int)
    df['StokesShift'] = df['Absorption max (nm)'] - df['Emission max (nm)']
    df['StokesShift_Ratio'] = df['StokesShift'] / (df['Emission max (nm)'] + 1e-6)

    # Chromophore-solvent interactions
    df['ChromSolv_Mw_Ratio'] = df['Chrom_MolWt'] / (df['Solv_MolWt'] + 1e-6)
    df['ChromSolv_LogP_Diff'] = df['Chrom_MolLogP'] - df['Solv_MolLogP']
    df['ChromSolv_TPSA_Diff'] = df['Chrom_TPSA'] - df['Solv_TPSA']

    # Energy features
    df['IsPolarSolvent'] = (df['Solv_TPSA'] > 20).astype(int)
    df['Emission_Energy_eV'] = 1240 / (df['Emission max (nm)'] + 1e-6)
    df['Absorption_Energy_eV'] = 1240 / (df['Absorption max (nm)'] + 1e-6)
    df['StokesShift_eV'] = df['Absorption_Energy_eV'] - df['Emission_Energy_eV']


    return df

# =========================================================
# LOAD pKa MODEL (ONCE)
# =========================================================
model_PKA = load_model()
descriptor_names = model_PKA.feature_name_


# =========================================================
# MAIN FEATURE PIPELINE
# =========================================================
def build_features(df, kmeans_model=None):
    df = df.copy()
    eps = 1e-6

    # =====================================================
    # 1. CANONICAL SMILES
    # =====================================================
    df["Chromophore"] = df["Chromophore"].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
    )

    # =====================================================
    # 2. RDKit DESCRIPTORS (CHROMOPHORE)
    # =====================================================
    chrom_desc = []

    for smi in df["Chromophore"]:
        mol = Chem.MolFromSmiles(smi)
        chrom_desc.append(calculate_all_rdkit_descriptors(mol))

    chrom_desc_df = pd.DataFrame(chrom_desc).add_prefix("Chrom_")

    # =====================================================
    # 3. SOLVENT DESCRIPTORS
    # =====================================================
    solv_desc = []

    for smi in df["Solvent"]:
        mol = Chem.MolFromSmiles(smi)
        solv_desc.append(calculate_all_rdkit_descriptors(mol))

    solv_desc_df = pd.DataFrame(solv_desc).add_prefix("Solv_")

    # =====================================================
    # 4. 3D DESCRIPTORS
    # =====================================================
    chrom_3d = calculate_3d_descriptors_for_smiles_series(df["Chromophore"])
    chrom_3d = chrom_3d.add_prefix("Chrom3D_")

    solv_3d = calculate_3d_descriptors_for_smiles_series(df["Solvent"])
    solv_3d = solv_3d.add_prefix("Solv3D_")

    # =====================================================
    # 5. pKa PREDICTION
    # =====================================================
    desc_df = pd.DataFrame([
        smiles_to_rdkit_descriptors(s, descriptor_names)
        for s in df["Chromophore"].astype(str)
    ])

    df["Predicted_pKa"] = model_PKA.predict(desc_df)

    # =====================================================
    # 6. MERGE ALL FEATURES
    # =====================================================
    df = pd.concat([
        df.reset_index(drop=True),
        chrom_desc_df,
        solv_desc_df,
        chrom_3d,
        solv_3d
    ], axis=1)

    # =====================================================
    # 7. CLUSTER FEATURE (Morgan FP)
    # =====================================================
    if kmeans_model is not None:
        clusters = []

        for smi in df["Chromophore"]:
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                clusters.append(np.nan)
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048
            )

            X = np.array(fp).reshape(1, -1)
            cluster = kmeans_model.predict(X)[0]

            clusters.append(cluster)

        df["Cluster"] = clusters

    # =====================================================
    # 8. ENGINEER FEATURES (YOUR LOGIC)
    # =====================================================
    df = engineer_features(df)

    return df

import joblib

# Load models
pipeline = joblib.load("fluoresencePredictor.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")

data= [
    {
        "Chromophore": None,   # MUST
        "Solvent": None,                  # MUST
        "Absorption max (nm)": None,
        "Emission max (nm)": None,
        "Quantum yield": None
    }
]

df = pd.DataFrame(data)

# Build full features
df_features = build_features(df, kmeans_model)

# Select trained features
X = df_features[pipeline["features"]]

# Apply preprocessing
X_proc = pipeline["imputer"].transform(X)
X_proc = pipeline["power_transformer"].transform(X_proc)
X_proc = pipeline["scaler"].transform(X_proc)
X_proc = pipeline["minmax"].transform(X_proc)

# Predict
pred_log = pipeline["model"].predict(X_proc)
pred = np.expm1(pred_log)

# Compare with real values
results = pd.DataFrame({
    "Predicted Lifetime": pred
})

print(results)