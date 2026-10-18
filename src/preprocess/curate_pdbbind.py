import os
import shutil
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# ============================================================
# PATH CONFIG
# ============================================================
PROJECT_ROOT = os.path.abspath("../../")

INDEX_GENERAL = os.path.join(PROJECT_ROOT, "data/raw/PDBbind_v2020/index/INDEX_general_PL_data.2020")
INDEX_REFINED = os.path.join(PROJECT_ROOT, "data/raw/PDBbind_v2020/index/INDEX_refined_data.2020")

RAW_DIRS = [
    os.path.join(PROJECT_ROOT, "data/raw/PDBbind_v2020/refined-set"),
    os.path.join(PROJECT_ROOT, "data/raw/PDBbind_v2020/general-set")
]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/curated_pdbbind")

# ============================================================
# PARSE INDEX 
# ============================================================
def parse_index(index_file, source_name):
    records = []
    with open(index_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
                
            cols = line.split()
            if len(cols) < 5: 
                continue

            try:
                pdb_id = cols[0].lower()
            
                try:
                    resolution = float(cols[1])
                except Exception:
                    resolution = 3.0  
                pk = float(cols[3])
            except Exception:
                continue

            affinity = cols[4].lower()

            if any(x in affinity for x in ["<", ">", "~"]):
                continue

            records.append({
                "pdb_id": pdb_id,
                "resolution": resolution,
                "pk": pk,
                "source": source_name
            })
    return pd.DataFrame(records)

# ============================================================
# FIND COMPLEX DIRECTORY
# ============================================================
def find_complex_dir(pdb_id):
    for root in RAW_DIRS:
        complex_dir = os.path.join(root, pdb_id)
        if os.path.isdir(complex_dir):
            return complex_dir
    return None

# ============================================================
# LIGAND QUALITY CHECK
# ============================================================
def ligand_quality_check(sdf_file):
    try:
        mol = Chem.MolFromMolFile(sdf_file, sanitize=False)
        if mol is None: 
            return False, "rdkit"
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        if mol.GetNumHeavyAtoms() < 5:
            return False, "heavy_atoms"
        if Descriptors.MolWt(mol) < 100:
            return False, "mw"

        return True, mol
    except Exception:
        return False, "rdkit"

# ============================================================
# MAIN EXECUTION
# ============================================================
def curate_pdbbind():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading PDBbind index with high-volume inclusion policy...")

    df_refined = parse_index(INDEX_REFINED, "refined")
    df_general = parse_index(INDEX_GENERAL, "general")

    df = pd.concat([df_refined, df_general], ignore_index=True)
    df = df.drop_duplicates(subset=["pdb_id"], keep="first")

    print(f"Initial matched index complexes: {len(df)}")

    stats = {
        "resolution": 0,
        "missing": 0,
        "rdkit": 0,
        "heavy_atoms": 0,
        "mw": 0,
        "success": 0
    }

    keep_records = []
    for _, row in df.iterrows():
        pdb_id = row["pdb_id"]

        if row["resolution"] > 3.0:
            stats["resolution"] += 1
            continue

        complex_dir = find_complex_dir(pdb_id)
        if complex_dir is None:
            stats["missing"] += 1
            continue

        ligand_file = os.path.join(complex_dir, f"{pdb_id}_ligand.sdf")
        if not os.path.exists(ligand_file):
            stats["missing"] += 1
            continue

        valid, result = ligand_quality_check(ligand_file)
        if not valid:
            stats[result] += 1
            continue

        output_complex_dir = os.path.join(OUTPUT_DIR, pdb_id)
        try:
            shutil.copytree(complex_dir, output_complex_dir, dirs_exist_ok=True)
        except Exception as e:
            print(f"Copy failed: {pdb_id} -> {e}")
            continue

        keep_records.append(row)
        stats["success"] += 1

    df_final = pd.DataFrame(keep_records)
    output_csv = os.path.join(OUTPUT_DIR, "curated_indices.csv")
    df_final.to_csv(output_csv, index=False)

    print("\n========== DATA CURATION SUMMARY ==========")
    print(f"Resolution >3.0Å (Filtered)  : {stats['resolution']}")
    print(f"Missing PDB/SDF assets       : {stats['missing']}")
    print(f"RDKit Parse failures         : {stats['rdkit']}")
    print(f"Medicinal Heavy atoms <5     : {stats['heavy_atoms']}")
    print(f"Medicinal MW <100 Da         : {stats['mw']}")
    print(f"🔥 Final Valid Complexes     : {stats['success']}")
    print("=================================================")
    print(f"\nCurated Assets Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    curate_pdbbind()