import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def load_and_prepare_data(file_path, properties_path):
    # Load data
    data = pd.read_csv(file_path)
    # Remove the fourth Unnamed column and observations with pIC50 == BLINDED
    data = data.iloc[:, 0:3]
    data = data[data['pIC50 (IC50 in microM)'] != "BLINDED"]
    data = data.reset_index(drop=True)
    
    # Load data with properties 
    data_properties = pd.read_csv(properties_path)
    
    # Merge data with their compounds properties
    data = pd.merge(data, data_properties, on = 'SMILES', how = 'left')
    
    # Drop unnecessary columns
    data = data.drop(['Compound No.', 'pIC50 (IC50 in microM)', 'CID',
           'MolecularFormula', 'InChI', 'InChIKey', 'IUPACName','Charge', 'IsotopeAtomCount', 
                  'DefinedAtomStereoCount', 'UndefinedBondStereoCount', 'CovalentUnitCount'], axis=1)
    
    data['pIC50'] = pd.to_numeric(data['pIC50'], errors='coerce') 
    # Remove NaNs values 
    data = data.dropna(axis=0)
    data = data.reset_index(drop=True)
    
    return data

def add_molecular_features(data):
    # Add molecular features
    data['mol'] = data['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
    data['mol_with_H'] = data['mol'].apply(lambda x: Chem.AddHs(x))
    data['num_of_atoms'] = data['mol'].apply(lambda x: x.GetNumAtoms())
    data['num_of_heavy_atoms'] = data['mol'].apply(lambda x: x.GetNumHeavyAtoms())
    
    # Count specific atoms
    def count_atoms_by_symbol(atom_symbols, data):
        for symbol in atom_symbols:
            data[f'num_{symbol}'] = data['mol'].apply(
                lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)
            )
    
    count_atoms_by_symbol(['C', 'O', 'N', 'Cl', 'Br', 'F'], data)
    
    # Add other descriptors
    data['tpsa'] = data['mol'].apply(lambda x: Descriptors.TPSA(x)) 
    data['mol_w'] = data['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    data['num_valence_electrons'] = data['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    data['num_heteroatoms'] = data['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
    
    return data

