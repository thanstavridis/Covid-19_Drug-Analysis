from rdkit.Chem import Draw
from rdkit import Chem

#Molecule visualization
def visualize_molecules(data, n=20):
    data_sorted = data.sort_values(by='pIC50', ascending=True)
    # Visualize top and bottom molecules
    mols_low = data_sorted['mol'][:n]
    mols_high = data_sorted['mol'].tail(n)
    
    img_low = Draw.MolsToGridImage(mols_low, molsPerRow=4, 
                                 legends=list(data_sorted['SMILES'][:n].values))
    img_low.save('low_pIC50_molecules.png')
    print("Saved low pIC50 molecules to low_pIC50_molecules.png")
    
    img_high = Draw.MolsToGridImage(mols_high, molsPerRow=4, 
                                  legends=list(data_sorted['SMILES'].tail(n).values))
    img_high.save('high_pIC50_molecules.png')
    print("Saved high pIC50 molecules to high_pIC50_molecules.png")

# Top N activity cliffs visualization
def visualize_activity_cliffs(cliffs_df, data, n=3):
    for i, (_, row) in enumerate(cliffs_df.nlargest(n, 'ΔpIC50').iterrows()):
        mols = [Chem.MolFromSmiles(row['SMILES_1']), Chem.MolFromSmiles(row['SMILES_2'])]
        mcs = Chem.MolFromSmarts(row['MCS'])
        matches = [mol.GetSubstructMatch(mcs) for mol in mols]

        pIC50_1 = data[data['SMILES'] == row['SMILES_1']]['pIC50'].values[0]
        pIC50_2 = data[data['SMILES'] == row['SMILES_2']]['pIC50'].values[0]

        img = Draw.MolsToGridImage(
            mols,
            highlightAtomLists=matches,
            legends=[f"pIC50: {pIC50_1}", f"pIC50: {pIC50_2}"]
        )

        filename = f'cliff_{i+1}.png'
        img.save(filename)
        print(f"Saved image: {filename} displaying 3 activity cliffs with largest ΔpIC50")


