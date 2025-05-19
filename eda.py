import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold


#Statistical evaluation
def calculate_pvalues(df):
    df = df.dropna()  
    cols = df.columns
    pvals = pd.DataFrame(np.ones((len(cols), len(cols))), columns=cols, index=cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j:
                _, p = pearsonr(df[cols[i]], df[cols[j]])
                pvals.iloc[i, j] = p
            else:
                pvals.iloc[i, j] = 0.0  
    return pvals

#Correlation analysis
def analyze_correlations(data):
    # Compute Pearson correlation matrix
    corr_matrix = data.corr() 
    corr_matrix = round(corr_matrix, 2)
    print(f"Pearson Correlation matrix: {corr_matrix}")
    
    p_value_matrix = calculate_pvalues(data)
    filtered_corr = corr_matrix.copy()
    filtered_corr[p_value_matrix > 0.05] = 0
    print(f"Filtered Correlation matrix (statistically evaluated): {filtered_corr}")
    
    fig = px.imshow(filtered_corr, text_auto=True, aspect="auto")
    fig.show()
    return filtered_corr

#Identifiy activity cliffs (pairs of structurally similar molecules that differ significantly in their biological activity (ΔpIC50))
def find_activity_cliffs(data, delta_threshold=1.0):
    """
    Input:
    data: A DataFrame containing molecular information with columns:
    delta_threshold: The minimum difference in pIC50 values to consider a pair as an activity cliff. Default is 1.0.
    Output:
    pd.DataFrame: A DataFrame containing the identified activity cliffs with columns:
        - 'SMILES_1': SMILES string of the first molecule.
        - 'SMILES_2': SMILES string of the second molecule.
        - 'ΔpIC50': The difference in pIC50 between the two molecules.
        - 'Similarity': Structural similarity calculated using Maximum Common Substructure (MCS).
        - 'MCS': SMARTS string of the MCS between the two molecules.
    """
    cliffs = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            mol1 = data.iloc[i]['mol']
            numAtoms1 = data.iloc[i]['num_of_atoms']
            mol2 = data.iloc[j]['mol']
            numAtoms2 = data.iloc[j]['num_of_atoms']
            # Find maximum common substructure
            mcs = rdFMCS.FindMCS([mol1, mol2])
            if mcs.numAtoms < min(numAtoms1, numAtoms2):
                similarity = mcs.numAtoms / max(numAtoms1,numAtoms2)
                if similarity > 0.7:  # more than 70% structural similarity
                    delta_pIC50 = abs(data.iloc[i]['pIC50'] - data.iloc[j]['pIC50'])
                    if delta_pIC50 >= delta_threshold:
                        cliffs.append((data.iloc[i]['SMILES'], data.iloc[j]['SMILES'], 
                                     delta_pIC50, similarity, mcs.smartsString))
    return pd.DataFrame(cliffs, columns=['SMILES_1', 'SMILES_2', 'ΔpIC50','Similarity','MCS'])

#Scaffold analysis
def analyze_scaffolds(data):
    data['Scaffold'] = data['mol'].apply(lambda x: 
        Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(x)))
    
    scaffold_stats = data.groupby('Scaffold').agg(
        count=('pIC50', 'size'),
        mean_pIC50=('pIC50', 'mean'),
        min_pIC50=('pIC50', 'min'),
        max_pIC50=('pIC50', 'max')
    ).sort_values('count', ascending=False)
        
    plt.figure(figsize=(10, 6))
    plt.scatter(scaffold_stats['count'], scaffold_stats['mean_pIC50'], 
                c=scaffold_stats['max_pIC50']-scaffold_stats['min_pIC50'],
                cmap='viridis', alpha=0.6)
    plt.colorbar(label='pIC50 Range')
    plt.xlabel('Number of Compounds')
    plt.ylabel('Average pIC50')
    plt.title('Scaffold Analysis')
    plt.show()
    return scaffold_stats

