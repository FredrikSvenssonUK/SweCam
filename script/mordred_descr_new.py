"""
Script for the Tox21 challenge.
https://github.com/FredrikSvenssonUK/SweCam/blob/main/LICENSE

Calculate Mordred descriptors.
Uses https://github.com/JacksonBurns/mordred-community

Moriwaki, H., Tian, YS., Kawashita, N. et al. Mordred: a molecular 
descriptor calculator. J Cheminform 10, 4 (2018). 
https://doi.org/10.1186/s13321-018-0258-y
"""


__author__ = "Ulf Norinder, Fredrik Svensson"
__date__ = "15/08/24"


### Imports ###
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from mordred import Calculator, descriptors


### Config ###
infile_train = "data/tox24_challenge_train_standard.csv"
infile_test = "data/tox24_challenge_test_standard.csv"
infile_leaderboard = "data/tox24_challenge_leaderboard_standard.csv"

outfile_train = "data/tox24_challenge_train_mordred.csv"
outfile_test = "data/tox24_challenge_test_mordred.csv"
outfile_leaderboard = "data/tox24_challenge_leaderboard_mordred.csv"


### Main ###
if __name__ == '__main__':
    
    calc = Calculator(descriptors, ignore_3D=True)
    
    # Read input train data and convert to RDKit mol objects
    df = pd.read_csv(infile_train,  sep='\t', header = 0, index_col = None)
    smiles_list = df['SMILES'].to_list()
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    # Calculate descriptrs
    outdf = calc.pandas(mols)
    
    # Drop unwanted columns
    outdf = outdf.drop(columns=['ABC','ABCGG','WPath'], axis=1, errors='ignore')
    cols = outdf.columns
    #outdf = pd.DataFrame(outdf)
    outdf = outdf.astype(str)
    for col in cols:
        outdf[col] = np.where(outdf[col].str.contains('max() arg is an empty sequence', case=False), 0, outdf[col])
    
    outdf = outdf.apply(pd.to_numeric,errors='coerce')
    # Drop columns with nan
    outdf = outdf.dropna(axis=1)
    
    df = df.drop(columns=['SMILES'], axis=1, errors='ignore')
    outdf = pd.concat([df, outdf], axis=1)
    colslst = outdf.columns
    outdf.to_csv(outfile_train, sep='\t', header=False, index=False)
    
    # Read input test data and convert to RDKit mol objects
    df = pd.read_csv(infile_test,  sep='\t', header = 0, index_col = None)
    smiles_list = df['SMILES'].to_list()
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    # Calculate descriptrs
    outdf = calc.pandas(mols)
    # Drop unwanted columns
    outdf = outdf.drop(columns=['ABC','ABCGG','WPath'], axis=1, errors='ignore')
    outdf = outdf.apply(pd.to_numeric,errors='coerce')
    # Drop columns with nan
    outdf = outdf.dropna(axis=1)
    
    df = df.drop(columns=['SMILES'], axis=1, errors='ignore')
    outdf = pd.concat([df, outdf], axis=1)
    outdf = outdf[colslst] # keep the same columns as in the train
    outdf.to_csv(outfile_test, sep='\t', header=False, index=False)
    
    # Read input leaderboard data and convert to RDKit mol objects
    df = pd.read_csv(infile_leaderboard,  sep='\t', header = 0, index_col = None)
    smiles_list = df['SMILES'].to_list()
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    # Calculate descriptrs
    outdf = calc.pandas(mols)
    # Drop unwanted columns
    outdf = outdf.drop(columns=['ABC','ABCGG','WPath'], axis=1, errors='ignore')
    outdf = outdf.apply(pd.to_numeric,errors='coerce')
    # Drop columns with nan
    outdf = outdf.dropna(axis=1)
    
    df = df.drop(columns=['SMILES'], axis=1, errors='ignore')
    outdf = pd.concat([df, outdf], axis=1)
    outdf = outdf[colslst] # keep the same columns as in the train
    outdf.to_csv(outfile_leaderboard, sep='\t', header=False, index=False)
