"""
Script for the Tox24 challenge.
https://github.com/FredrikSvenssonUK/SweCam/blob/main/LICENSE

Generate FPs and RDKit descriptors for the input compounds.
Input file is as provided from the challnge webpage (separate train and test csv files).

Before descriptor generation the script will desalt and standardize the 
structures. Standardized smiles will also be saved to a file.
Compounds will also be numbered as an ID.

Only compounds that pass all the descriptor generators will be written to
the result files.

Based on RDKit https://github.com/rdkit/rdkit
"""


__author__ = "Ulf Norinder, Fredrik Svensson"
__date__ = "15/08/24"


### Imports ###
from pathlib import Path

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from molvs import Standardizer


### Config ### 
train_path = Path("data/tox24_challenge_train.csv")
test_path = Path("data/tox24_challenge_test.csv")
leaderboard_path = Path("data/tox24_challenge_leaderboard.csv")

# Set up what RDKit descriptors to calculate
desc_list = ['Chi0','Chi0n','Chi0v','Chi1',\
'Chi1n','Chi1v','Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','Chi4v',\
'EState_VSA1','EState_VSA10','EState_VSA11','EState_VSA2','EState_VSA3',\
'EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8',\
'EState_VSA9','FractionCSP3','HallKierAlpha','HeavyAtomCount','Ipc',\
'Kappa1','Kappa2','Kappa3','LabuteASA','MolLogP','MolMR','MolWt',\
'NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles',\
'NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles',\
'NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms',\
'NumRotatableBonds','NumSaturatedCarbocycles','NumSaturatedHeterocycles',\
'NumSaturatedRings','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12',\
'PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5',\
'PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','RingCount','SMR_VSA1',\
'SMR_VSA10','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5','SMR_VSA6','SMR_VSA7',\
'SMR_VSA8','SMR_VSA9','SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12',\
'SlogP_VSA2','SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7',\
'SlogP_VSA8','SlogP_VSA9','TPSA','VSA_EState1','VSA_EState10','VSA_EState2',\
'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',\
'VSA_EState8','VSA_EState9']


#### Main ####

if __name__ == "__main__":
    
    infile_list = [test_path, train_path, leaderboard_path]
    
    for file_index, infile in enumerate(infile_list):
        
        print("Processing %s" % infile)
        
        # configure output files
        outfile_path_fp = infile.parent / (infile.stem + '_fp.csv')
        outfile_path_rdkit = infile.parent / (infile.stem + '_rdkit.csv')
        outfile_path_smiles = infile.parent / (infile.stem + '_standard.csv')
        
        outfile_rdkit = open(outfile_path_rdkit, 'w')
        outfile_fp = open(outfile_path_fp, 'w')
        outfile_smiles = open(outfile_path_smiles, 'w')
        
        # initiate the standardizers
        s = Standardizer()
        remover = SaltRemover()
        
        # initiate descriptor calculators
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        calc = MolecularDescriptorCalculator(desc_list)
        
        n = 0 # processed instances
        failed = 0 # failed instances
        
        """
        # Create new headers for the result files
        rdkit_list = " ".join(desc_list)
        rdkit_header = "SMILES ID Value %s\n" % rdkit_list
        outfile_rdkit.write(rdkit_header)
        fp_list = " ".join(["fp%s" % str(x+1) for x in range(2048)])
        fp_header = "SMILES ID Value %s\n" % fp_list
        outfile_fp.write(fp_header)
        """
        smiles_header = "SMILES\tID\tValue\n"
        outfile_smiles.write(smiles_header)
        
        # Read the input file
        with open(infile) as f:
            header = next(f) # remove header
            for count, line in enumerate(f):
                
                line = line.strip()
                
                # Check if processing the train or test file to determine how to handle the y values.
                if file_index == 0:
                    smiles = line
                    value = 'y'
                else:
                    smiles, value = line.split(",")
                ID = count+1
                
                mol = Chem.MolFromSmiles(smiles)
                n+=1 # Counter for processed molecules
                
                if mol is None: # Skip if molecule not workable
                    failed +=1
                    print(ID)
                    continue
                
                if "." in smiles: # Only desalt if there is a salt indicated.
                    mol = remover.StripMol(mol)#, dontRemoveEverything=True) # desalt
                    """
                    if mol.GetNumAtoms() < 2: # remove one atom molecules
                        print("Salt only!")
                        print(line)
                        failed +=1
                        continue
                    """
                    
                try:
                    smol = s.standardize(mol)
                except:
                    print("Standard fail")
                    print(ID, smiles)
                    print(Chem.MolToSmiles(mol))
                    print(line)
                    failed +=1
                    continue
                
                standard_smiles = Chem.MolToSmiles(smol)
                new_line = "%s\t%s\t%s\n" % (standard_smiles, ID, value)
                outfile_smiles.write(new_line)
                
                # Calculate descriptors
                try:
                    # RDDKit Descriptors
                    descrs = calc.CalcDescriptors(smol)
                    descrs = "\t".join([str(x) for x in descrs])
                    rdkit_line = "%s\t%s\t%s\n" % (ID, value, descrs)
                    
                    # Fingerprints
                    #fp = AllChem.GetMorganFingerprintAsBitVect(smol, 2, nBits=2048)
                    fp = mfpgen.GetFingerprint(smol)
                    fp = "\t".join([str(x) for x in fp])
                    fp_line = "%s\t%s\t%s\n" % (ID, value, fp)
                    
                except:
                    failed+=1
                    continue
                else:
                    outfile_fp.write(fp_line)
                    outfile_rdkit.write(rdkit_line)

        print("Processed: ", n)
        print("Failed: ", failed)
        
        # Clean up
        outfile_rdkit.close()
        outfile_fp.close()
        outfile_smiles.close()
