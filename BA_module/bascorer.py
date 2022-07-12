import os
import sys

BAMODULEPATH = '/home/csy/work/JTVAE_Re/BA_module'
if BAMODULEPATH not in sys.path: sys.path = [BAMODULEPATH] + sys.path

import pickle
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from module.helpers import Mycall, load_checkpoint_eval
from module.DNN import Test


def normalize_SMILES(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        smi_rdkit = Chem.MolToSmiles(
                        mol,
                        isomericSmiles=False,   # modified because this option allows special tokens (e.g. [125I])
                        kekuleSmiles=False,     # default
                        rootedAtAtom=-1,        # default
                        canonical=True,         # default
                        allBondsExplicit=False, # default
                        allHsExplicit=False     # default
                    )
    except:
        smi_rdkit = None
    return smi_rdkit
    
    
def prepareBReward(filepath_regressor,
                   filepath_protein_voca,
                   filepath_smiles_voca,
                   device=None, use_cuda=False):
    if device is None:
        device = torch.device('cpu')

    ### Loading the pretrained model
    regressor = load_checkpoint_eval(filepath_regressor, use_cuda, device)
    regressor.to(device)

    ### Loading caller for inputs
    with open(filepath_protein_voca, "rb") as f:
        Protein_voca = pickle.load(f)

    with open(filepath_smiles_voca, "rb") as f:
        SMILES_voca = pickle.load(f)

    mc = Mycall(Protein_voca, SMILES_voca, use_cuda)
    return regressor, mc
    
    
def calc_binding_affinity(smiles, aminoseq, regressor, mc):
    pSeq_SMILES_list = []
    pSeq_SMILES_list.append((aminoseq, smiles))
    test_loader = DataLoader(dataset=pSeq_SMILES_list, batch_size=400, collate_fn=mc)

    test_module = Test(regressor, test_loader)
    
    ba_reg = test_module.predict()[0]
    return ba_reg
    
    
class DTA(object):
    def __init__(self, target, use_cuda, device):
        super(DTA, self).__init__()
        
        self.protein_id = target
        if self.protein_id == 'Bcl-2':
            ## P10415
            self.protein_seq = "MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAASRDPVARTSPLQTPAAPGAAAGPALSPVPPVVHLTLRQAGDDFSRRYRRDFAEMSSQLHLTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREMSPLVDNIALWMTEYLNRHLHTWIQDNGGWDAFVELYGPSMRPLFDFSWLSLKTLLSLALVGACITLGAYLGHK"
        elif self.protein_id == 'Bcl-xl':
            ## Q07817
            self.protein_seq = "MSQSNRELVVDFLSYKLSQKGYSWSQFSDVEENRTEAPEGTESEMETPSAINGNPSWHLADSPAVNGATGHSSSLDAREVIPMAAVKQALREAGDEFELRYRRAFSDLTSQLHITPGTAYQSFEQVVNELFRDGVNWGRIVAFFSFGGALCVESVDKEMQVLVSRIAAWMATYLNDHLEPWIQENGGWDTFVELYGNNAAAESRKGQERFNRWFLTGMTVAGVVLLGSLFSRK"
        elif self.protein_id == 'Bcl-w':
            ## Q92843
            self.protein_seq = "MATPASAPDTRALVADFVGYKLRQKGYVCGAGPGEGPAADPLHQAMRAAGDEFETRFRRTFSDLAAQLHVTPGSAQQRFTQVSDELFQGGPNWGRLVAFFVFGAALCAESVNKEMEPLVGQVQEWMVAYLETQLADWIHSSGGWAEFTALYGDGALEEARRLREGNWASVRTVLTGAVALGALVTVGAFFASK"
        else:
            print("[ERROR] Please enter either 'Bcl-2', 'Bcl-xl', or 'Bcl-w'")
            
        ## Initialize Binding Affinity Predictor
        self.predictor = self._get_predictor(use_cuda, device)

        
    def __call__(self, smiles, use_normalization=True):
        rdkit_smiles = normalize_SMILES(smiles) if use_normalization else smiles
        #print('----')
        #print(rdkit_smiles)
        #print(self.protein_seq)
        #print(self.predictor[0])
        #print(self.predictor[1])
        #print('----')
        try:
            ba = calc_binding_affinity(rdkit_smiles, self.protein_seq, self.predictor[0], self.predictor[1])
        except:
            ba = 0
        return ba
        
        
    def _get_predictor(self, use_cuda, device):
        filepath_regressor    = os.path.join(BAMODULEPATH, "model/train_merged.pth")
        filepath_protein_voca = os.path.join(BAMODULEPATH, "model/Sequence_voca.pkl")
        filepath_smiles_voca  = os.path.join(BAMODULEPATH, "model/SMILES_voca.pkl")
        regressor, mc = prepareBReward(filepath_regressor, filepath_protein_voca, filepath_smiles_voca, device=device, use_cuda=use_cuda)
        return (regressor, mc)
        
        
if __name__=='__main__':
    ## GPU check
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(use_cuda, device)
    
    ## Init DTA
    scorer_bcl2  = DTA('Bcl-2', use_cuda, device)
    scorer_bclxl = DTA('Bcl-xl', use_cuda, device)
    scorer_bclw  = DTA('Bcl-w', use_cuda, device)
    
    ## Example
    smi = 'CC1(CCC(=C(C1)CN2CCN(CC2)C3=CC=C(C=C3)C(=O)NS(=O)(=O)C4=CC(=C(C=C4)NC(CCN5CCOCC5)CSC6=CC=CC=C6)S(=O)(=O)C(F)(F)F)C7=CC=C(C=C7)Cl)C'
    print(f'PubChem_canonical: {smi}')
    print(f'RDKit_canonical: {normalize_SMILES(smi)}')
    
    ## Run
    ba_bcl2  = scorer_bcl2(smi)
    ba_bclxl = scorer_bclxl(smi)
    ba_bclw  = scorer_bclw(smi)
    
    ## Results
    print(f'[BCL-2] : {ba_bcl2:.3f}')
    print(f'[BCL-XL] : {ba_bclxl:.3f}')
    print(f'[BCL-W] : {ba_bclw:.3f}')