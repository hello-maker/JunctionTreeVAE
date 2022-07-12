import numpy as np
import pickle
import torch
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from module.DNN import *
from module.RNN import *

def load_checkpoint_eval(filepath, USE_CUDA, device):
    # https://tutorials.pytorch.kr/beginner/saving_loading_models.html
    
    if USE_CUDA:
        checkpoint = torch.load(filepath, map_location = device)
    else:
        checkpoint = torch.load(filepath, map_location = device)

    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()    
    return model 
    
def get_regression_result(labels, predictions):

    labels = np.array(labels)
    predictions = np.array(predictions)

    RMSE = mean_squared_error(labels, predictions)**0.5
    PCC = pearsonr(labels, predictions)
    CI = concordance_index(labels, predictions)
    r2 = r2_score(labels, predictions)
    
    return RMSE, PCC[0], CI, r2
    
class test_Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
        
class Mycall:
    def __init__(self, protein_voca, compound_voca, USE_CUDA):
        self.protein_voca = protein_voca
        self.compound_voca = compound_voca
        self.USE_CUDA = USE_CUDA
        
    def __call__(self, batch):
        
        sampling_protein, sampling_smiles = list(), list()
        
        for i in batch:
           
            sampling_protein.append(i[0])
            sampling_smiles.append(i[1])            

        protein_lines = [self.protein_voca.indexesFromSentence(line) for line in sampling_protein]
        protein_lengths = np.array([len(line) for line in protein_lines])
        max_protein_len = np.max(protein_lengths)

        smiles_lines = [self.compound_voca.indexesFromSentence(line) for line in sampling_smiles]
        smiles_lengths = np.array([len(line) for line in smiles_lines])
        max_smiles_len = np.max(smiles_lengths) 

        protein_input = torch.LongTensor([[self.protein_voca.word2index[w] if w in self.protein_voca.word2index else self.protein_voca.word2index["<unk>"] 
            for w in line[:-1]] + [self.protein_voca.word2index["<pad>"]] * (max_protein_len - len(line)) 
                for line in protein_lines]
        )

        compound_input = torch.LongTensor([[self.compound_voca.word2index[w] if w in self.compound_voca.word2index else self.compound_voca.word2index["<unk>"] 
            for w in line[:-1]] + [self.compound_voca.word2index["<pad>"]] * (max_smiles_len - len(line)) 
                for line in smiles_lines]
        )

        # sort protein sequences
        protein_sorted_lengths = torch.LongTensor([torch.max(torch.nonzero(protein_input[i,:])) + 1 for i in range(protein_input.size(0))])
        protein_sorted_lengths, sorted_idx = protein_sorted_lengths.sort(0, descending=True)
        protein_input = protein_input[sorted_idx]

        # for reverse sort
        protein_reverse_sort_dict = dict()
        for idx, val in enumerate(sorted_idx):
            protein_reverse_sort_dict[val] = idx 
        protein_reverse_sort_index = np.array([i[1] for i in sorted(protein_reverse_sort_dict.items())])
    
        # sort SMILES
        compound_sorted_lengths = torch.LongTensor([torch.max(torch.nonzero(compound_input[i,:])) + 1 for i in range(compound_input.size(0))])
        compound_sorted_lengths, sorted_idx = compound_sorted_lengths.sort(0, descending=True)
        compound_input = compound_input[sorted_idx]        

        # for reverse sort
        compound_reverse_sort_dict = dict()
        for idx, val in enumerate(sorted_idx):
            compound_reverse_sort_dict[val] = idx 
        compound_reverse_sort_index = np.array([i[1] for i in sorted(compound_reverse_sort_dict.items())])
        
        return Batch_dnn(protein_input, protein_sorted_lengths.tolist(), self.USE_CUDA, pad_index = 0), \
                Batch_dnn(compound_input, compound_sorted_lengths.tolist(), self.USE_CUDA, pad_index = 0), \
                    protein_reverse_sort_index, compound_reverse_sort_index  