import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from module.DNN import *
from module.RNN import *

def load_checkpoint_s2s(filepath, USE_CUDA, device):
    
    if USE_CUDA:
        checkpoint = torch.load(filepath, map_location = device)
    else:
        checkpoint = torch.load(filepath, map_location = torch.device('cpu'))
    
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
 
    for parameter in model.parameters():
        parameter.requires_grad = False
 
    encoder = model.encoder
    decoder_bridge = model.decoder.bridge
    embed = model.embed

    model.eval()

    return encoder, decoder_bridge, embed

class SMILES_en(nn.Module):
    def __init__(self, encoder, decoder_bridge, embed):  
        super(SMILES_en, self).__init__()
        self.smiles_encoder = encoder
        self.smiles_decoder_bridge = decoder_bridge
        self.smiles_embed = embed
        
    def forward(self, src, src_mask, src_lengths):
        encoder_hidden, encoder_final = self.smiles_encoder(self.smiles_embed(src), src_mask, src_lengths)
        
        smiles_init = self.smiles_decoder_bridge(encoder_final)
        return smiles_init, encoder_hidden

class clustering_Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

class Mycall:
    def __init__(self, compound_voca, USE_CUDA):
        self.compound_voca = compound_voca
        self.USE_CUDA = USE_CUDA
        
    def __call__(self, batch):
        
        sampling_smiles = [i for i in batch]

        smiles_lines = [self.compound_voca.indexesFromSentence(line) for line in sampling_smiles]
        smiles_lengths = np.array([len(line) for line in smiles_lines])
        max_smiles_len = np.max(smiles_lengths) 

        compound_input = torch.LongTensor([[self.compound_voca.word2index[w] if w in self.compound_voca.word2index else self.compound_voca.word2index["<unk>"] 
            for w in line[:-1]] + [self.compound_voca.word2index["<pad>"]] * (max_smiles_len - len(line)) 
                for line in smiles_lines]
        )

        # sort SMILES
        compound_sorted_lengths = torch.LongTensor([torch.max(torch.nonzero(compound_input[i,:])) + 1 for i in range(compound_input.size(0))])
        compound_sorted_lengths, sorted_idx = compound_sorted_lengths.sort(0, descending=True)
        compound_input = compound_input[sorted_idx]        

        # for reverse sort
        compound_reverse_sort_dict = dict()
        for idx, val in enumerate(sorted_idx):
            compound_reverse_sort_dict[val] = idx 
        compound_reverse_sort_index = np.array([i[1] for i in sorted(compound_reverse_sort_dict.items())])
        
        return Batch_dnn(compound_input, compound_sorted_lengths.tolist(), self.USE_CUDA, pad_index = 0), compound_reverse_sort_index 

class Transforms:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
    
    def get_(self):
        
        with torch.no_grad():
            self.model.eval()
            
            total_test_outputs = list()    
            for idx, (batch, batch_reverse_index) in enumerate(self.test_loader):
                outputs, _ = self.model.forward(batch.src, batch.src_mask, batch.src_lengths)
                
                latent_vector = torch.cat([outputs[0], outputs[1]], dim = 1)
                latent_vector = latent_vector[batch_reverse_index]

                total_test_outputs.extend(latent_vector.detach().cpu().numpy()) 
                
            return np.array(total_test_outputs) 

def SMILES_encoder_load(filepath_voca, filepath_model, USE_CUDA, GPU_NUM):
    if USE_CUDA:
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        print ('Current cuda device ', torch.cuda.current_device())
        
    with open(filepath_voca, "rb") as f:
        SMILES_voca = pickle.load(f)
        
    c_s2s_encoder, c_s2s_bridge, c_s2s_embed = load_checkpoint_s2s(filepath_model, USE_CUDA, device)
    SMILES_encoder = SMILES_en(c_s2s_encoder, c_s2s_bridge, c_s2s_embed )    
    
    return SMILES_voca, SMILES_encoder
    
class ligand_Voca:
    def __init__(self, name):
        self.name = name 
        self.word2index = {"<pad>":0, "<unk>":1, "<sos>":2, "<eos>":3}
        self.word2count = {}
        self.index2word = {0:"<pad>", 1:"<unk>", 2:"<sos>", 3:"<eos>"}
        self.num_words = 4
    
    def make_dict(self, lines):
        for line in lines:
            self.addWord(line)
        
    def addWord(self, sentence):
        for word in self.tokenzied(sentence):
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                self.word2count[word] += 1
            
    def indexesFromSentence(self, sentence):
        return ["<sos>"] + [word for word in sentence] + ["<eos>"]
    
    def tokenzied(self, input):
        return set(input)   

"""
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


class SMILES_en(nn.Module):
    def __init__(self, encoder, decoder_bridge, embed):  
        super(SMILES_en, self).__init__()
        self.smiles_encoder = encoder
        self.smiles_decoder_bridge = decoder_bridge
        self.smiles_embed = embed
        
    def forward(self, src, src_mask, src_lengths):
        encoder_hidden, encoder_final = self.smiles_encoder(self.smiles_embed(src), src_mask, src_lengths)
        
        smiles_init = self.smiles_decoder_bridge(encoder_final)
        return smiles_init, encoder_hidden
"""
               