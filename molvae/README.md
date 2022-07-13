# Molecule Generation
Suppose the repository is downloaded at `$PREFIX/JunctionTreeVAE` directory. First set up environment variables:
```
export PYTHONPATH=$PREFIX/JunctionTreeVAE
```
Our ZINC dataset is in `icml18-jtnn/data/zinc` (copied from https://github.com/mkusner/grammarVAE). 
We follow the same train/dev/test split as previous work. 

## Deriving Vocabulary 
If you are running our code on a new dataset, you need to compute the vocabulary from your dataset.
To perform tree decomposition over a set of molecules, run
```
python jtnn/mol_tree.py < data/zinc15/select_train.txt > data/zinc15/vocab.txt
```
This gives you the vocabulary of cluster labels over the dataset `select_train.txt`.

## Training
We trained VAE model in two phases:
1. We train our model for three epochs without KL regularization term (So we are essentially training an autoencoder).
Pretrain our model as follows (with hidden state dimension=450, latent code dimension=56, graph message passing depth=3):
```
mkdir pre_model/
CUDA_VISIBLE_DEVICES=0 python molvae/pretrain.py --train data/zinc15/select_train.txt --vocab /home/csy/work/JunctionTreeVAE/data/zinc15/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --save_dir molvae/pre_model
```
PyTorch by default uses all GPUs, setting flag `CUDA_VISIBLE_DEVICES=0` forces PyTorch to use the first GPU (1 for second GPU and so on).

The final model is saved at pre_model/model.2

2. Train out model with KL regularization, with constant regularization weight $beta$. 
We found setting beta > 0.01 greatly damages reconstruction accuracy.
```
mkdir vae_model/
CUDA_VISIBLE_DEVICES=0 python molvae/vaetrain.py --train data/zinc15/select_train.txt --vocab /home/csy/work/JunctionTreeVAE/data/zinc15/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --lr 0.0007 --beta 0.005 --model molvae/pre_model/model --save_dir molvae/vae_model/
```
