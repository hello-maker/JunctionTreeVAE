# Accelerated Training of Junction Tree VAE

## Training
Step 1: Preprocess the data:
```
python fast_molvae/preprocess.py --train data/zinc15/select_train.txt --split 100 --jobs 16  
mkdir moses-processed
mv tensor* data/zinc15/moses-processed
```
This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use small value for `--split` if you are working with smaller datasets.

Step 2: Train VAE model with KL annealing. 
```
mkdir vae_model/
python fast_molvae/vae_train.py --train data/zinc15/moses-processed --vocab /home/csy/work/JunctionTreeVAE/data/zinc15/vocab.txt --save_dir vae_model/
```
Default Options:

`--beta 0` means to set KL regularization weight (beta) initially to be zero.

`--warmup 40000` means that beta will not increase within first 40000 training steps. It is recommended because using large KL regularization (large beta) in the beginning of training is harmful for model performance.

`--step_beta 0.002 --kl_anneal_iter 1000` means beta will increase by 0.002 every 1000 training steps (batch updates). You should observe that the KL will decrease as beta increases.

`--max_beta 1.0 ` sets the maximum value of beta to be 1.0. 

`--save_dir vae_model`: the model will be saved in vae_model/

Please note that this is not necessarily the best annealing strategy. You are welcomed to adjust these parameters.
