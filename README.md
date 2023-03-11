# TC_CancerImmunotherapy_Challenge1
Code and solution for Cancer Immunotherapy Data Science Grand Challenge - Challenge 1

Challenge 1 submission - Ai VU HONG (vuhongai)

This submission ranked 5th in the last public leaderboard but unfortunately out of top 10 in the final private leaderboard. In fact, the prediction of the 3 genes Fosb, Mafk, Stat3 are indeed not so bad, but Ets1 is completely unexpected (for everyone) where the unlabeled class account for 99%, I wonder whose solution can predict that.

## 1 Overview
In order to predict the 5-state proportions of different knockout in the test set, I learned the minimal presentation of all 15077 genes (gene embedding - 32-dimension vector) from provided scRNA-seq data. The [scETM model](https://github.com/hui2000ji/scETM) was used to trained and extract gene embedding. Next, multiple fully-connected neural networks was trained mapping 32-D gene embedding vector to 5-D output vector, resulting multiple predictions on 7 held-out test set. These predictions were then filtered and averaged for final submission. 

By running the notebook [step3_prediction.ipynb](./step3_prediction.ipynb), you will able to reproduce the 2 .csv files in the [solution folder](../solution/). 

## 2 Installation
Python version: 3.10.8

Create and activate new virtual environment for the project:
```bash
anaconda3/bin/conda create -n tcells python=3.10.8
source anaconda3/bin/activate tcells
```

Install dependencies

Note: Make sure that torch is already installed before installing scETM

Precisely, notebook step1 only use scETM and torch while step2 and step3 will need tensorflow.

```bash
pip install torch torchvision torchaudio
pip install tensorflow
pip install scETM anndata scanpy pandas numpy
pip install -U scikit-learn
```

## 3 Usage

### Training scETM model for gene and topic embeddings
The code used for training scETM model can be found in [here](./step1_gene_embedding_extraction.ipynb)
A notable modification from original model is that the size of gene embedding reduced from 400 to 32 in this case. The model was trained for 12000 epoches, the checkpoints can be found in [here](./submission/checkpoints/scETM_01_14-12_57_32/model-12000)

Before running the notebook, please make sure that the scRNA-seq data (sc_training.h5ad file) is downloaded in the [data folder](./data/).

Gene embedding of all 15077 genes was precomputed and save in [here](./submission/embedding/gene_embedding_32.npy). However, it can also be done by calling scETM model as a Pytorch model:
```python
import torch
from scETM import scETM

# define scETM model
model = scETM(adata.n_vars, # number of genes/variables
              n_batches=4, 
              trainable_gene_emb_dim=32,
             )

# load pretrained model
model.load_state_dict(torch.load("./submission/checkpoints/scETM_01_14-12_57_32/model-12000"))
model.eval()

# calculate and save the gene embedding vector
model.get_all_embeddings_and_nll(adata)
gene_embedding = np.array(adata.varm['rho'])
np.save(f"./submission/embedding/gene_embedding_{emb_dim}", gene_embedding)
```

### Training multiple perceptron to predict 5-state proportion
The code used for training, evaluating and predicting cell states can be found in [here](./step2_neural_network.ipynb). Briefly, 3 genes ('Rps6', 'Zfp292', 'Prdm1') were left-out as final validation for the model performance. Next, 10-fold cross-validation of the training set of 61 remaining genes was trained to map 32-dimension gene embedding vector to 5-dimension output vector. Each k-fold train/val set generated 96 different models, by tuning different hyperparameters, which can be found in the notebook. In total, 960 models were generated, and ranked by validation MAE [score](./submission/predictions/), its weights can be found in this [folder](./submission/checkpoints/NN/).


### Select best prediction based on similarity
Despite the score on validation set from previous step is quite good (mae_val<0.1), the predictions on 3 held-out test set ('Rps6', 'Zfp292', 'Prdm1') varied a lot from model to model. Therefore, it is neccesary to filter rather directly averaging all generated predictions. I did that by selecting 2 predictions from 2 best models for each k-fold training based on mae_val (with mae_val < 0.1), resulting a list of total 20 predictions. Each of these predictions (5-dimension vector) was calculated mean_squared_error 'distance' with other 19 vectors, counted how many other vectors with mse<4e-3 and selecting the one(s) with highest number of similar predictions. The average of the this selection will be submitted. By running this [script](./step3_prediction.ipynb), you can reproduce the validation_output.csv and test_output.csv in [solution](../solution/) folder.
