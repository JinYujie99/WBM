# WBM
PyTorch Implementation for "Wasserstein Barycenter Matching for Graph Size Generalization" (Our code is based on the open source [GOOD, Gui, 2022](https://github.com/divelab/GOOD) library.)



## Usage

1. Installation

Installation for project usages (recommended by GOOD(https://github.com/divelab/GOOD))
```
cd code
pip install -e.
```

Then you need to install the Python Optimal Transport (POT) package
```
pip install POT
```

2. Dataset Preparation

For NCI109, NCI1, PROTEINS and DD, you can download them from https://zenodo.org/record/7109766#.Yy7AGC8w1pQ, which is released by [SizeShiftRef, DavideBuffelli, 2022](https://github.com/DavideBuffelli/SizeShiftReg). Unzip the folder and put it inside the `code/storage/datasets`  folder. Remove the `0/original/processed/` directory in each dataset's root folder to accomodate the new version of PyG.

For GOOD-Motif and GOOD-HIV, you can download them from  https://drive.google.com/file/d/15YRuZG6wI4HF7QgrLI52POKjuObsOyvb/view?usp=sharing and  https://drive.google.com/file/d/1CoOqYCuLObnG5M0D8a2P2NyL61WjbCzo/view?usp=sharing , respectively. Put it inside the `code/storage/datasets`  folder.


3. Train the model

The GOOD libirary provides the CLI `goodtg` to access the main function located at `GOOD.kernel.main:goodtg`. Choose a config file in `configs/GOOD_configs` and run `goodtg` to train a model. For example, you can train a GIN model with the WBM layer on PROTEINS dataset with 3 different runs by run the following instructions:
```
goodtg --config_path GOOD_configs/PROTEINS/size/covariate/WBM.yaml --model_name GIN_WBM --exp_round 1
goodtg --config_path GOOD_configs/PROTEINS/size/covariate/WBM.yaml --model_name GIN_WBM --exp_round 2
goodtg --config_path GOOD_configs/PROTEINS/size/covariate/WBM.yaml --model_name GIN_WBM --exp_round 3
```
To reproduce the results in our paper, it is recommended to use the default hyperparameters in the config files. You can also manually set these hyperparameters by modifying them in CLI. For example, you can train a GCN-WBM model on PROTEINS, with the number of Wasserstein barycenter being 8 and the tradeoff coefficient of WBM loss being 0.01:
```
goodtg --config_path GOOD_configs/PROTEINS/size/covariate/WBM.yaml --model_name GCN_WBM --Katoms 8 --ood_param 0.01 
```
The finally reported results include validation loss, validation metric, test loss and test metric. For GOOD-Motif and GOOD-HIV, there are two kinds of validation sets : in-domain validation and out-of-domain validation. We use the out-of-domain validation set in our experiment so please check the results with regard to the `Out-of-Domain Checkpoint`.

## Requirements
Environments used in our experiments:
* Python 3.8.0
* PyTorch 1.11.0
* PyTorch-Geometric 2.1.0
* Python Optimal Transport 0.8.2


