# Learning to Configure Separators in Branch-and-Cut

This directory is the official implementation for our NeurIPS 2023 paper *Learning to Configure Separators in Branch-and-Cut*. This README file provides instructions on environment setup, data collection, restricted configuration space construction, model training and testing.

## Relevant Links
You may find this project at: [Project Website](https://mit-wu-lab.github.io/learning-to-configure-separators/), [arXiv](https://arxiv.org/abs/2311.05650), [OpenReview](https://openreview.net/forum?id=gf5xJVQS5p).
```
@inproceedings{li2023learning,
title={Learning to Configure Separators in Branch-and-Cut},
author={Sirui Li and Wenbin Ouyang and Max B. Paulus and Cathy Wu},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=gf5xJVQS5p}
}
```

## Environment Setup
Our implementation uses python 3.8.13 and Pytorch 1.9.0. The other package dependencies are listed in `requirement.txt` and can be installed with the following command:
```
pip install -r requirements.txt
```
Beside the listed python packages we also use a custom version of the [SCIP](https://www.scipopt.org) solver (v7.0.2) and the [PySCIPOpt](https://github.com/scipopt/PySCIPOpt) interface (v3.3.0), kindly provided by the authors of [1], that allows activating and deactivating separators at intermediate separation rounds during each solve (see Appendix A.6.3 for details). We are informed that the authors of [1] will make the code publicly available soon. The SCIP and PySCIPOpt plug-in follows a similar design to the one found at https://github.com/avrech/learning2cut, which is publicly available.

> [1] Paulus, Max B., et al. "Learning to cut by looking ahead: Cutting plane selection via imitation learning." International conference on machine learning. PMLR, 2022.

## Data Collection
We provide code for data generation in the `./data_generation` folder for Tang et al. [2] and Ecole [3] benchmarks. We also provide download instructions and data filtering code for real-world large-scale benchmarks (NNV, MIPLIB, Load Balancing). The collected data are stored in the `./data` folder.

### Generating instances from Tang et al. and Ecole
One can generate the data simply by:
```
cd data_generation
python generate_tang.py
python generate_ecole.py
```
The default save directory is `./data`, and can be changed by specifying the `--path_to_data_dir` argument of two programs. One can refer to [2] and [3] for changing MILP parameters. We do not filter instances from Tange et al. and Ecole.

- An example Packing instance from Tang et al. can be found in `./example/example_packing_instance.mps`.
- We include the size of the instances in the suffix. For example, if the save directory is `./data`, packing instances will be saved in `./data/packing-60-60`

> [2] Tang, Yunhao, Shipra Agrawal, and Yuri Faenza. "Reinforcement learning for integer programming: Learning to cut." International conference on machine learning. PMLR, 2020.
> [3] Prouvost, Antoine, et al. "Ecole: A gym-like library for machine learning in combinatorial optimization solvers." arXiv preprint arXiv:2011.06069 (2020).

### Downloading instances for the real-world benchmarks
The NN Verification dataset can be downloaded at https://github.com/deepmind/deepmind-research/tree/master/neural_mip_solving.
The MIPLIB dataset can be downloaded at https://miplib.zib.de/download.html.
The Load Balancing dataset can be downloaded at https://github.com/ds4dm/ml4co-competition/blob/main/START.md.

We filter out some of the instances from the real-world benchmarks with the following code (see Appendix A.6.5 for details):
```
cd data_generation
python data_filter.py --instances [MILP class to be filtered] \
                      --path_to_raw_data_dir [directory of the downloaded data] \
                      --path_to_data_dir [save directory] \
                      --time_limit [time limit] \
                      --gap_limit [gap limit] \
                      --index_end [total number of the instances to consider]
```
Specifically, we can run the following commands to filter NNV, MIPLIB, and Load Balancing, assuming the raw data is saved in the `./data/raw/{instances}` directory:
```
cd data_generation
python data_filter.py --instances nnv --path_to_raw_data_dir ../data/raw --path_to_data_dir ../data --time_limit 120 --index_end 1800
python data_filter.py --instances load_balancing --path_to_raw_data_dir ../data/raw --path_to_data_dir ../data --time_limit 300 --gap_limit 0.1 --index_end 1600
python data_filter.py --instances miplib --path_to_raw_data_dir ../data/raw --path_to_data_dir ../data --time_limit 300 --gap_limit 0.1 --index_end 1065
```
The dataset after filtering will be stored in the `./data` folder.


### Splitting datasets into train, validation and test sets
After generating or filtering the problem instances, one can separate the datasets into train, validation and test sets. The default saving directories are `./data/{instance}_train`, `./data/{instance}_val` and `./data/{instance}_test`. And the default ratio for these three sets is 8:1:1. 

For example, after generating 1000 packing instances, copy and paste the first 800 instances from `./data/packing-60-60` to `./data/packing-60-60_train`, 801st to 900th instances to `./data/packing-60-60_val` and the last 100 instances to`./data/packing-60-60_test`.

For the hetergeneous dataset MIPLIB, to ensure the distributions of the train, validation and test sets are roughly the same, a random shuffle of the instances are needed. Assume after filtering, $n$ instances are selected. Then, a randomize permutation from 1 to n should be stored in `./data/miplib/sequence.npy` to indicate the sequence of the instances after shuffling.

----------------------------------------------

## Restricted Space Construction
We provide instructions on constructing the restricted configuration space, following Alg.1 in Appendix A.3. 
### Searching a good instance-agnostic configuration
We first use random search to find a good instance-agnostic configuration, which we later use to construct the "Near Best Random" initial configuration subset (See Appendix A.3).
```
cd restricted_space_generation
python find_best_constant.py --instances [the MILP class] \
                             --path_to_data_dir [directory of the data] \
                             --gap_limit [gap limit]
```
- For example, one can run the following code for the Packing MILP class in Tang et al.: 
```
cd restricted_space_generation
python find_best_constant.py --instances packing-60-60 --path_to_data_dir ../data --gap_limit 0.0
```
The result will be stored as a `best_ins_agnostic.npy` file in the `./restricted_space/{instances}` folder.

### Sampling the large initial configuration subset 
We apply the "Near Zero" and "Near Best Random" strategies as described in  Appendix A.3.1 to construct the large initial configuration subset for later constructing the restricted space.
```
cd restricted_space_generation
python explore_subspace.py --instances [the MILP class] \ 
                           --path_to_data_dir [directory of the data] \
                           --path_to_ins_agnst [directory of the best instance-agnostic configuration] \
                           --mode [the strategy to apply] \
```
- For example, one can run the following code for applying the "Near Zero" strategy to the Packing MILP class in Tang et al.:
```
cd restricted_space_generation
python explore_subspace.py --instances packing-60-60 --path_to_data_dir ../data --path_to_ins_agnst ../restricted_space/ --mode near_zero
```
- and the following code for applying "Near Best Random":
```
cd restricted_space_generation
python explore_subspace.py --instances packing-60-60 --path_to_data_dir ../data --path_to_ins_agnst ../restricted_space/ --mode near_mask
```

The results will be stored in the `./restricted_space/{instances}/{strategy_name}` folder. 

We then run the following code to combine the above sampled configurations into a single configuration subset $S$:
```
cd restricted_space_generation
python get_actions.py --instances [the MILP class] \
                      --path_to_space_dir [directory of the sampled configurations]
```
The code will generate two `.npy` files: `action_space.npy` contains all the configurations in this set $S$; `action_scores.npy` includes the relative time improvement of each configuration in $S$ on each MILP instance in $\mathcal{K}_{small}$.

- For example, one can run the following code to abtain the configuration subset $S$ for Packing MILP class.
```
cd restricted_space_generation
python get_actions.py --instances packing-60-60 --path_to_space_dir ../restricted_space/
```

### Constructing the restricted space $A$
We construct the restricted space $A$ by running `./restricted_space/restricted_space_generation.ipynb` which analyzes `action_space.npy` and `action_scores.npy`. The MILP class can be specified by changing the `instances` variable in the first cell.

By adjusting `our_size` (candidate size of $A$) and `Bs` (candidate threshold $b$) in the $5^{th}$ cell, different plots will be generated from the $6^{th}$ and $7^{th}$ cells (similar to Fig. 6 and 7 in Appendix A.3.2). One should choose $|A|$ and $b$ based on the plots, as described in Appendix A.3.2. 

Finally, we run the last cell to store the restricted space $A$ in `./restricted_space/{instances}/restricted_actions_space.npy`, and the instance-agonstic ERM configuration (one of ours heuristic variants) will be stored to `./restricted_space/{instances}/inst_agnostic_action.npy`.

- Example restricted action space could be found in `./example/restricted_space/{instances}/restricted_actions_space.npy`

## Training
We provide instructions for training the models for $k = 1, 2$ at separation rounds $n_1, n_2$ within the restricted configuration space $A$. 

For training the model $\tilde{f}_{\theta}^{1}$ at $k = 1$ (we set $n_1=0$ across all benchmarks), we run the following:
```
python train_k1.py --instances [the MILP class]
```
The default directory of the restricted action space is `./restricted_space/{instance}/restricted_actions_space.npy`. One can change that by modifying the `--actiondir` and the `--actions_name`. The default directory for saving the model and $Z$ matrix is `./model/{instances}_k1`. One can change the save directory by modifying the `--savedir` argument.

For training the model $\tilde{f}_{\theta}^{2}$ at $k = 2$, we then run:
```
python train_k2.py --instances [the MILP class] \
                   --model_0_path [path to the model of k=1] \
                   --Z_0_path [Matrix Z for k=1] \
                   --step_k2 [the value of n_2]
```
- For example, one can run the following code to train the model  $\tilde{f}_{\theta}^{2}$ at $k = 2$ for the Packing MILP class in Tang et al.:
```
python train_k2.py --instances packing-60-60 --model_0_path ./model/packing-60-60_k1/model-42 --Z_0_path ./model/packing-60-60_k1/Z-42 --step_k2 5
```
The default directory for saving the model and $Z$ matrix is at $k_2$ is `./model/{instances}_k2`. One can change the save directory by modifying the `--savedir` argument.

- Example pretrained models $\tilde{f}_{\theta}^{1}$, $\tilde{f}_{\theta}^{2}$ and the associated $Z^1, Z^2$ matrices can be found at: `./example/model/{instances}/model_k1`, `./example/model/{instances}/model_k2`, `./example/model/{instances}/Z_k1`, and `./example/model/{instances}/Z_k2`. 

## Testing

To test the models $\tilde{f}_{\theta}^{1}$ and $\tilde{f}_{\theta}^{2}$ at $k = 1, 2$, we can run:
```
python test_k2.py --instances [the MILP class] \
                   --model_0_path [path the model of k=1] \
                   --Z_0_path [Matrix Z for k=1] \
                   --model_path [path the model of k=2] \
                   --Z_path [Matrix Z for k=2] \
                   --step_k2 [the value of n_2]
```
- For example, one can run the following code to test the models for the Packing MILP class in Tang et al.:
```
python test_k2.py --instances packing-60-60 --model_0_path ./model/packing-60-60_k1/model-42 --Z_0_path ./model/packing-60-60_k1/Z-42 --model_path ./model/packing-60-60_k2/model-42 --Z_path ./model/packing-60-60_k2/Z-42 --step_k2 5
```
The testing results will be printed to the console.