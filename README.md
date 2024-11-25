# Efficient Visual Object Tracking with Temporal Context-Aware Token Learning and Scale Adaptive Token Pruning
Implementation of the paper Efficient Visual Object Tracking with Temporal Context-Aware Token Learning and Scale Adaptive Token Pruning, **ICONIP 2024**.

## Install the environment
```
conda create -n tcsa python=3.8
conda activate tcsa
bash tcsa_env.yaml
```

## Project Paths Setup
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like this:
```
${PROJECT_ROOT}
 -- data
     -- lasot
         |-- airplane
         |-- basketball
         |-- bear
         ...
     -- got10k
         |-- test
         |-- train
         |-- val
     -- coco
         |-- annotations
         |-- images
     -- trackingnet
         |-- TRAIN_0
         |-- TRAIN_1
         ...
         |-- TRAIN_11
         |-- TEST
```

## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under $PROJECT_ROOT$/pretrained_models
```
python tracking/train.py --script tcsa --config vitb_256 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```
Replace --config with the desired model config under experiments/tcsa.

## Evaluation
Put the checkpoint into $PROJECT_ROOT$/output/config_name/... or modify the checkpoint path in testing code.
Change the corresponding values of lib/test/evaluation/local.py to the actual benchmark saving paths

Some testing examples:

- GOT10K-test
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4
python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_got10k_ep100
```

- LaSOT
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
  
- TrackingNet
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
python lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_ep300
```

## Acknowledgments
Our project is developed upon [OSTrack](https://github.com/botaoye/OSTrack). Thanks for their contributions which help us to quickly implement our ideas.


