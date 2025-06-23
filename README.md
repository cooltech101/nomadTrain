# Training Setup for NoMaD

Original resources for the family of GNM, ViNT and NoMaD can be found below. This repository is based on this [original](https://github.com/robodhruv/visualnav-transformer) and was edited to work on a Windows 10 setup.  

[Project Page](https://general-navigation-models.github.io) | [Citing](https://github.com/robodhruv/visualnav-transformer#citing) | [Pre-Trained Models](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)

---

General Navigation Models are general-purpose goal-conditioned visual navigation policies trained on diverse, cross-embodiment training data, and can control many different robots in zero-shot. They can also be efficiently fine-tuned, or adapted, to new robots and downstream tasks. The family of models is described in the following research papers:
1. [GNM: A General Navigation Model to Drive Any Robot](https://sites.google.com/view/drive-any-robot) (_October 2022_, presented at ICRA 2023)
2. [ViNT: A Foundation Model for Visual Navigation](https://general-navigation-models.github.io/vint/index.html) (_June 2023_, presented at CoRL 2023)
3. [NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration](https://general-navigation-models.github.io/nomad/index.html) (_October 2023_)

## Overview
This repository contains code for training the family of models with your own data and pre-trained model checkpoints. The repository follows the organization from [GNM](https://github.com/PrieureDeSion/drive-any-robot).

- `./train/train.py`: training script to train or fine-tune the ViNT model on your custom data.
- `./train/vint_train/models/`: contains model files for GNM, ViNT, and some baselines.
- `./train/process_bags.py`: script to process rosbags (ROS1) into training data.

## Train

This subfolder contains code for processing datasets and training models from your own data.

### Pre-requisites

The codebase had been tested on VS Code 1.100.3 on Windows 10. I used a Nvidia RTX 2080 TI eGPU with CUDA 11.8 for hardware acceleration. I used Python 3.12.9 and ran the training in a miniconda virtual environment. 

### Setup
Run the commands below inside the `vint_release/` (topmost) directory:
1. Set up the conda environment:
    ```bash
    conda env create -f train/train_environment.yml
    ```
2. Source the conda environment:
    ```
    conda activate vint_train
    ```
3. Install the vint_train packages:
    ```bash
    pip install -e train/
    ```
4. Install the `diffusion_policy` package from this [repo](https://github.com/real-stanford/diffusion_policy):
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```


### Data-Wrangling
In the [papers](https://general-navigation-models.github.io), they train on a combination of publicly available and unreleased datasets. Below is a list of publicly available datasets used for training; please contact the respective authors for access to the unreleased data.
- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [TartanDrive](https://github.com/castacks/tartan_drive)
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links)
- [GoStanford2 (Modified)](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing)
- [SACSoN/HuRoN](https://sites.google.com/view/sacson-review/huron-dataset)

They recommend you to download these (and any other datasets you may want to train on) and run the processing steps below.

#### Data Processing 

Some sample scripts are provided to process these datasets, either directly from a rosbag or from a custom format like HDF5s:
1. Run `process_bags.py` with the relevant args, or `process_recon.py` for processing RECON HDF5s. You can also manually add your own dataset by following our structure below (if you are adding a custom dataset, please checkout the [Custom Datasets](#custom-datasets) section).
```bash
python process_bags.py --dataset-name <dataset name> --input-dir <path to input directory> --output-dir <path to output directory>
```
2. Run `data_split.py` on your dataset folder with the relevant args.
```bash
python data_split.py -i <path to input directory> -d <dataset name>
```

After step 1 of data processing, the processed dataset should have the following structure:

```
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_2.jpg
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
    	├── 0.jpg
    	├── 1.jpg
    	├── ...
        ├── T_N.jpg
        └── traj_data.pkl
```  

Each `*.jpg` file contains an forward-facing RGB observation from the robot, and they are temporally labeled. The `traj_data.pkl` file is the odometry data for the trajectory. It’s a pickled dictionary with the keys:
- `"position"`: An np.ndarray [T, 2] of the xy-coordinates of the robot at each image observation.
- `"yaw"`: An np.ndarray [T,] of the yaws of the robot at each image observation.


After step 2 of data processing, the processed data-split should the following structure inside `/visualnav-transformer/train/vint_train/data/data_splits`:

```
├── <dataset_name>
│   ├── train
|   |   └── traj_names.txt
└── └── test
        └── traj_names.txt 
``` 

### Training your General Navigation Models
Run this inside the `vint_release/train` directory:
```bash
set KMP_DUPLICATE_LIB_OK=TRUE
python train.py -c <path_of_train_config_file>
```
The premade config yaml files are in the `train/config` directory. 

#### Custom Config Files
You can use one of the premade yaml files as a starting point and change the values as you need. `config/vint.yaml` is good choice since it has commented arguments. `config/defaults.yaml` contains the default config values (don't directly train with this config file since it does not specify any datasets for training).

#### Collecting and processing custom ROS2 bags
You can collect your own ROS2 bags from your robot embodiment for finetuning NoMaD. Use the ROS2 implementation found [here] (https://github.com/cooltech101/joeyNav). Collect the data in ROS2 format, then convert back to ROS1 format and process it as described above. 

In an Ubuntu environment, record the image and odometry topics in ROS2 format.
``` bash
ros2 bag record <image topic> <odom topic>
```

Install the rosbags package. This package contains tools to convert rosbags between ROS1 and ROS2 formats. 
``` bash
pip install rosbags
```

Convert to ROS1 format.
``` bash
rosbags-convert --src <path to ROS2 bag dir> --dst <name the new ROS1 bag>
```

Process the ROS1 bags using process_bags.py and data_split.py. The results of data_split.py will be found in `/visualnav-transformer/train/vint_train/data/data_splits/<dataset_name>`. Ensure to add the relevant configurations described in the "Custom Datasets" section. 

#### Custom Datasets
It is critical that the average distance between waypoints in the dataset be entered as accurately as possible for optimal training results. To this end, try to maintain consistent linear velocity throughout the recorded trajectories. Upon completing the recordings, use getspacing.py on a few ros2bags to get a good estimate of the average distance between waypoints.
```bash
python getspacing.py <path to ros2bag directory>
```

Make sure your dataset and data-split directory follows the structures provided in the [Data Processing](#data-processing) section. Locate `train/vint_train/data/data_config.yaml` and append the following:
```
<dataset_name>:
    metric_waypoints_distance: <average_distance_in_meters_between_waypoints_in_the_dataset>
```

Locate your training config file and add the following text under the `datasets` argument (feel free to change the values of `end_slack`, `goals_per_obs`, and `negative_mining`):
```
<dataset_name>:
    data_folder: <path_to_the_dataset>
    train: data/data_splits/<dataset_name>/train/ 
    test: data/data_splits/<dataset_name>/test/ 
    end_slack: 0 # how many timesteps to cut off from the end of each trajectory  (in case many trajectories end in collisions)
    goals_per_obs: 1 # how many goals are sampled per observation
    negative_mining: True # negative mining from the ViNG paper (Shah et al.)
```


#### Training your model from a checkpoint
Instead of training from scratch, you can also load an existing checkpoint from the published results.
Add `load_run: <project_name>/<log_run_name>`to your .yaml config file in `vint_release/train/config/`. The `*.pth` of the file you are loading to be saved in this file structure and renamed to “latest”: `vint_release/train/logs/<project_name>/<log_run_name>/latest.pth`. This makes it easy to train from the checkpoint of a previous run since logs are saved this way by default. Note: if you are loading a checkpoint from a previous run, check for the name the run in the `vint_release/train/logs/<project_name>/`, since the code appends a string of the date to each run_name specified in the config yaml file of the run to avoid duplicate run names. 


If you want to use othe original checkpoints, you can download the `*.pth` weights files from [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).


### Loading the model weights

Save the model weights *.pth file in `vint_release/deployment/model_weights` folder.




