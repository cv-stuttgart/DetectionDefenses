# Detection Defenses: An Empty Promise against Adversarial Patch Attacks on Optical Flow

This repository contains the source code for:

_'Detection Defenses: An Empty Promise against Adversarial Patch Attacks on Optical Flow'_

<details>
<summary> Abstract </summary>

Adversarial patches undermine the reliability of optical flow predictions when placed in arbitrary scene locations.
Therefore, they pose a realistic threat to real-world motion detection and its downstream applications.
Potential remedies are defense strategies that detect and remove adversarial patches, but their influence on the underlying motion prediction has not been investigated.
In this paper, we thoroughly examine the currently available detect-and-remove defenses ILP and LGS for a wide selection of state-of-the-art optical flow methods, and illuminate their side effects on the quality and robustness of the final flow predictions.
In particular, we implement defense-aware attacks to investigate whether current defenses are able to withstand attacks that take the defense mechanism into account.
Our experiments yield two surprising results: Detect-and-remove defenses do not only lower the optical flow quality on benign scenes, in doing so, they also harm the robustness under patch attacks for all tested optical flow methods except FlowNetC.
As currently employed detect-and-remove defenses fail to deliver the promised adversarial robustness for optical flow, they evoke a false sense of security.
</details>

# Initial setup

You may create an variable `PATCHATTACK_HOME` in your `.bashrc` file, which points to the root directory of this repository. This will be used by the scripts to reproducing the paper-results

```shell
export PATCHATTACK_HOME=/path/to/adaptive-patch-attack
```

## Setup virtual environment

```shell
python3 -m venv $PATCHATTACK_HOME/venv/patch-attack
source $PATCHATTACK_HOME/venv/patch-attack/bin/activate
```

## Install required packages

Change into scripts folder and execute the script which installs all required packages via pip. As each package is installed succesively, you can debug errors for specific packages later.

```shell
cd $PATCHATTACK_HOME/scripts
./install_packages.sh
```

The code is tested on torch 1.12.1, torchvision 0.13.1 and cuda 11.3.

### Spatial Correlation Sampler

If the installation of the `spatial-correlation-sampler` works and you have a cuda capable machine, open `helper_functions/config_specs.py` and make sure to set the variable `"correlationSamplerOnlyCPU":` to `False`. This will speed up computations, e.g. when using PWCNet.
When using FlowNetC on a machine with a GPU but with the CPU version of the correlation sampler, you may need manually set the device to 'cpu' in the FlowNetC functions [`correlate`](models/FlowNetC/util.py#L42)

If the spatial-correlation-sampler does not install run the following script to install a cpu-only version:

```shell
cd scripts
bash install_scs_cpu.sh
```

When loading gcc and CUDA versions from modules, you need to make sure the versions are compatible and may adjust `GCC_HOME` and other variables. See more informations in [this issue](https://github.com/ClementPinard/Pytorch-Correlation-extension/issues/1#issuecomment-478052992). One solution presented changes the variables as follows:

<details>
<summary> Click to expand </summary>

```bash
# used to compile .cu and for cudnn
export GCC_HOME=/path/to/gcc # eg /usr/local/gcc-9.4.0
export PATH=$GCC_HOME/bin/:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib:$GCC_HOME/lib64:$GCC_HOME/libexec:$LD_LIBRARY_PATH
export CPLUS_INCLUDE="$GCC_HOME/include:$CPLUS_INCLUDE"
export C_INCLUDE="$GCC_HOME/include:$C_INCLUDE"
export CXX=$GCC_HOME/bin/g++
export CC=$GCC_HOME/bin/gcc # for make
CC=$GCC_HOME/bin/gcc # for cmake

# complime using nvcc with gcc
export EXTRA_NVCCFLAGS="-Xcompiler -std=c++98"

# pip install
pip install spatial-correlation-sampler
```

</details>

## Loading Flow Models

Download the weights for a specific model by changing into the `scripts/` directory and executing the bash script for a specific model:

```shell
cd scripts
./load_[model]_weights.sh
```

Here `[model]` should be replaced by one of the following options:

```shell
[ all | flownetc | flownet2 | flownetcrobust | pwcnet  | spynet | raft | gma | flowformer ]
```

Note: the load_model scripts remove .git files, which are often write-protected and then require an additional confirmation on removal. To automate this process, consider to execute instead

```shell
yes | ./load_[model]_weights.sh
```

### Compiling CUDA Extensions for FlowNet2

Please refer to the [pytorch documentation](https://github.com/NVIDIA/flownet2-pytorch.git) how to compile the channelnorm, correlation and resample2d extensions.
If all else fails, go to the extension folders `/models/FlowNet/{channelnorm,correlation,resample2d}_package`, manually execute

```bash
python3 setup.py install
```

and potentially replace `cxx_args = ['-std=c++11']` by `cxx_args = ['-std=c++14']`, and the list of `nvcc_args` by `nvcc_args = []` in every setup.py file.
If manually compiling worked, you may need to add the paths to the respective .egg files in the `{channelnorm,correlation,resample2d}.py files`, e.g. for channelnorm via

```python
sys.path.append("/lib/pythonX.X/site-packages/channelnorm_cuda-0.0.0-py3.6-linux-x86_64.egg")
import channelnorm_cuda
```

The `site-packages` folder location varies depending on your operation system and python version.

## Datasets

For training and evaluation of the patches we use the [_KITTI Raw_](http://www.cvlibs.net/datasets/kitti/raw_data.php) and [_KITTI 2015_](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) datasets. But the code also supports the [_Sintel_](http://sintel.is.tue.mpg.de/) dataset.
For the KITTI raw dataset, we prepare the data analogously to [Ranjan et al.](https://github.com/anuragranj/flowattack). Download the raw data using the official [KITTI Raw](http://www.cvlibs.net/datasets/kitti/raw_data.php) [download script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) and prepare the data with the python script of [Ranjan et al.](https://github.com/anuragranj/flowattack)

```shell
./raw_data_downloader.sh
python3 scripts/prepare_kittiraw_data.py /path/to/raw/kitti/dataset/ --dataset-format 'kitti' --dump-root /path/to/resulting/formatted/data/ --width 1280 --height 384 --num-threads 1 --with-gt
```

The datasets are assumed to be in a similar layout as for training [RAFT](https://github.com/princeton-vl/RAFT#required-data):

```
├── datasets
    ├── KITTIRaw
      ├── 2011_09_26_drive_0001_sync_02
      ├── 2011_09_26_drive_0001_sync_03
      ...
    ├── KITTI
        ├── testing
        ├── training
```

If you have them already saved somewhere else, you may link to the files with

```bash
mkdir datasets
cd datasets
ln -s /path/to/KITTIRaw KITTIRaw
ln -s /path/to/KITTI KITTI
```

or specify the paths and names directly in `helper_functions/config_specs.py`.

# Code Usage

## Training Patches

To train patches with defenses execute

```shell
python3 attack_patch_withDefense.py --net=[FlowNetC,FlowNetCRobust,PWCNet,SpyNet,RAFT,GMA,FlowFormer,FlowNet2,FlowNet2C,FlowNet2S] --defense=[none,lgs,ilp] --loss=[acs_target, acs_lgs, acs_ilp] --save_frequency 10
```

By default, this trains a patch with a diameter of 100 pixels with acs loss to the negative flow target. The patch is trained for 1000 iterations with a batch size of 1.

All available argument options are displayed via

```shell
python3 attack_patch_withDefense.py --help
```

The following arguments are useful to reproduce the paper results:

### Training Arguments

| argument                | Choices                                                         | description                                                  |
| ----------------------- | --------------------------------------------------------------- | ------------------------------------------------------------ |
| `--net`                 | FlowNetC, FlowNetCRobust, PWCNet, SpyNet, RAFT, GMA, FlowFormer | The network for which to execute                             |
| `--patch_size`          | `<size>` eg. 100                                                | The patch size                                               |
| `--optimizer`           | ifgsm, sgd                                                      | The optimizer to use                                         |
| `--lr`                  | `<lr>` eg. 0.1                                                  | The learning rate                                            |
| `--loss`                | acs_target, acs_lgs, acs_ilp                                    | Allows to specify the loss function                          |
| `--target`              | zero, neg_flow                                                  | Allows to set a target.                                      |
| `--n`                   | `<n_steps>` eg. 1000                                            | The number of optimizer steps per image.                     |
| `--change_of_variables` | `True` or `False`                                               | Allows whether to use a change of variables during training. |

### Defense Arguments

| argument    | Choices         | description                                                             |
| ----------- | --------------- | ----------------------------------------------------------------------- |
| `--defense` | none, lgs, ilp  | The defense to use                                                      |
| `--k`       | `<k>` eg. 16    | The blocksize for the defense                                           |
| `--o`       | `<o>` eg. 8     | The overlap for the defense                                             |
| `--t`       | `<t>` eg. 0.175 | The blockwise filtering threshold for the defense                       |
| `--s`       | `<s>` eg. 15.0  | The smoothing/scaling parameter depending on whether LGS or ILP is used |
| `--r`       | `<r>` eg. 5     | The inpainting radius for the ILP defense                               |
| `--alpha`   | `<alpha>`       | Weight of the additional loss term                                      |

### Dataset Arguments

| argument          | Choices                   | description               |
| ----------------- | ------------------------- | ------------------------- |
| `--dataset`       | KittiRaw, Kitti15, Sintel | The dataset to use.       |
| `--dataset_stage` | training, evaluation      | The dataset stage to use. |

### Other Arguments

| argument                   | Choices                           | description                                                                           |
| -------------------------- | --------------------------------- | ------------------------------------------------------------------------------------- |
| `--eval_after`             | `True` or `False`                 | Whether to evaluate the patch on KITTI 2015 after training.                           |
| `--seed`                   | `<seed>` eg. 0                    | The seed to use for reproducibility. If -1 is specified, a random seed is used.       |
| `--custom_weight_path`     | `<path>` eg. /path/to/weights.pth | Allows to specify a custom weight path for the network.                               |
| `--custom_experiment_name` | `<name>`                          | Custom mlflow experiment name. The string is appended to the default experiment name. |
| `--save_frequency`         | `<freq>` eg. 100                  | The frequency with which to save Patch, Images, Perturbed images and Flows            |

## Evaluating Existing Patches

To evaluate patches, you can find the patches of the paper in the `patches` folder. To evaluate them, you can use the following command:

```shell
python3 evaluate_patch_withDefense.py --patch_name /path/to/patch.png --net=[FlowNetC,FlowNetCRobust,PWCNet,SpyNet,RAFT,GMA,FlowFormer] --defense=[none,lgs,ilp] --dataset=Kitti15 --dataset_stage=training
```

For example:

```shell
python3 evaluate_patch_withDefense.py --patch_name patches/FlowNetC_ilp.png --net=FlowNetC --defense=ilp --dataset=Kitti15 --dataset_stage=training --n 200
```

The metrics in mlflow are named as follows: `<metric>_avg_<from>_<to>`:

- metric: `acs`, `aee`, `mse`

- `from`/`to`:
  | `gt` | `undef` | `def` | `adv` | `advdef`|
  | ---- | ------- | ----- | ----- | ------- |
  | $f*$ | $f$     | $f_D$ | $f^A$ | $f^A_D$ |

This means, the defended robustness evaluated in the paper is denoted as `acs_avg_def_advdef`.

## Reproducing the Paper Results

To reproduce the paper results, you can use the following scripts in the `training_scripts` folder:

1. [`optimized_parameters.sh`](./training_scripts/optimized_parameters.sh)

- train all networks with all defenses and the adaptive parameters from the paper
- The seed is not set in this skript, but in the next one
- When running as a slurm script, make sure to create a folder `logs` for the slurm logs

2. [`AnyNet.sh`](./training_scripts/AnyNet.sh)

- Train a specific network with all possible parameter combinations
- configure `net`, `weights` and defense parameters in the script
- slurm script that executes the training script with different seeds in parallel
- Creates one mlflow experiment per defense and network
- The patches are evaluated on KITTI 2015 training set after training and the metrics saved in the mlflow experiment.

3. [`get_best_performing_run.py`](./training_scripts/get_best_performing_run.py)

- Copies all patches from `AnyNet.sh` to a folder `results/<net>\_<defense>\_evals/
- In this folder are:
  - `best-patches`: All patches where the parameter combination performs better than the others on average (mean over the seeds) ie in this folder the parameter combinations are the same.
  - `patches`: all patches in the order of their performance
  - a csv and latex table of the parameters sorted by a metric
  - You need to set the experiment ids in the script to your experiment ids

4. [`manual_patch.py`](./training_scripts/manual_patch.py)

- creates a manual patch and saves it in the `results` folder in the same structure as `get_best_performing_run.py`

5. [`eval_patches.sh`](./training_scripts/eval_patches.sh)

- Evaluates the `n` best patches saved in the `results/<net>_<defense>_evals/best-patches` folder. (`n` is set in the script)
- Patches are evaluated on KITTI 2015 training set for all different defenses and saved in a separate mlflow experiment
- Evaluate the manual patch separately

6. [`create_table.py`](./training_scripts/create_table.py)

- creates the robustness table (Tab. 2) from the results of `eval_patches.sh` as well as the distance to the ground truth flow
- This script is the basis for almost all result tables in the paper

7. [`defense_parameter.sh`](./training_scripts/defense_parameter.sh)

- For a fixed patch, starts evaluations for different defense parameters. (blocksize, overlap, threshold, smoothing/scaling parameter and inpainting radius)
- The patch is specified in the script

8. [`eval_defense_params.py`](./training_scripts/eval_defense_params.py)

- Creates the defense parameter plots in the supplementary material from the results of `defense_parameter.sh`

# Data Logging and Progress Tracking

Training progress and output images are tracked with [MLFlow](https://mlflow.org/) in `mlruns/`, and output images and flows are additionally saved in `experiment_data/`.
In `experiment_data/`, the folder structure is `<networkname>_<attacktype>_<perturbationtype>/`, where each subfolder contains different runs of the same network with a specific perturbation type.

To view the mlflow data locally or through Visual Studio Code, navigate to the root folder of this repository, execute

```shell
mlflow ui
```

and follow the link that is displayed. This leads to the web interface of mlflow.

If the data is on a remote host, the below procedure will get the mlflow data displayed.

## Progress tracking with MLFlow (remote server)

Visual Studio Code automatically forwards the mlflow port to the local machine. In the integrated terminal mlflow can be started as locally.

Otherwise, the following steps are required:

Identify the remote's public IP adress via

```shell
curl ifconfig.me
```

then start mlflow on remote machine:

```shell
mlflow server --host 0.0.0.0
```

On your local PC, replace 0.0.0.0 with the public IP and visit the following address in a web-browser:

```shell
http://0.0.0.0:5000
```

# Adding external defenses

To add your own defense, perform the following steps:

1. In [`helper_functions/defenses.py`](helper_functions/defenses.py), add your own class `YourDefense(Defense)` that inherits from the `Defense` class. The class should have a forward function taking `I1`, `I2` and an optional Mask `M and return the defended images.

2. In [`attack_patch_withDefense.py`](attack_patch_withDefense.py#370) line 370, add your defense:

```python
elif args.defense == "your_defense":
  D = YourDefense(your, parameters)
```

3. Make sure, your defense is added in the [`helper_functions/parsing_file`](helper_functions/parsing_file.py#64) line 64 with all additional parameters as arguments.

# Adding External Models

The framework is built such that custom (PyTorch) models can be included. To add an own model, perform the following steps:

1. Create a directory `models/your_model` containing all the required files for the model.
2. Make sure that all import calls are updated to the correct folder. I.e change:

   ```python
   from your_utils import your_functions # old

   # should be changed to:
   from models.your_model.your_utils import your_functions # new
   ```

3. In [`helper_functions/ownutilities.py`](helper_functions/ownutilities.py) modify the following functions:

   - [`import_and_load()`](helper_functions/ownutilities.py#L64): Add the following lines:

     ```python
     elif net == 'your_model':
       # mandatory: import your model i.e:
       from models.your_model import your_model

       # optional: you can outsource the configuration of your model e.g. as a .json file in models/_config/
       with open("models/_config/your_model_config.json") as file:
         config = json.load(file)
       # mandatory: initialize model with your_model and load pretrained weights
       model = your_model(config)
       weights = torch.load(path_weights, map_location=device)
       model = load_state_dict(weights)
     ```

   - [`preprocess_img()`](helper_functions/ownutilities.py#L279): Make sure that the input is adapted to the forward pass of your model.
     The dataloader provides rgb images with range `[0, 255]`. The image dimensions differ with the dataset.
     You can use the padder class make the spatial dimensions divisible by a certain divisor.

     ```python
     elif network == 'your_model':
     # example: normalize rgb range to [0,1]
     images = [(img / 255.) for img in images]
     # example: initialize padder to make spatial dimension divisible by 64
     padder = InputPadder(images[0].shape, divisor=64)
     # example: apply padding
     output = padder.pad(\*images)
     # mandatory: return output
     ```

   - [`model_takes_unit_input()`](helper_functions/ownutilities.py#L392): Add your model to the respective list, if it expects input images in `[0,1]` rather than `[0,255]`.

   - [`compute_flow()`](helper_functions/ownutilities.py#L341): Has to return a tensor `flow` originating from the forward pass of your model with the input images `x1` and `x2`.
     If your model needs further preprocessing like concatenation perform it here:

     ```python
     elif network == 'your_model':
       # optional:
       model_input = torch.cat((x1, x2), dim=0)

     # mandatory: perform forward pass
       flow = model(model_input)
     ```

   - `postprocess_flow()`: Rescale the spatial dimension of the output `flow`, such that they coincide with the original image dimensions. If you used the padder class during preprocessing it will be automatically reused here.

4. Add your model to the possible choices for `--net` in [`helper_functions/parsing_file.py`](helper_functions/parsing_file.py#L17) (i.e. `[... | your_model]`)

# External Models and Dependencies

The code of this repository is based on the code of the [Perturbation Constrained Flow Attack (PCFA)](https://github.com/cv-stuttgart/PCFA) by Schmalfuss et al.

The patch attacks are based on the patch attack by [Ranjan et al.](https://github.com/anuragranj/flowattack).

The defenses are based on their original publications

- [Naseer et al. for LGS](https://arxiv.org/pdf/1807.01216.pdf)
- [Anand et al. for ILP](https://ieeexplore.ieee.org/abstract/document/9356338)

The implementations are significantly influenced by an [(inofficial) implementation](<(https://github.com/metallurk/local_gradients_smoothing)>) of Local gradients smoothing

## Models

- [_RAFT_](https://github.com/princeton-vl/RAFT)
- [_GMA_](https://github.com/zacjiang/GMA.git)
- [_FlowFormer_](https://github.com/drinkingcoder/FlowFormer-Official)
- [_PWC-Net_](https://github.com/NVlabs/PWC-Net.git) and [Flow Attack](https://github.com/anuragranj/flowattack.git)
- [_SpyNet_](https://github.com/sniklaus/pytorch-spynet.git) and [Flow Attack](https://github.com/anuragranj/flowattack.git)
- [_FlowNet_](https://github.com/NVIDIA/flownet2-pytorch.git)
- [_FlowNetCRobust_](https://github.com/lmb-freiburg/understanding_flow_robustness)

## Additional code

- Augmentation and dataset handling (`datasets.py` `frame_utils.py` `InputPadder`) from [RAFT](https://github.com/princeton-vl/RAFT)

- Path configuration (`conifg_specs.py`) inspired by [this post](https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py)

- File parsing (`parsing_file.py`): idea from [this post](https://stackoverflow.com/a/60418265/13810868)
