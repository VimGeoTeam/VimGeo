# VimGeo: Efficient Cross-View Geo-Localization with Vision Mamba Architecture
[[🔗 Project](https://github.com/VimGeoTeam/VimGeo)] [[📘 Paper (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/133)]

**It has been accepted by IJCAI-25**

This is a PyTorch implementation of the “VimGeo: Efficient Cross-View Geo-Localization with Vision Mamba Architecture”.


<div style="text-align:center;">
    <image src="figure/figure2.jpg" style="width:100%;" style="height:100%;"/>
    <p>
        <strong>
                (a) Architecture of the proposed VimGeo model. (b) Visualization of the Channel Group Pooling (CGP) module.
        </strong>
    </p>
</div>

# Environment Setup for Pretraining

### For NVIDIA GPUs:

1. **Python Environment**:
   - Use Python 3.10.13:
     ```bash
     conda create -n your_env_name python=3.10.13
     ```

2. **PyTorch Installation**:
   - Install PyTorch 2.1.1 with CUDA 11.8:
     ```bash
     pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
     ```

### General Requirements:

1. **Install Required Packages**:
   - First, clone the following repository:
     ```bash
     git clone https://github.com/hustvl/Vim.git
     ```
   - Install dependencies:
     ```bash
     pip install -r vim/vim_requirements.txt
     ```

2. **Special Package Installation**:
   - **causal_conv1d**:
     - Choose the appropriate version based on your system, then install:
       ```bash
       # Example command, replace with the correct version if needed
       pip install -e causal_conv1d>=1.1.0
       ```

   - **mamba**:
     - Ensure system compatibility and note that this library is modified in the Vim project.
     - Installation steps:
       ```bash
       # Example command, ensure system compatibility
       use the `mamba-1p1p1` files already included in our project folder to install
       ```
     - **Important**:  
       The modifications to `mamba-1p1p1` are based on our model's source project.  
       For details, please refer to https://github.com/hustvl/Vim/tree/main
     ```

Please adjust the steps based on your system and project needs, ensuring all paths and version numbers are correct.


# Dataset

Please download [CVUSA](http://mvrl.cs.uky.edu/datasets/cvusa/), [CVACT](https://github.com/Liumouliu/OriCNN) and [VIGOR](https://github.com/Jeff-Zilence/VIGOR). You may need to modify  dataset path in "dataloader".

# Model Zoo

| Dataset          | R@1          | R@5      | R@10     | R@1%    | Hit      |
| ---------------- | ------------ | -------- | -------- | ------- | -------- |
| CVUSA            | 96.19%   | 98.62%   | 99.00%   | 99.52%  | -        |
| CVACT_val        | 87.62%   | 94.88%   | 96.06%   | 98.06%  | -        |
| CVACT_test       | 81.69%   | 92.42% | 94.32% | 97.19%  | -        |
| VIGOR Same-Area  | 55.24%       | 80.75%   | 76.12%   | 97.30%  | 57.43%   |
| VIGOR Cross-Area | 19.31%   | 37.50%   | 46.03%   | 86.96%  | 20.72%   |

### Note
All related results are available at [Hugging Face](https://huggingface.co/HuangJavelin/VimGeo/tree/main).

# Usage

## Training
To train our models on the respective datasets, simply run the following scripts:
1. For CVUSA:
    ```bash
    bash train_CVUSA.sh
    ```


2. For CVACT (validation set):

    ```bash
    bash train_CVACT.sh
    ```

3. For CVACT (test set):

    ```bash
    bash train_CVACT_test.sh
    ```

4. For VIGOR Same-Area:

    ```bash
    bash train_VIGOR.sh
    ```

5. For VIGOR Cross-Area:

    ```bash
    bash train_VIGOR_cross.sh
    ```

These scripts contain all necessary parameters and configurations to train our method on each dataset for 50 epochs.


## Evaluation

You should organize the downloaded pre-trained models in the following way:

- `./result_cvusa/`
  - `model_best.pth.tar`
  - `checkpoint.pth.tar`
- `./result_cvact/`
  - `model_best.pth.tar`
  - `checkpoint.pth.tar`
- `./result_vigor/`
  - `model_best.pth.tar`
  - `checkpoint.pth.tar`
- `./result_vigor_cross/`
  - `model_best.pth.tar`
  - `checkpoint.pth.tar`




**Note:**
To evaluate the models, simply add the `-e` option to the corresponding training script:
Modify the following files by adding `-e` to the command line in each script:
- `VimGeoTeam/VimGeo/train_VIGOR.sh`
- `VimGeoTeam/VimGeo/train_CVUSA.sh`
- `VimGeoTeam/VimGeo/train_CVACT.sh`
- `VimGeoTeam/VimGeo/train_CVACT_test.sh`
- `VimGeoTeam/VimGeo/train_VIGOR_cross.sh`


#References and Acknowledgements


[FRGeo](https://github.com/zqwlearning/FRGeo-Code)，[Vim](https://github.com/hustvl/Vim)，[TransGeo](https://github.com/Jeff-Zilence/TransGeo2022)，[ConvNeXt](https://github.com/facebookresearch/ConvNeXt)，[CVUSA](http://mvrl.cs.uky.edu/datasets/cvusa/)，[VIGOR](https://github.com/Jeff-Zilence/VIGOR)，[OriCNN](https://github.com/Liumouliu/OriCNN)，[Deit](https://github.com/facebookresearch/deit)，[MoCo](https://github.com/facebookresearch/moco)

# Contact

If you have any questions, please feel free to reach out:  
2112304047@mail2.gdut.edu.cn


# Citation

<!-- ```latex
@inproceedings{zhang2024aligning,
  title={Aligning Geometric Spatial Layout in Cross-View Geo-Localization via Feature Recombination},
  author={Zhang, Qingwang and Zhu, Yingying},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7251--7259},
  year={2024}
}
``` -->