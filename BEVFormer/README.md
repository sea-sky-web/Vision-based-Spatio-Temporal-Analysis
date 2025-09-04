# Vision-based Spatio-Temporal Analysis using BEVFormer

This project adapts the BEVFormer model for spatio-temporal analysis on the Wildtrack dataset. It provides scripts to load the Wildtrack dataset, extract Bird's-Eye-View (BEV) features, and visualize them.

## Directory Structure

-   `BEVFormer/`: Contains the core model code, custom scripts, and configurations.
    -   `custom/`: Custom modules for this project.
        -   `wildtrack_dataset.py`: Custom data loader for the Wildtrack dataset.
        -   `visualize_bev.py`: Utilities for visualizing BEV features.
    -   `projects/configs/`: Model and dataset configuration files.
    -   `validate_wildtrack.py`: Main script to run validation and visualization.
-   `Wildtrack_dataset/`: Should contain the Wildtrack dataset files.
-   `README.md`: This file.

## 1. Environment Setup

It is recommended to use a virtual environment (e.g., conda or venv).

### Prerequisites
-   Python 3.8+
-   PyTorch 1.10.1+
-   CUDA 11.3+

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Vision-based-Satio-Temporal-Analysis
    ```

2.  **Create and activate a conda environment:**
    ```bash
    conda create -n bevformer python=3.8
    conda activate bevformer
    ```

3.  **Install PyTorch and related libraries:**
    ```bash
    pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```

4.  **Install MMCV and MMDetection:**
    *Note: BEVFormer relies on specific versions of these libraries. Please follow the official installation guide if you encounter issues.*
    ```bash
    pip install openmim
    mim install mmcv-full==1.6.0
    mim install mmdet==2.26.0
    mim install mmsegmentation==0.29.1
    ```

5.  **Install other dependencies:**
    ```bash
    cd BEVFormer
    pip install -r requirements.txt
    cd ..
    ```

## 2. Data Preparation

1.  **Download the Wildtrack Dataset**: Obtain the dataset from its official source and place its contents into the `Wildtrack_dataset/` directory.

2.  **Link the Data**: The application expects the data to be in `BEVFormer/data/Wildtrack`. Create a symbolic link to avoid duplicating data.

    -   **On Windows (Command Prompt as Administrator):**
        ```cmd
        cd BEVFormer
        mklink /D data\Wildtrack ..\Wildtrack_dataset
        cd ..
        ```

    -   **On Linux / macOS:**
        ```bash
        cd BEVFormer
        ln -s ../Wildtrack_dataset data/Wildtrack
        cd ..
        ```

## 3. Pre-trained Model

1.  **Download the BEVFormer Checkpoint**: The validation script requires the pre-trained `bevformer_base_epoch_24.pth` model.
    
    Download it from the official BEVFormer repository release page:
    [https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_base_epoch_24.pth](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_base_epoch_24.pth)

2.  **Place the Model**: Move the downloaded `.pth` file into the `BEVFormer/` directory.

## 4. Usage

The main script for running the validation and visualization is `validate_wildtrack.py`.

-   **Navigate to the `BEVFormer` directory:**
    ```bash
    cd BEVFormer
    ```

-   **Run the script:**
    ```bash
    python validate_wildtrack.py
    ```

This script will:
1.  Load the model and the Wildtrack dataset.
2.  Process the first frame of the dataset.
3.  Extract the BEV feature map.
4.  Save the visualization as `test_bev.png` in the `BEVFormer` directory.

## 5. How it Works

The core logic resides in `validate_wildtrack.py`. It uses a custom configuration (`projects/configs/bevformer/custom_wildtrack.py`) to initialize the BEVFormer model. The `custom/wildtrack_dataset.py` script handles the specific loading and pre-processing requirements of the Wildtrack dataset. Finally, `custom/visualize_bev.py` generates a heatmap from the resulting BEV tensor.