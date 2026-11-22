## Visual transformer - Segformer Setting

## 1. Structure

```text
Bosch-DRIE-ML/
├── .gitignore
├── README.md
└── segformer/
    ├── checkpoints/
    ├── images_annotated/
    ├── images_original/
    ├── masks/
    ├── outputs/
    ├── test_images/
    ├── trench_crops/
    ├── extract_trenches.py
    ├── predict.py
    ├── prepare_masks.py
    ├── test_model.py
    ├── train.py
    └── train_round2.py
```

SEM images, annotation images, masks, and generated prediction images are not included in this repository.

## 2. Environment Setup

The following instructions are intended for Windows PowerShell in Visual Studio Code.

### Step 1: Clone the Repository

git clone https://github.com/0324-sam/Bosch-DRIE-ML.git
cd Bosch-DRIE-ML\segformer


### Step 2: Create and Activate a Virtual Environment

Required Python version: **3.11.9** (development environment).

type in the followings in terminal

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1


If PowerShell blocks activation:


Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1


### Step 3: Install PyTorch

For an NVIDIA GPU with a compatible driver, install the CUDA 12.8 build:


python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128


For CPU-only installation, replace `cu128` with `cpu` in the index URL.

### Step 4: Install Other Dependencies

The following versions match the original development environment:


python -m pip install transformers==5.22.2 accelerate==1.15.0 datasets==5.2.0 numpy==2.4.6 opencv-python==5.0.0.93 pillow==13.0.0 matplotlib==3.11.2 pandas==3.1.1 scipy==1.17.1 scikit-learn==1.9.1 safetensors==0.8.0 tqdm==4.70.1 PyYAML==6.0.3


Additional transitive dependencies will be installed automatically where available. The full environment has not yet been independently reproduced on another computer.

## 3. Verify Installation

Check the Python version:


python --version

Check PyTorch and CUDA availability:


python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"


Check the main dependencies:


python -c "import transformers, cv2, numpy, PIL; print('All core imports OK')"


Check package compatibility:


python -m pip check


Expected development environment:

| Component | Version |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.11.0+cu128 |
| CUDA build | 12.8 |
| Transformers | 5.22.2 |
| OpenCV | 5.0.0 |
| NumPy | 2.4.6 |
| Pillow | 13.0.0 |

GPU availability may differ depending on the hardware and installation.

## 4. Dataset Preparation

All image files must be obtained separately and placed in the appropriate local directories.

Create the required directories if they are missing:


"checkpoints","images_annotated","images_original","masks","outputs","test_images","trench_crops" | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }


- `images_original/`: Original SEM images
- `images_annotated/`: Manually annotated SEM images
- `masks/`: Segmentation masks
- `test_images/`: Test images
- `trench_crops/`: Extracted trench regions
- `outputs/`: Generated predictions and results
- `checkpoints/`: Saved trained models

Before running training or inference, verify the file paths and expected image/mask formats in the Python scripts.

## 5. Model Verification

To verify that the pretrained SegFormer-B0 model loads successfully:


python test_model.py


This loads the pretrained SegFormer architecture and checks model initialization. It does not evaluate the fine-tuned trench boundary segmentation checkpoint.

Training and prediction commands should be added after their dataset paths and required inputs are verified.

## 6. Planned Model Integration

Future implementations may include:

- DGCNN
- PointNeXt
- MeshGraphNet
- Additional vFab simulation and preprocessing utilities

Each model may require a separate virtual environment and additional dependencies.

## Notes

- Do not upload SEM images, annotation images, masks, or generated prediction images.
- Do not commit local virtual environments.
- Use the documented environment versions when reproducing results.
- Verify dependencies and dataset paths before running experiments.
