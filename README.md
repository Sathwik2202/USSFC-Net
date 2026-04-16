## USSFCNet for Remote Sensing Change Detection

This repository implements **USSFCNet**, a U-Net–style Siamese network with **Multi-Scale Dilated Convolution + Self-Statistics Feature Calibration (MSDConv-SSFC)** blocks for binary change detection between two remote sensing images.

The code is configured for the **LEVIR-CD** dataset (building change detection) but can be adapted to other datasets with the same two-time-step input format.

---

### Project Structure

- **`train.py`**: Entry point for training and validation.
- **`paths.py`**: Dataset root and split paths (currently configured for LEVIR-CD).
- **`dataset.py`**: Dataset loader that implements `RsDataset` class for loading paired remote sensing images (time t1 and t2) with corresponding change detection labels. Handles data augmentation and preprocessing for training and validation.
- **`networks/USSFCNet.py`**: Main network definition (USSFCNet, First_DoubleConv, DoubleConv).
- **`networks/modules/CMConv.py`**: Channel-masked convolution (CMConv) used to build multi-scale features.
- **`networks/modules/MSDConv_SSFC.py`**: Multi-Scale Dilated Conv + SSFC integration block.
- **`networks/modules/SSFC.py`**: Spatial-Spectral Feature Cooperation attention module.
- **`metrics.py`**: Confusion matrix and metric computation (precision, recall, F1, IoU).
- **`utils.py`**: Training and validation epoch loops (`train_epoch`, `val_epoch`).

---

### Installation

#### Prerequisites
- Python 3.9 or higher
- pip or conda package manager
- CUDA 11.8+ (optional, for GPU acceleration)

#### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/Sathwik2202/USSFC-Net.git
cd USSFC-Net
```

2. **Create a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

#### Key Dependencies
- **PyTorch & TorchVision**: Deep learning framework
- **OpenCV**: Computer vision and image processing
- **Albumentations**: Data augmentation
- **scikit-learn/image**: ML utilities and image processing
- **TensorBoard**: Training visualization
- **pandas, matplotlib, seaborn**: Data analysis and visualization

#### GPU Support
For GPU acceleration, ensure CUDA is installed and run:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Data Setup (LEVIR-CD)

In `paths.py` the LEVIR-CD dataset root is assumed to be:

```python
dataset_levircd = '/data/LEVIR_CD'
```

with the following directory layout:

- `LEVIR_CD/train/A` – time \(t_1\) images
- `LEVIR_CD/train/B` – time \(t_2\) images
- `LEVIR_CD/train/label` – binary change masks
- `LEVIR_CD/val/A`, `LEVIR_CD/val/B`, `LEVIR_CD/val/label` – validation split
- `LEVIR_CD/test/A`, `LEVIR_CD/test/B`, `LEVIR_CD/test/label` – test split

Adjust `dataset_levircd` in `paths.py` if your dataset is stored elsewhere.

---

### Running Training

1. **Ensure your dataset is properly configured** in `paths.py`

2. **Start training:**
```bash
python train.py
```

#### Training Configuration

The training script (`train.py`) includes:

- **Model**: USSFCNet(3, 1, ratio=0.5) with GPU/CPU auto-detection
- **Loss Function**: Binary Cross-Entropy (BCELoss)
- **Optimizer**: Adam with lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-4
- **Data Transforms**: ToTensor + normalization (mean=0.5, std=0.5)
- **Best Model Saving**: Automatically saves best weights to `ckps/best_model.pth` based on validation F1-score

#### Customizable Parameters

Edit `train.py` to modify:
- Learning rate and optimizer settings
- Batch size and number of epochs
- Model `ratio` parameter to control channel width
- Augmentation settings in `dataset.py`

---

### Dataset Loader

The `dataset.py` file implements the `RsDataset` class for loading paired remote sensing images (time t1 and t2) with corresponding change detection labels. It handles data augmentation and preprocessing for training and validation.


