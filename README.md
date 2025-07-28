# SSMAE: Semi-Supervised Masked Autoencoder for Image Classification

This repository implements a Semi-Supervised Masked Autoencoder (SSMAE) for image classification, combining masked autoencoding with pseudo-labeling and supervised learning. The code is designed for research and experimentation on datasets such as CIFAR-10 and ImageNet.

## Features

- **Masked Autoencoder (MAE) backbone** for self-supervised representation learning.
- **Semi-supervised training** using both labeled and unlabeled data.
- **Pseudo-labeling** with confidence thresholding and strong/weak augmentations.
- **Flexible model sizes**: base, large, huge.
- **Finetuning and encoder freezing** for downstream classification tasks.
- **PyTorch implementation** with modular code for easy extension.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/atik666/SSMAE.git
   cd SSMAE/ssmae
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision tqdm
   ```

   Additional dependencies may be required for your environment.

## Dataset Preparation

- Organize your data as follows:
  ```
  data/
    cifar10/
      train/
        labeled_10/      # Labeled training images (10% per class)
        unlabeled/       # Unlabeled training images
      test/              # Test images
  ```
- You can use [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) or your own dataset in ImageFolder format.

## Usage

### Pretraining (Semi-Supervised MAE)

```bash
python pretrain.py
```
- Trains the SSMAE model using both labeled and unlabeled data.
- Pseudo-labeling is enabled after a warmup period based on high-confidence accuracy.

### Finetuning

```bash
python finetune.py
```
- Finetunes the pretrained MAE model for classification.
- Supports full finetuning and encoder freezing.

### Main Functions

- `create_mae_model(model_size, num_classes)`: Create MAE model of specified size.
- `train_SSMAE_w_unlabeled(...)`: Semi-supervised training loop.
- `finetune_model(...)`: Finetune for classification.
- `freeze_encoder_finetune(...)`: Finetune only classifier head.

## Model Checkpoints

- Pretrained and finetuned models are saved in the `models/` directory.
- You can resume training or finetuning from saved checkpoints.


