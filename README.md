# Image Classifier for Flower Species

A deep learning project that trains an image classifier to recognize different species of flowers using PyTorch and transfer learning. The project includes command-line applications for training a model and making predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Model Architectures](#model-architectures)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)

## Project Overview

This project implements an image classification pipeline using transfer learning with pre-trained convolutional neural networks. The classifier can identify 102 different flower species with high accuracy. The application consists of two main scripts:

- `train.py` - Train a new network on a dataset of images
- `predict.py` - Use a trained network to predict the class of an input image

## Features

- **Multiple Architecture Support**: Choose from DenseNet121, VGG16, or ResNet50
- **Customizable Hyperparameters**: Set learning rate, hidden units, epochs, and dropout
- **GPU Support**: Optional GPU acceleration for training and inference
- **Transfer Learning**: Leverages pre-trained ImageNet weights
- **Top-K Predictions**: Get multiple class predictions with probabilities
- **Human-Readable Labels**: Map class indices to actual flower names
- **Checkpoint System**: Save and load trained models

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- NumPy
- Pillow
- Matplotlib (for visualization in Jupyter notebook)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flower-classifier.git
cd flower-classifier
```

2. Install required packages:
```bash
pip install torch torchvision numpy pillow matplotlib
```

3. Download the flower dataset and organize it in the following structure:
```
flowers/
├── train/
├── valid/
└── test/
```

## Dataset

This project uses a dataset of 102 flower categories. Each category contains images of flowers belonging to that species. The dataset should be organized into training, validation, and test sets.

You can download the dataset from [Oxford 102 Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) or use a similar flower dataset.

## Usage

### Training

Train a new network on your dataset:

**Basic training with default settings:**
```bash
python train.py flowers
```

**Training with custom hyperparameters:**
```bash
python train.py flowers --arch vgg16 --learning_rate 0.001 --hidden_units 1024 --epochs 10 --gpu
```

**Available options:**
- `data_dir`: Path to dataset directory (required)
- `--save_dir`: Directory to save checkpoint (default: current directory)
- `--arch`: Model architecture - choices: `densenet121`, `vgg16`, `resnet50` (default: densenet121)
- `--learning_rate`: Learning rate (default: 0.003)
- `--hidden_units`: Number of hidden units in classifier (default: 512)
- `--epochs`: Number of training epochs (default: 5)
- `--dropout`: Dropout probability (default: 0.3)
- `--gpu`: Use GPU for training

**Example output:**
```
==================================================
TRAINING CONFIGURATION
==================================================
Data directory: flowers
Architecture: densenet121
Hidden units: 512
Learning rate: 0.003
Epochs: 5
==================================================

Training on cuda
==================================================
Epoch 1/5.. Train loss: 3.524.. Validation loss: 1.823.. Validation accuracy: 0.534
Epoch 2/5.. Train loss: 1.654.. Validation loss: 1.124.. Validation accuracy: 0.712
Epoch 3/5.. Train loss: 1.234.. Validation loss: 0.876.. Validation accuracy: 0.784
Epoch 4/5.. Train loss: 1.045.. Validation loss: 0.723.. Validation accuracy: 0.823
Epoch 5/5.. Train loss: 0.923.. Validation loss: 0.645.. Validation accuracy: 0.851
==================================================
Training completed!
```

### Prediction

Use a trained model to predict flower species:

**Basic prediction:**
```bash
python predict.py flowers/test/1/image_06743.jpg
```

**Prediction with all options:**
```bash
python predict.py flowers/test/1/image_06743.jpg --checkpoint checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

**Available options:**
- `image_path`: Path to input image (required)
- `--checkpoint`: Path to checkpoint file (default: checkpoint.pth)
- `--top_k`: Return top K most likely classes (default: 5)
- `--category_names`: Path to JSON file mapping categories to flower names
- `--gpu`: Use GPU for inference

**Example output:**
```
==================================================
PREDICTION RESULTS
==================================================

Top 5 predictions:
--------------------------------------------------
1. pink primrose (Class 1): 98.45%
2. hard-leaved pocket orchid (Class 10): 1.23%
3. canterbury bells (Class 3): 0.18%
4. sweet pea (Class 45): 0.08%
5. english marigold (Class 62): 0.06%
--------------------------------------------------

Most likely class: pink primrose
Probability: 98.45%
==================================================
```

## Model Architectures

The project supports three pre-trained architectures:

### DenseNet121 (Default)
- **Parameters**: ~7M (trainable)
- **Advantages**: Efficient feature reuse, good performance
- **Best for**: Balanced performance and speed

### VGG16
- **Parameters**: ~138M (total)
- **Advantages**: Simple architecture, well-understood
- **Best for**: High accuracy when computational resources are available

### ResNet50
- **Parameters**: ~23M (trainable)
- **Advantages**: Deep architecture with residual connections
- **Best for**: Complex pattern recognition

All architectures use:
- Custom fully-connected classifier
- ReLU activation
- Dropout for regularization
- LogSoftmax output layer
- Negative Log Likelihood Loss

## Project Structure
```
flower-classifier/
├── train.py                 # Training script
├── predict.py              # Prediction script
├── checkpoint.pth          # Saved model checkpoint
├── cat_to_name.json        # Category to name mapping
├── flowers/                # Dataset directory
│   ├── train/
│   ├── valid/
│   └── test/
├── Image Classifier Project.ipynb  # Jupyter notebook for development
└── README.md
```

## Results

The trained model achieves the following performance metrics:

- **Training Loss**: ~0.9 (after 5 epochs)
- **Validation Accuracy**: ~85% (after 5 epochs)
- **Test Accuracy**: ~85%

Performance can be improved by:
- Training for more epochs
- Using data augmentation
- Fine-tuning hyperparameters
- Using ensemble methods

## Implementation Details

### Image Preprocessing
- Resize shortest side to 256 pixels
- Center crop to 224x224
- Normalize with ImageNet mean and std
- Convert to PyTorch tensor

### Data Augmentation (Training only)
- Random rotation (±30°)
- Random resized crop
- Random horizontal flip

### Transfer Learning Approach
1. Load pre-trained model (ImageNet weights)
2. Freeze convolutional layers
3. Replace classifier with custom architecture
4. Train only classifier weights
5. Fine-tune on flower dataset

## Category Names

The `cat_to_name.json` file maps numerical class labels to flower species names. Example:
```json
{
    "1": "pink primrose",
    "2": "hard-leaved pocket orchid",
    "3": "canterbury bells",
    ...
}
```



## Acknowledgments

- Dataset: [Oxford 102 Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- Pre-trained models: PyTorch's torchvision.models
- Framework: PyTorch
