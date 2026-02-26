# predict.py
import argparse
import torch
import json
import numpy as np
from PIL import Image
from torchvision import models
from torch import nn
from collections import OrderedDict

def get_input_args():
    """
    Parse command line arguments for prediction
    """
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    
    # Required arguments
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                        help='Path to checkpoint file (default: checkpoint.pth in current directory)')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=5, 
                        help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None,
                        help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference if available')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    """
    Load a checkpoint and rebuild the model
    
    Args:
        filepath: Path to the checkpoint file
    
    Returns:
        model: The loaded model with trained weights
    """
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Get architecture and build base model
    arch = checkpoint['arch']
    
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        input_size = model.fc.in_features
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Rebuild classifier with saved hyperparameters
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    num_classes = len(checkpoint['class_to_idx'])
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_units, num_classes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    # Attach classifier to model based on architecture
    if arch == 'densenet121':
        model.classifier = classifier
    elif arch == 'vgg16':
        model.classifier = classifier
    elif arch == 'resnet50':
        model.fc = classifier
    
    # Load the trained weights
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load class_to_idx mapping
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model
    
    Args:
        image_path: Path to the image file
    
    Returns:
        torch.Tensor: Processed image ready for model input
    """
    # Load image
    pil_image = Image.open(image_path)
    
    # Resize the image where shortest side is 256 pixels, keeping aspect ratio
    width, height = pil_image.size
    if width < height:
        new_width = 256
        new_height = int((256 / width) * height)
    else:
        new_height = 256
        new_width = int((256 / height) * width)
    
    pil_image = pil_image.resize((new_width, new_height))
    
    # Crop out the center 224x224 portion of the image
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Convert to numpy array and normalize pixel values to 0-1
    np_image = np.array(pil_image) / 255.0
    
    # Normalize with ImageNet means and standard deviations
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions: PyTorch expects (color_channels, height, width)
    # PIL/numpy has (height, width, color_channels)
    np_image = np_image.transpose((2, 0, 1))
    
    # Convert to PyTorch tensor
    tensor_image = torch.from_numpy(np_image).float()
    
    return tensor_image

def predict(image_path, model, device, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model
    
    Args:
        image_path: Path to the image file
        model: Trained PyTorch model
        device: Device to run inference on (cuda or cpu)
        topk: Number of top predictions to return
    
    Returns:
        probs: Top K probabilities
        classes: Top K class labels
    """
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Process the image
    image = process_image(image_path)
    
    # Add batch dimension: model expects (batch_size, channels, height, width)
    image = image.unsqueeze(0)
    
    # Move image to the same device as the model
    image = image.to(device)
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Forward pass through the model
        logps = model(image)
        # Convert log probabilities to probabilities
        ps = torch.exp(logps)
        
        # Get the top K probabilities and their indices
        top_probs, top_indices = ps.topk(topk, dim=1)
        
        # Convert tensors to lists (move to CPU first if on GPU)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Invert the class_to_idx dictionary to get idx_to_class mapping
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        
        # Convert indices to actual class labels
        top_classes = [idx_to_class[idx] for idx in top_indices]
    
    return top_probs, top_classes

def load_category_names(json_file):
    """
    Load category names from JSON file
    
    Args:
        json_file: Path to JSON file containing category name mappings
    
    Returns:
        dict: Dictionary mapping category labels to names
    """
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def main():
    """
    Main function to orchestrate the prediction process
    """
    # Get command line arguments
    args = get_input_args()
    
    # Print prediction configuration
    print("\n" + "=" * 50)
    print("PREDICTION CONFIGURATION")
    print("=" * 50)
    print(f"Image path: {args.image_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Top K predictions: {args.top_k}")
    print(f"GPU enabled: {args.gpu}")
    if args.category_names:
        print(f"Category names file: {args.category_names}")
    print("=" * 50)
    
    # Set device based on GPU availability and user preference
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.gpu and not torch.cuda.is_available():
            print("\nWarning: GPU requested but not available. Using CPU instead.")
    
    print(f"\nRunning inference on {device}")
    
    # Load the checkpoint
    print("\nLoading checkpoint...")
    model = load_checkpoint(args.checkpoint)
    
    # Make prediction
    print("Making prediction...")
    probs, classes = predict(args.image_path, model, device, args.top_k)
    
    # Load category names if provided
    if args.category_names:
        cat_to_name = load_category_names(args.category_names)
        class_names = [cat_to_name[c] for c in classes]
    else:
        class_names = classes
    
    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"\nTop {args.top_k} predictions:")
    print("-" * 50)
    
    for i, (prob, class_label, class_name) in enumerate(zip(probs, classes, class_names), 1):
        if args.category_names:
            print(f"{i}. {class_name} (Class {class_label}): {prob*100:.2f}%")
        else:
            print(f"{i}. Class {class_label}: {prob*100:.2f}%")
    
    print("-" * 50)
    print(f"\nMost likely class: {class_names[0]}")
    print(f"Probability: {probs[0]*100:.2f}%")
    print("=" * 50 + "\n")
    
    # Also print in the format requested for programmatic use
    print("\nProbabilities:")
    print(probs)
    print("\nClasses:")
    print(classes)

if __name__ == '__main__':
    main()