# train.py
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os

def get_input_args():
    """
    Parse command line arguments for training
    """
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')
    
    # Basic arguments
    parser.add_argument('data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='.', 
                        help='Directory to save checkpoint (default: current directory)')
    parser.add_argument('--arch', type=str, default='densenet121', 
                        choices=['densenet121', 'vgg16', 'resnet50'],
                        help='Model architecture (densenet121, vgg16, or resnet50)')
    
    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.003, 
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, 
                        help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    
    # GPU option
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU for training if available')
    
    return parser.parse_args()

def load_data(data_dir):
    """
    Load and transform training, validation, and test datasets
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    # Define transforms for training set with data augmentation
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Define transforms for validation set (no augmentation)
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Define transforms for test set (no augmentation)
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return train_loader, valid_loader, test_loader, train_data.class_to_idx

def build_model(arch, hidden_units, dropout, num_classes=102):
    """
    Build and return a pretrained model with custom classifier
    
    Args:
        arch: Model architecture name (densenet121, vgg16, or resnet50)
        hidden_units: Number of hidden units in the classifier
        dropout: Dropout probability
        num_classes: Number of output classes (default: 102 for flowers dataset)
    
    Returns:
        model: The constructed model
        input_size: Size of the input features to the classifier
    """
    # Load pretrained model based on architecture choice
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
    
    # Freeze pretrained parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Build custom classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_units, num_classes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    # Replace the pretrained classifier with our custom classifier
    # Different architectures store their classifier in different attributes
    if arch == 'densenet121':
        model.classifier = classifier
    elif arch == 'vgg16':
        model.classifier = classifier
    elif arch == 'resnet50':
        model.fc = classifier
    
    return model, input_size

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    """
    Train the model and print training/validation metrics
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to train on (cuda or cpu)
        epochs: Number of training epochs
    """
    print(f"\nTraining on {device}")
    print("=" * 50)
    
    # Move model to the appropriate device
    model.to(device)
    
    for epoch in range(epochs):
        # -----------------------
        # Training phase
        # -----------------------
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # Move inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            logps = model(inputs)
            loss = criterion(logps, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate training loss
            running_loss += loss.item()
        
        # Calculate average training loss
        train_loss = running_loss / len(train_loader)
        
        # -----------------------
        # Validation phase
        # -----------------------
        model.eval()  # Set model to evaluation mode
        valid_loss = 0.0
        valid_accuracy = 0.0
        
        # Turn off gradients for validation
        with torch.no_grad():
            for inputs, labels in valid_loader:
                # Move inputs and labels to device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()
                
                # Calculate accuracy
                ps = torch.exp(logps)
                top_class = ps.argmax(dim=1)
                equals = top_class == labels
                valid_accuracy += equals.float().mean().item()
        
        # Calculate average validation metrics
        valid_loss /= len(valid_loader)
        valid_accuracy /= len(valid_loader)
        
        # Print training and validation statistics
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {train_loss:.3f}.. "
              f"Validation loss: {valid_loss:.3f}.. "
              f"Validation accuracy: {valid_accuracy:.3f}")
    
    print("=" * 50)
    print("Training completed!")

def save_checkpoint(model, arch, class_to_idx, hidden_units, dropout, 
                    learning_rate, epochs, save_dir):
    """
    Save model checkpoint with all necessary information for loading later
    
    Args:
        model: The trained model
        arch: Model architecture name
        class_to_idx: Dictionary mapping classes to indices
        hidden_units: Number of hidden units in classifier
        dropout: Dropout probability
        learning_rate: Learning rate used for training
        epochs: Number of epochs trained
        save_dir: Directory to save the checkpoint
    """
    # Create save directory if it doesn't exist (only if not current directory)
    if save_dir != '.' and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create checkpoint dictionary with all necessary info
    checkpoint = {
        'arch': arch,
        'class_to_idx': class_to_idx,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'state_dict': model.state_dict()
    }
    
    # Save checkpoint in the specified directory
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")

def main():
    """
    Main function to orchestrate the training process
    """
    # Get command line arguments
    args = get_input_args()
    
    # Print training configuration
    print("\n" + "=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"Architecture: {args.arch}")
    print(f"Hidden units: {args.hidden_units}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Dropout: {args.dropout}")
    print(f"GPU enabled: {args.gpu}")
    print("=" * 50)
    
    # Set device based on GPU availability and user preference
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.gpu and not torch.cuda.is_available():
            print("\nWarning: GPU requested but not available. Using CPU instead.")
    
    # Load and prepare data
    print("\nLoading data...")
    train_loader, valid_loader, test_loader, class_to_idx = load_data(args.data_dir)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(valid_loader.dataset)}")
    
    # Build the model with specified architecture
    print(f"\nBuilding {args.arch} model...")
    model, input_size = build_model(args.arch, args.hidden_units, args.dropout)
    model.class_to_idx = class_to_idx
    
    # Define loss function
    criterion = nn.NLLLoss()
    
    # Define optimizer (only train classifier parameters)
    # Different architectures store classifier in different attributes
    if args.arch == 'resnet50':
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Train the model
    train_model(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)
    
    # Save the trained model checkpoint
    save_checkpoint(model, args.arch, class_to_idx, args.hidden_units, 
                    args.dropout, args.learning_rate, args.epochs, args.save_dir)

if __name__ == '__main__':
    main()