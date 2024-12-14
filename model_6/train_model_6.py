import pandas as pd
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model_6 import AlexNetConvNet
from torchvision.utils import make_grid
from torch import nn, optim
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False  

Plot_Saving_Directory = 'plots_model_6'

def to_tensor(df):
    return torch.from_numpy(df.to_numpy())

def to_dataloader(x, y, batch_size=64, random_seed=42):
    ts = TensorDataset(x, y)
    return DataLoader(ts, batch_size=batch_size, shuffle=True, worker_init_fn=np.random.seed(random_seed))

# Instantiate the model
model = AlexNetConvNet()
print("Architecture of the Model:", model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load data
train_df = pd.read_csv('../dataset/sign_mnist_train.csv')
test_df = pd.read_csv('../dataset/sign_mnist_test.csv')

train_labels = to_tensor(train_df['label'])
train_images = to_tensor(train_df[train_df.columns[1:]])
test_labels = to_tensor(test_df['label'])
test_images = to_tensor(test_df[test_df.columns[1:]])

train_dataloader = to_dataloader(train_images, train_labels)
test_dataloader = to_dataloader(test_images, test_labels)

# Visualize some training images
dataiter = iter(train_dataloader)
images, labels = next(dataiter)
images = images.reshape(64, 1, 28, 28)
images = images.repeat(1, 3, 1, 1)  # Convert to 3-channel

grid = make_grid(images, nrow=16)
grid = grid.numpy()
fig, ax = plt.subplots(dpi=300)
ax.imshow(np.transpose(grid, (1, 2, 0)))
ax.set_xticks([])
ax.set_yticks([])

# fig.savefig('plots/data-grid.pdf', dpi=300)

@torch.no_grad()
def get_accuracy(net, loader, device):
    total = 0
    correct = 0
    for data in loader:
        images, labels = data
        images = images.reshape(-1, 1, 28, 28).float()
        images = images.repeat(1, 3, 1, 1)  # Convert to 3-channel
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

def train_net(net, trainloader, testloader, epochs=250, 
              criterion=nn.CrossEntropyLoss(), lr=0.001, patience=5, max_grad_norm=1.0):
    
    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True)
    
    training_loss = []
    training_acc = []
    testing_acc = []
    epoch_list = list(range(1, epochs + 1))
    
    best_test_acc = 0.0
    epochs_since_improvement = 0 
    epoch_count = 0
    for epoch in epoch_list:
        epoch_count += 1
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.reshape(-1, 1, 28, 28).float()
            inputs = inputs.repeat(1, 3, 1, 1)  # Convert to 3-channel
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

            optimizer.step()
            running_loss += loss.item()

        # Calculate training and testing accuracies
        train_accuracy = get_accuracy(net, trainloader, device)
        test_accuracy = get_accuracy(net, testloader, device)
        
        # Update training statistics
        training_loss.append(running_loss / i)
        training_acc.append(train_accuracy)
        testing_acc.append(test_accuracy)
        
        # Print stats
        print(f"Epoch {epoch}: loss: {training_loss[-1]:.2f} | train_acc: {train_accuracy:.2f} | test_acc: {test_accuracy:.2f}")
        
        # Early Stopping Check
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            epochs_since_improvement = 0  # Reset counter for early stopping
        else:
            epochs_since_improvement += 1
        
        # If no improvement for 'patience' epochs, stop training
        if epochs_since_improvement >= patience:
            print(f"Early stopping after {epoch} epochs due to no improvement.")
            break
        
        # Adaptive learning rate adjustment based on validation performance
        scheduler.step(test_accuracy)

    print("\nTraining complete")
    return dict(net=net, 
                epochs=list(range(1, epoch_count + 1)), 
                training_loss=training_loss, 
                training_acc=training_acc, 
                testing_acc=testing_acc)

# Define saving directory for plots
Plot_Saving_Directory = "./plots"
if not os.path.exists(Plot_Saving_Directory):
    os.makedirs(Plot_Saving_Directory)
    print(f"Directory '{Plot_Saving_Directory}' created.")
else:
    print(f"Directory '{Plot_Saving_Directory}' already exists.")

# Train the model
model = AlexNetConvNet()
outputs = train_net(model, train_dataloader, test_dataloader)
torch.save(model.state_dict(), 'model_alexnet.weights')
print(outputs['training_loss'])
print(outputs['epochs'])

# Visualize results
sns.set(style='whitegrid')

# Loss Plot
fig, ax = plt.subplots(dpi=300)
sns.lineplot(x=outputs['epochs'], y=outputs['training_loss'], ax=ax, marker='o', color='b')
ax.set_xlabel('Epochs')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss over Epochs')
fig.tight_layout()
fig.savefig(Plot_Saving_Directory+'/loss.pdf', dpi=300)

# Training Accuracy Plot
fig, ax = plt.subplots(dpi=300)
sns.lineplot(x=outputs['epochs'], y=outputs['training_acc'], ax=ax, marker='o', color='b')
ax.set_xlabel('Epochs')
ax.set_ylabel('Training Accuracy')
ax.set_title('Training Accuracy over Epochs')
fig.tight_layout()
fig.savefig(Plot_Saving_Directory+'/train-acc.pdf', dpi=300)

# Testing Accuracy Plot
fig, ax = plt.subplots(dpi=300)
sns.lineplot(x=outputs['epochs'], y=outputs['testing_acc'], ax=ax, marker='o', color='b')
ax.set_xlabel('Epochs')
ax.set_ylabel('Testing Accuracy')
ax.set_title('Testing Accuracy over Epochs')
fig.tight_layout()
fig.savefig(Plot_Saving_Directory+'/test-acc.pdf', dpi=300)