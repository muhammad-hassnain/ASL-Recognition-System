import pandas as pd
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model_1 import ConvNet
from torchvision.utils import make_grid
from torch import nn, optim
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Set the random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # For GPU if available
torch.backends.cudnn.deterministic = True  # Ensures deterministic results for cudnn
torch.backends.cudnn.benchmark = False  # Might affect performance, but ensures reproducibility

Plot_Saving_Directory = 'plots_model_1'

def to_tensor(df):
    return torch.from_numpy(df.to_numpy())

def to_dataloader(x, y):
    ts = TensorDataset(x, y)
    return DataLoader(ts, batch_size=64, shuffle=True, worker_init_fn=np.random.seed(random_seed))  # Ensures DataLoader's shuffling is reproducible

model = ConvNet()
print("Architecture of the MODEL:", model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_df = pd.read_csv('dataset/sign_mnist_train.csv')
test_df = pd.read_csv('dataset/sign_mnist_test.csv')

train_labels = to_tensor(train_df['label'])
train_images = to_tensor(train_df[train_df.columns[1:]])

test_labels = to_tensor(test_df['label'])
test_images = to_tensor(test_df[test_df.columns[1:]])

train_dataloader = to_dataloader(train_images, train_labels)
test_dataloader = to_dataloader(test_images, test_labels)

# get some random training images
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

fig.savefig('plots/data-grid.pdf', dpi=300)

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

# Check if the directory exists, and create it if not
if not os.path.exists(Plot_Saving_Directory):
    os.makedirs(Plot_Saving_Directory)
    print(f"Directory '{Plot_Saving_Directory}' created.")
else:
    print(f"Directory '{Plot_Saving_Directory}' already exists.")

model = ConvNet()
model.load_state_dict(torch.load('model_1.weights'))  # Load the saved model weights
model.to(device)
model.eval()  # Set the model to evaluation mode

# outputs = train_net(model, train_dataloader, test_dataloader)
# torch.save(model.state_dict(), 'model_1.weights')
# print(outputs['training_loss'])
# print(outputs['epochs'])

# # Loss Plot
# sns.set(style='whitegrid')
# fig, ax = plt.subplots(dpi=300)
# sns.lineplot(x=outputs['epochs'], y=outputs['training_loss'], ax=ax, marker='o', color='b')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Training Loss')
# ax.set_title('Training Loss over Epochs')
# fig.tight_layout()
# fig.savefig(Plot_Saving_Directory+'/loss.pdf', dpi=300)

# # Training Accuracy Plot
# sns.set(style='whitegrid')
# fig, ax = plt.subplots(dpi=300)
# sns.lineplot(x=outputs['epochs'], y=outputs['training_acc'], ax=ax, marker='o', color='b')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Training Accuracy')
# ax.set_title('Training Accuracy over Epochs')
# fig.tight_layout()
# fig.savefig(Plot_Saving_Directory+'/train-acc.pdf', dpi=300)

# # Testing Accuracy Plot
# sns.set(style='whitegrid')
# fig, ax = plt.subplots(dpi=300)
# sns.lineplot(x=outputs['epochs'], y=outputs['testing_acc'], ax=ax, marker='o', color='b')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Testing Accuracy')
# ax.set_title('Testing Accuracy over Epochs')
# fig.tight_layout()
# fig.savefig(Plot_Saving_Directory+'/test-acc.pdf', dpi=300)


def calculate_metrics(net, trainloader, testloader, device):
    # Initialize lists to store predictions and true labels
    all_train_preds = []
    all_train_labels = []
    all_test_preds = []
    all_test_labels = []
    
    # Get predictions and labels for training data
    net.eval()
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.reshape(-1, 1, 28, 28).float()
            images = images.repeat(1, 3, 1, 1)  
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        # Get predictions and labels for test data
        for data in testloader:
            images, labels = data
            images = images.reshape(-1, 1, 28, 28).float()
            images = images.repeat(1, 3, 1, 1)  
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    # Calculate weighted metrics for training data
    train_accuracy = accuracy_score(all_train_labels, all_train_preds)
    train_precision = precision_score(all_train_labels, all_train_preds, average='macro')
    train_recall = recall_score(all_train_labels, all_train_preds, average='macro')
    train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')

    # Calculate weighted metrics for test data
    test_accuracy = accuracy_score(all_test_labels, all_test_preds)
    test_precision = precision_score(all_test_labels, all_test_preds, average='macro')
    test_recall = recall_score(all_test_labels, all_test_preds, average='macro')
    test_f1 = f1_score(all_test_labels, all_test_preds, average='macro')

    # Print the weighted metrics
    print(f"\nTrain Metrics:")
    print(f"Accuracy: {train_accuracy:.2f}")
    print(f"Precision: {train_precision:.2f}")
    print(f"Recall: {train_recall:.2f}")
    print(f"F1-Score: {train_f1:.2f}")

    print(f"\nTest Metrics:")
    print(f"Accuracy: {test_accuracy:.2f}")
    print(f"Precision: {test_precision:.2f}")
    print(f"Recall: {test_recall:.2f}")
    print(f"F1-Score: {test_f1:.2f}")

    with open(Plot_Saving_Directory+"/metrics_output.txt", "w") as file:
    # Print to the file using the 'file' argument
        print("\nTrain Metrics:", file=file)
        print(f"Accuracy: {train_accuracy:.2f}", file=file)
        print(f"Precision: {train_precision:.2f}", file=file)
        print(f"Recall: {train_recall:.2f}", file=file)
        print(f"F1-Score: {train_f1:.2f}", file=file)
        
        print("\nTest Metrics:", file=file)
        print(f"Accuracy: {test_accuracy:.2f}", file=file)
        print(f"Precision: {test_precision:.2f}", file=file)
        print(f"Recall: {test_recall:.2f}", file=file)
        print(f"F1-Score: {test_f1:.2f}", file=file)

# Call the function to calculate and print the metrics
calculate_metrics(model, train_dataloader, test_dataloader, device)

# Print model parameters
print("Model Parameters:\n")
for name, param in model.named_parameters():
    print(f"Name: {name}")
    print(f"Shape: {param.shape}")
    print(f"Requires Grad: {param.requires_grad}")
    print("-" * 40)

# Open the file in write mode
with open(Plot_Saving_Directory+"/model_parameters.txt", "w") as f:
    # Write the header
    f.write("Model Parameters:\n\n")
    
    # Loop through the model parameters and write to the file
    for name, param in model.named_parameters():
        f.write(f"Name: {name}\n")
        f.write(f"Shape: {param.shape}\n")
        f.write(f"Requires Grad: {param.requires_grad}\n")
        f.write("-" * 40 + "\n")