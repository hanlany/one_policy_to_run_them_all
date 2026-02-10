import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class TeacherStudentDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.states = torch.FloatTensor(data['states'])
        self.actions = torch.FloatTensor(data['actions'])
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class StudentPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        super(StudentPolicy, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def train():
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    # HIDDEN_DIMS = [1024, 512, 256]

    HIDDEN_DIMS = [1024, 1024, 1024, 1024, 1024]
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    dataset_path = 'teacher_dataset.npz'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    print(f"Loading dataset from {dataset_path}...")
    dataset = TeacherStudentDataset(dataset_path)
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    input_dim = dataset.states.shape[1]
    output_dim = dataset.actions.shape[1]
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    
    model = StudentPolicy(input_dim, output_dim, hidden_dims=HIDDEN_DIMS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Training Loop
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            
            optimizer.zero_grad()
            predicted_actions = model(states)
            loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                predicted_actions = model(states)
                loss = criterion(predicted_actions, actions)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'student_model_best.pth')
            # print("Saved best model.")

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")
    print("Best model saved to 'student_model_best.pth'")

if __name__ == "__main__":
    train()
