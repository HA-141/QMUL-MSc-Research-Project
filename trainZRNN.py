import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib
import yaml
import os
from zrnn.models import ZemlianovaRNN
from zrnn.datasets import PulseStimuliDatasetZRNN

matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting (e.g., on servers)


def train(model, dataloader, optimizer, criterion, config, device):
    model.train()  # Set the model to training mode
    min_loss = float('inf')  # Initialize the minimum loss to a large value

    num_epochs = config['trainingZRNN']['epochs']
    early_stopping_loss = config['trainingZRNN']['early_stopping_loss']

    print("\nStarting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        for i_stim, i_cc, targets, _ in dataloader:
            i_stim, i_cc, targets = i_stim.to(device), i_cc.to(device), targets.to(device)  # Move data to GPU
            
            # Initialise hidden state
            batch_size = i_stim.shape[0]
            hidden = model.initHidden(batch_size).to(device)

            # Train model
            outputs, hidden = model(i_stim, i_cc, hidden)

            # Save output and optimise 
            outputs = torch.stack(outputs, dim=1)
            loss = criterion(outputs[..., 0], targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate loss for this epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['trainingZRNN']['epochs']}, Loss: {avg_loss}")

        if avg_loss < min_loss:
            min_loss = avg_loss # update new best loss

            # Combine folder and file name into full path and save current model
            save_path = os.path.join(config['ZRNN_dir']['save_folder'], config['ZRNN_dir']['save_path'])
            torch.save(model.state_dict(), save_path)

            print(f"Model saved with improvement at epoch {epoch+1} with loss {min_loss}")

        if avg_loss <= early_stopping_loss:
            print("Early stopping threshold reached.")
            break

    return model

def main(config_path='config_IZRNN.yaml'): 
    
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Training with device:', device)

    print ("generating training dataset")
    train_dataset = PulseStimuliDatasetZRNN(config['trainingZRNN']['periods'], size=config['trainingZRNN']['dataset_size'], dt=config['modelZRNN']['dt'], PRNNtraining=False)
    train_dataloader = DataLoader(train_dataset, batch_size=config['trainingZRNN']['batch_size'], shuffle=True)
    print ("done")
  
    model = ZemlianovaRNN(config['modelZRNN']['input_dim'], config['modelZRNN']['hidden_dim'], config['modelZRNN']['output_dim'], config['modelZRNN']['dt'], config['modelZRNN']['tau'], config['modelZRNN']['excit_percent'], sigma_rec=config['modelZRNN']['sigma_rec']).to(device)
    print ("model initialised")

    optimizer = optim.Adam(model.parameters(), lr=config['trainingZRNN']['learning_rate'])
    criterion = nn.MSELoss() # make function to replace that

    # Ensure the folder exists and create it
    os.makedirs(config['ZRNN_dir']['save_folder'], exist_ok=True)

    model = train(model, train_dataloader, optimizer, criterion, config, device)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
