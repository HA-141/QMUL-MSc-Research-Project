import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import yaml
import os
from zrnn.models import PRNN
from zrnn.datasets import PulseStimuliDatasetPRNN

# Define Masked MSE calculation
def masked_MSE_lossFN (output_tensor, target_tensor, lengths_istim):
    
    # Move lengths to correct device
    lengths_istim_device = lengths_istim.to(target_tensor.device)

    seq_len = target_tensor.shape[1]
    padding_mask = (torch.arange(seq_len, device=target_tensor.device).unsqueeze(0) < lengths_istim_device.unsqueeze(1)).float()
    padding_mask = padding_mask.to(output_tensor.device)

    padded_squareDiff = ((output_tensor - target_tensor)**2).squeeze(-1)

    masked_Diff = padded_squareDiff * padding_mask # Zero out padded -1 tokens
    Mse = masked_Diff.sum()/padding_mask.sum()
    
    return Mse

# Main training function
def train(model, train_loader, val_loader, optimizer, config, device):
    best_val_loss = float('inf') 
    train_losses = []             
    val_losses = []  
    saved_epochs = []             
    
    num_epochs = config['training']['epochs']
    early_stopping_threshold = config['training'].get('early_stopping_loss', 0.0001)

    print("\nStarting training...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train() 
        epoch_train_loss = 0

        for I_stim_batch, period_batch, I_stim_lengths in train_loader:

            inputs_Istim = I_stim_batch.unsqueeze(-1).to(device) 
                        
            target_period = period_batch.unsqueeze(-1).to(device) 
            
            # Get the actual batch size of the CURRENT batch
            current_batch_size = inputs_Istim.shape[0]

            hidden = model.initHidden(current_batch_size)

            predicted_period = model(inputs_Istim, hidden)
            
            loss = masked_MSE_lossFN(predicted_period, target_period, I_stim_lengths) #criterion
            
            # Add logging here for input and output example
            if epoch == 0 and torch.rand(1).item() < 0.01:  # Random small chance to log a few samples
                for i in range(min(3, current_batch_size)):
                    last_valid_idx = I_stim_lengths[i] - 1

                    sample_prediction = predicted_period[i, last_valid_idx, 0].item() 
                    print("\n--- Example Sample ---")
                    print("Target Period:", period_batch[i, last_valid_idx].item())
                    print("Predicted Period:", sample_prediction)


            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

            epoch_train_loss += loss.item() * current_batch_size # do account for packing of different sample lengths


        avg_train_loss = epoch_train_loss / len(train_loader.dataset) 
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        epoch_val_loss = 0
        with torch.no_grad(): # Disable gradient calculation for validation
            for I_stim_batch_val, period_batch_val, I_stim_lengths in val_loader: 
                inputs_Istim_val = I_stim_batch_val.unsqueeze(-1).to(device)
                target_peroid_val = period_batch_val.unsqueeze(-1).to(device)
                
                # Get the actual batch size of the CURRENT validation batch
                current_val_batch_size = inputs_Istim_val.shape[0]

                hidden_val = model.initHidden(current_val_batch_size)

                predicted_period_val = model(inputs_Istim_val, hidden_val)

                val_loss = masked_MSE_lossFN(predicted_period_val, target_peroid_val, I_stim_lengths) # criterion
                epoch_val_loss += val_loss.item() * current_val_batch_size

            current_val_loss = epoch_val_loss / len(val_loader.dataset) 
            val_losses.append(current_val_loss)

        # --- Logging and Model Saving (based on Validation Loss) ---
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {current_val_loss:.6f}")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss # set new best loss
            saved_epochs.append(epoch)

            # Combine folder and file name into full path
            save_path = os.path.join(config['training']['save_folder'], config['training']['save_path'])
            torch.save(model.state_dict(), save_path) #torch.save(model.state_dict(),  config['training']['save_path'])
            print(f"Model saved with improvement at epoch {epoch+1} with Val Loss: {best_val_loss:.6f}")
        
        # update epoch losses
        save_epoch_losses(train_losses, val_losses, saved_epochs, config)   

        # --- Early Stopping Check (based on Validation Loss) ---
        if current_val_loss <= early_stopping_threshold:
            print(f"Early stopping triggered at epoch {epoch+1} as validation loss ({current_val_loss:.6f}) "
                  f"is below threshold ({early_stopping_threshold:.6f}).")
            break

    print("Training finished!")
    return model

def save_epoch_losses(test_losses, val_losses, saved_epochs, config):
    # save best the loss of every epoch
        # Combine folder and file name into full path
        save_path = os.path.join(config['training']['save_folder'], config['training']['training_outputs'])
        torch.save({'training_losses': test_losses, # training loss per epoch
                    'validation_losses': val_losses, # validation loss per epoch
                    'saved epochs' : saved_epochs
                    }, save_path) #config['training']['training_outputs']

def custom_collate (batch):
    
    I_stim_seqs = [item[0] for item in batch]  # I_stim sequences, shape varies
    I_Stim_lengths = torch.tensor ([item[1] for item in batch]) # I_Stim lengths
    periods = [item[2] for item in batch] # scaled periods, same shape as I_Stim seqs

    padded_I_stim = pad_sequence(I_stim_seqs, batch_first=True, padding_value= -1)
    padded_periods = pad_sequence(periods, batch_first=True, padding_value=-1)

    return padded_I_stim, padded_periods, I_Stim_lengths


def main(config_path='config_PRNN.yaml'):

    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Training with device:', device)

    #test_dataset = PulseStimuliDataset(periods = config['training']['periods'], size=config['training']['dataset_size'], dt=config['model']['dt'])
    train_dataset = PulseStimuliDatasetPRNN(periods = config['training']['periods'], size=config['training']['dataset_size'], dt=config['model']['dt'], PRNNtraining=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=custom_collate)    
    print ('done')

    #val_dataset = PulseStimuliDataset(periods=config['validation']['periods'], size=config['validation']['dataset_size'], dt=config['model']['dt'])
    val_dataset = PulseStimuliDatasetPRNN(periods=config['validation']['periods'], size=config['validation']['dataset_size'], dt=config['model']['dt'], PRNNtraining=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['validation']['batch_size'], shuffle=False, collate_fn=custom_collate) # No shuffle for validation
    print ('done')

    model = PRNN(config['model']['input_dim'], config['model']['hidden_dim'], config['model']['output_dim'], config['model']['num_layers']).to(device)
   
    print ("model initialised")

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    #criterion = nn.MSELoss()

    # Ensure the folder exists and create it
    os.makedirs(config['training']['save_folder'], exist_ok=True)

    print ("optimiser and criterion done")
    model = train(model, train_dataloader, val_dataloader, optimizer, config, device)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
