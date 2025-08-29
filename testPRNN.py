# --- PyTorch ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
# --- Numerical / Scientific ---
import numpy as np
import random
from scipy.signal import find_peaks
from scipy.stats import linregress
# --- PCA ---
from sklearn.decomposition import PCA
# --- Visualization ---
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# --- OS / File Handling ---
import os
import yaml 
# --- Project-specific modules ---
from zrnn.models import PRNN
from zrnn.datasets import PulseStimuliDatasetPRNN, generate_stimuli


matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting (e.g., on servers)

# Graph training and validation losses
def training_analysis(config):
    save_dir = config['training']['save_folder']
    try:
        training_output = torch.load(os.path.join(save_dir, config['training']['training_outputs']))
        try:    
            train_losses = training_output['training_losses']
            val_losses = training_output['validation_losses']
            saved_epochs = training_output['saved epochs']
        except:
            print("Error occured, skipping training analysis")
            return
    except FileNotFoundError:
        print ("Training output file doesn't exist")
        return

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')

    # Plot red dots for saved epochs (lowest val loss)
    saved_val_losses = [val_losses[epoch] for epoch in saved_epochs]
    plt.plot(saved_epochs, saved_val_losses, 'ro', label='Saved Epoch (Min Val Loss)')

    plt.title('Training and Validation Loss of RNN', fontsize=22 ) 
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss (MSE)', fontsize=18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
                              "training_validation_loss_period_prediction_lstm.png")) 
    plt.close()
    print(f"Saved training/validation loss to {os.path.join(save_dir, 'training_validation_loss_period_prediction_lstm.png')}")

# Track predictions for select periods per-timestep
def Trace_prediction(config, device, model, chosen_periods):
    plt.figure(figsize=(12, 6))
    
    # Generate a color for each period
    colors = cm.tab10(np.linspace(0, 1, len(chosen_periods)))  # tab10 gives up to 10 distinct colors

    for idx, chosen_period in enumerate(chosen_periods):
        color = colors[idx]
        
        # --- Generate stimulus ---
        T_onset = np.random.uniform(0, 0.1)
        _, i_stim_seq, _, _ = generate_stimuli(
            chosen_period, T_onset=T_onset, duration=52, PRNN_training=False
        )
        inputs_Istim = torch.tensor(i_stim_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

        # --- Forward pass ---
        with torch.no_grad():
            hidden = model.initHidden(inputs_Istim.size(0))
            outputs = model(inputs_Istim, hidden, return_hidden=False)

        # --- Target tensor ---
        target_tensor = torch.full_like(inputs_Istim, chosen_period)

        # --- Plot both prediction and target with the same color ---
        plt.plot(outputs.squeeze().cpu().numpy(), color=color, label=f"Predicted {chosen_period}s")
        plt.plot(target_tensor.squeeze().cpu().numpy(), linestyle="--", color=color, label=f"Target {chosen_period}s")

    # --- Global plot formatting ---
    plt.title("Predicted vs Target Periods during whole duration", fontsize = 20)
    plt.xlabel("Time step", fontsize = 20)
    plt.ylabel("Period value", fontsize = 20)
    plt.grid(True)

    # Ticks
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)

    # Move legend outside
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize = 18)
    plt.tight_layout()
    
    plot_path = os.path.join(config['training']['save_folder'], "period_comparison_overlay.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved overlay plot to {plot_path}")

# Analyse output from testing unque periods
def Output_analysis (config, true_periods, predicted_periods):
    # --- Plot True vs. Predicted Periods ---
    plt.figure(figsize=(8, 8))
    plt.scatter(true_periods, predicted_periods, alpha=0.7)
    # Plotting the ideal prediction line (y=x)
    max_val = max(np.max(true_periods), np.max(predicted_periods))
    min_val = min(np.min(true_periods), np.min(predicted_periods))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

    # Fit a linear regression line
    slope, intercept, r_value, p_value, std_err = linregress(true_periods, predicted_periods)
    print(f"Linear regression: slope={slope:.3f}, intercept={intercept:.3f}, p-value={p_value}")
    plt.plot([min(true_periods), max(true_periods)],
         [slope*min(true_periods)+intercept, slope*max(true_periods)+intercept],
         'g-', label=f'Fitted regression: y={slope:.2f}x+{intercept:.2f}')


    plt.title('True vs. Predicted', fontsize = 32)
    plt.xlabel('True Period (s)', fontsize = 32)
    plt.ylabel('Predicted Period (s)', fontsize = 32)
    plt.grid(True)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 18)
    plt.tight_layout()
    plot_path = os.path.join(config['training']['save_folder'], "test_true_vs_predicted_period.png") #config['training']['save_folder']
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved true vs. predicted period plot to {plot_path}")
    print("\n--- Model Testing Complete ---")

# Test model on unique periods
def test_model(model, test_loader, device, config):
    # --- Model Evaluation ---
    print("\n--- Model Evaluation ---")
    true_periods = []
    predicted_periods = []

    with torch.no_grad(): # Disable gradient calculations for inference
        for I_stim_batch, period_batch, I_stim_lengths in test_loader:
            inputs_Istim = I_stim_batch.unsqueeze(2).to(device)

            # Get the actual batch size of the CURRENT batch
            current_batch_size = inputs_Istim.shape[0]

            hidden = model.initHidden(current_batch_size) # Initialize hidden for this batch
            predicted_period_tensor = model(inputs_Istim, hidden, return_hidden=False) # Pass hidden to model

            #print ("preditions tensor:", predicted_period_tensor.shape)
            #print ("period_batch tensor:", period_batch.shape)

            # Loop through batch to get final prediction of each sample
            for i in range(current_batch_size):
                
                last_valid_idx = I_stim_lengths[i] - 1
                last_valid_prediction = predicted_period_tensor[i, last_valid_idx, 0].item() # get final prediction
                true_period = period_batch[i, last_valid_idx].item() # get true period for sample

                true_periods.append(true_period)
                predicted_periods.append(last_valid_prediction)

    true_periods = np.array(true_periods)
    predicted_periods = np.array(predicted_periods)

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predicted_periods - true_periods))
    print(f"Mean Absolute Error for Period Prediction on {config['test']['num_samples']} test samples: {mae:.4f}")

    # Calculate R-squared (coefficient of determination)
    ss_res = np.sum((true_periods - predicted_periods) ** 2)
    ss_tot = np.sum((true_periods - np.mean(true_periods)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    print(f"R-squared for Period Prediction: {r_squared:.4f}")

    Output_analysis(true_periods=true_periods, predicted_periods=predicted_periods, config=config)

# Analyse neuron activity for select periods for whole sample duration
def neuron_raster_peaks(model, device, chosen_periods, save_dir=None, threshold=0.5, idx=0):

    for chosen_period in chosen_periods:
        # --- Generate stimulus ---
        T_onset = np.random.uniform(0, 0.1)
        _, i_stim_seq, _, _ = generate_stimuli(
            chosen_period, T_onset=T_onset, duration=52, PRNN_training=False
        )
        inputs_Istim = (
            torch.tensor(i_stim_seq, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(device)
        )

        # --- Forward pass ---
        with torch.no_grad():
            hidden = model.initHidden(inputs_Istim.size(0))
            _, hidden_states, _ = model(inputs_Istim, hidden, return_hidden=True)

        # --- Convert to numpy & normalize per neuron to [-1, 1] ---
        hs = hidden_states[idx].cpu().numpy()  # (time, neurons)
        hs = hs / (np.max(np.abs(hs), axis=0) + 1e-8)

        num_neurons = hs.shape[1]

        # --- Collect positive & negative peaks ---
        neuron_pos_peaks, neuron_neg_peaks = [], []
        for n in range(num_neurons):
            pos_peaks, _ = find_peaks(hs[:, n], height=threshold)
            neg_peaks, _ = find_peaks(-hs[:, n], height=threshold)
            neuron_pos_peaks.append(pos_peaks)
            neuron_neg_peaks.append(neg_peaks)

        # --- Plot raster ---
        plt.figure(figsize=(20, 8))
        for n in range(num_neurons):
            plt.vlines(neuron_pos_peaks[n], n + 0.5, n + 1.5, color="red")
            plt.vlines(neuron_neg_peaks[n], n + 0.5, n + 1.5, color="blue")

        plt.xlabel("Time Steps", fontsize=24)
        plt.ylabel("Neuron Index", fontsize=24)
        plt.xticks(fontsize = 22)
        plt.yticks(fontsize = 22)
        plt.title(
            f"Neuron Raster Plot (+/- Threshold={threshold}) - Period={chosen_period}s",
            fontsize=20,
        )
        plt.ylim(0.5, num_neurons + 0.5)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="red", lw=2, label="Positive ≥ +threshold"),
            Line2D([0], [0], color="blue", lw=2, label="Negative ≤ -threshold"),
        ]
        plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=18)

        plt.tight_layout()

        # Save each figure separately
        plot_path = os.path.join(
            save_dir, f"neuron_raster_peaks_period_{chosen_period}.png"
        )
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        print(f"Saved neuron raster plot to {plot_path}")

# PCA plot of PRNN's hidden unit trajectories
def pca_shared_plot(model, device, save_dir=None):
    from matplotlib.lines import Line2D

    periods = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    all_rates, valid_periods = [], []

    # --- Generate hidden states ---
    for period in periods:
        T_onset = np.random.uniform(0, 0.1)
        _, i_stim_seq, _, _ = generate_stimuli(period, T_onset=T_onset, duration=52, PRNN_training=False)
        inputs_Istim = torch.tensor(i_stim_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            hidden = model.initHidden(inputs_Istim.size(0))
            _, hidden_states, _ = model(inputs_Istim, hidden, return_hidden=True)

        hs = hidden_states[0].cpu().numpy()
        hs = (hs - np.min(hs, axis=0)) / (np.max(hs, axis=0) - np.min(hs, axis=0) + 1e-8)
        hs = hs * 2 - 1

        all_rates.append(hs)
        valid_periods.append(period)

    if len(all_rates) == 0:
        print("No valid hidden states found!")
        return

    # --- Fit PCA ---
    combined_rates = np.vstack(all_rates)
    pca_3d = PCA(n_components=3)
    pca_3d.fit(combined_rates)
    print("Explained variance ratios (3D PCA):", pca_3d.explained_variance_ratio_)

    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_periods)))
    means, std_devs = [], []

    # --- 3D PCA ---
    fig = plt.figure(figsize=(12, 9))
    ax3d = fig.add_subplot(111, projection='3d')
    lines_3d = []

    for i, period in enumerate(valid_periods):
        traj_3d = pca_3d.transform(all_rates[i])
        mid = len(traj_3d) // 2

        ax3d.plot(traj_3d[:mid, 0], traj_3d[:mid, 1], traj_3d[:mid, 2],
                  linestyle=':', color=colors[i], linewidth=1.0, alpha=0.8)
        line2, = ax3d.plot(traj_3d[mid:, 0], traj_3d[mid:, 1], traj_3d[mid:, 2],
                           linestyle='-', color=colors[i], linewidth=1.5, alpha=0.9)

        ax3d.scatter(traj_3d[0,0], traj_3d[0,1], traj_3d[0,2], color='yellow', s=50, edgecolors='k', zorder=5)
        ax3d.scatter(traj_3d[-1,0], traj_3d[-1,1], traj_3d[-1,2], color='red', marker='s', s=60, edgecolors='k', zorder=6)

        diffs = np.diff(traj_3d, axis=0)
        step_lengths = np.linalg.norm(diffs, axis=1)
        means.append(np.mean(step_lengths))
        std_devs.append(np.std(step_lengths))

        lines_3d.append(line2)

    # Labels & title
    ax3d.set_xlabel("PC1", fontsize=18, labelpad=10)
    ax3d.set_ylabel("PC2", fontsize=18, labelpad=15)
    ax3d.set_zlabel("PC3", fontsize=18, labelpad=-35)  # closer to grid
    ax3d.set_title("PRNN hidden layer PCA", fontsize=22)
    ax3d.tick_params(axis='both', which='major', labelsize=20)

    # Legend: only colors
    color_legend = [Line2D([0], [0], color=colors[i], lw=3, label=f"Period {valid_periods[i]}") 
                    for i in range(len(valid_periods))]
    ax3d.legend(handles=color_legend, loc='upper left', title="Periods")

    if save_dir:
        plt.savefig(os.path.join(save_dir, "pca_trajectories_3d_vibrant.png"), bbox_inches='tight')
    plt.show()

    # --- 2D PCA ---
    plt.figure(figsize=(10, 8))
    lines_2d = []
    pca_2d = PCA(n_components=2)
    pca_2d.fit(combined_rates)

    for i, period in enumerate(valid_periods):
        traj_2d = pca_2d.transform(all_rates[i])
        mid = len(traj_2d) // 2

        plt.plot(traj_2d[:mid, 0], traj_2d[:mid, 1], linestyle=':', color=colors[i], linewidth=1.0, alpha=0.8)
        line2, = plt.plot(traj_2d[mid:, 0], traj_2d[mid:, 1], linestyle='-', color=colors[i], linewidth=1.5, alpha=0.9)
        lines_2d.append(line2)

        plt.scatter(traj_2d[0,0], traj_2d[0,1], color='yellow', s=50, edgecolors='k', zorder=5)
        plt.scatter(traj_2d[-1,0], traj_2d[-1,1], color='red', marker='s', s=60, edgecolors='k', zorder=6)

    plt.xlabel("PC1", fontsize=20, labelpad=10)
    plt.ylabel("PC2", fontsize=20, labelpad=10)
    plt.title("PCA trajectories on 2D plane", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Legend: only colors
    color_legend_2d = [Line2D([0], [0], color=colors[i], lw=3, label=f"Period {valid_periods[i]}") 
                       for i in range(len(valid_periods))]
    plt.legend(handles=color_legend_2d, title="Periods", title_fontsize = 16, fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))

    if save_dir:
        plt.savefig(os.path.join(save_dir, "pca_trajectories_2d_vibrant.png"), bbox_inches='tight')
    plt.show()

    # --- Mean vs Std trajectory length ---
    plt.figure(figsize=(8, 6))
    plt.scatter(means, std_devs, color='blue')
    for i, period in enumerate(valid_periods):
        plt.annotate(f"{period}", (means[i], std_devs[i]), xytext=(3, -12), textcoords='offset points', ha='center')

    plt.xlabel("Mean Trajectory Length", fontsize=18)
    plt.ylabel("Std of Trajectory Length", fontsize=18)
    plt.title("Mean vs Std of Trajectory Lengths", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "trajectory_mean_vs_std_vibrant.png"), bbox_inches='tight')
    plt.show()

def generate_unique_random_list(x, config):
    # Set of existing numbers
    existing_numbers = set(config['training']['periods']) | set(config['validation']['periods'])

    # Higher resolution: 0.020 to 1.000, step 0.001
    all_possible = {round(i * 0.001, 3) for i in range(20, 1001)}

    # Remove existing numbers
    available = list(all_possible - existing_numbers)

    if len(available) < x:
        raise ValueError(f"Not enough unique values to choose from: need {x}, but only {len(available)} available.")

    return random.sample(available, x) 

def custom_collate (batch):
    
    I_stim_seqs = [item[0] for item in batch]  # I_stim sequences, shape varies
    I_Stim_lengths = torch.tensor ([item[1] for item in batch]) # I_Stim lengths
    periods = [item[2] for item in batch] # scaled periods, same shape as I_Stim seqs

    padded_I_stim = pad_sequence(I_stim_seqs, batch_first=True, padding_value= -1)
    padded_periods = pad_sequence(periods, batch_first=True, padding_value=-1)

    return padded_I_stim, padded_periods, I_Stim_lengths

def main():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected'}")

    # config file
    config_path='config_PRNN.yaml' 
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Testing with device:', device)

    # --- Parameters (from config_PRNN.yaml) ---
    DURATION = 52
    print("make period list")
    # using random lsit generator
    PERIODS = generate_unique_random_list(x=config['test']['num_samples'], config=config)

    print ("make dataset")
    BATCH_SIZE = config['test']['batch_size'] # Can be 1 for detailed per-sample analysis, or larger
    MODEL_PATH = os.path.join(config['training']['save_folder'], config['training']['save_path'])  #config['training']['save_folder']
    print ("dataset built")

    os.makedirs(config['training']['save_folder'], exist_ok=True) #config['training']['save_folder']

    # load model
    model = PRNN(config['model']['input_dim'], config['model']['hidden_dim'],
                           config['model']['output_dim'], num_layers=config['model']['num_layers']).to(device)

    print(f"Loading model weights from: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode (disables dropout, BatchNorm updates, etc.)
    print("Model weights loaded successfully.")

    # --- Prepare Test Dataset and DataLoader ---
    print("\nCreating test dataset and data loader...")
    test_dataset = PulseStimuliDatasetPRNN(periods=PERIODS, duration=DURATION, dt=config['model']['dt'], size=config['test']['num_samples'], PRNNtraining=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate) # No shuffle for consistent evaluation
    print(f"Test samples: {len(test_dataset)}")
    print("Test Dataset and DataLoader created.")

    # Analysis of training
    training_analysis(config=config) # analyse loss over every epoch

    # Test model and analyse output
    test_model(model, test_loader, device, config)

    chosen_periods = [0.06, 0.2, 0.43, 0.6, 0.71, 0.95]
    Trace_prediction(config, device, model, chosen_periods)

    # Additional analyses
    # PCA
    pca_shared_plot(model, device, save_dir=config['training']['save_folder'])
    # Neuron activity raster
    chosen_periods = [0.2, 0.6, 0.95]
    neuron_raster_peaks(model, device, chosen_periods, save_dir=config['training']['save_folder'])

if __name__ == "__main__":
    import fire
    fire.Fire(main)