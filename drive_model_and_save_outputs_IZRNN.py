import os
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from zrnn.models import ZemlianovaRNN, PRNN
from zrnn.datasets import generate_stimuli

matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting (e.g., on servers)

N_RANDOM_PERIODS = 100

   
def drive_model_and_save_outputs(model1, model2, periods, device, config, time_steps=52000, discard_steps=2000):
    # Create directory to save outputs if it doesn't exist
    save_dir = os.path.join(config['testing']['output_folder'], "model_outputs")
    
    os.makedirs(save_dir, exist_ok=True)

    # Generate and process inputs
    for period in periods:
        _, I_stim_PRNN, _, _ = generate_stimuli(period, 0.05, duration=52.0, I_stim_without_clicks=False, PRNN_training=False)

        _, I_stim_ZRNN, _, _ = generate_stimuli(period, 0.05, duration=52.0, I_stim_without_clicks=True, PRNN_training=False)

        I_stim_PRNN = torch.tensor(I_stim_PRNN, dtype=torch.float32).unsqueeze(0).to(device)
        I_stim_PRNN = I_stim_PRNN.unsqueeze(-1)

        I_stim_ZRNN = torch.tensor(I_stim_ZRNN, dtype=torch.float32).unsqueeze(0).to(device)
        I_stim_ZRNN = I_stim_ZRNN.unsqueeze(-1)

        hidden1 = model1.initHidden(1)
        hidden2 = model2.initHidden(1).to(device)

        # Process the input through the model with no_grad for evaluation
        with torch.no_grad():
                period_predictions = model1(I_stim_PRNN, hidden1)    
                I_cc = 0.1/period_predictions
                outputs, hidden_states = model2(I_stim_ZRNN, I_cc, hidden2)
        
        outputs = torch.stack(outputs, dim=1)
        hidden_states = torch.stack(hidden_states, dim=1)

        outputs, hidden_states = outputs.cpu(), hidden_states.cpu()

        # Process output and hidden states

        outputs = np.array(outputs).squeeze()
        valid_outputs = outputs[discard_steps:]  # Discard the first 2000 steps

        hidden_states = np.array(hidden_states.cpu()).squeeze()
        valid_hidden_states = hidden_states[discard_steps:] # Discard the first 2000 steps

        # Apply bandpass filter
        fs = 1 / 0.001  # Sampling frequency (1000 Hz, since dt = 0.001 s)
        low = 1 / (period + 0.1)  # Low frequency of the bandpass filter
        high = 1 / (period - 0.1)  # High frequency of the bandpass filter
        b, a = butter(N=2, Wn=[low, high], btype='band', fs=fs)
        filtered_outputs = filtfilt(b, a, valid_outputs) 

        # Find peaks in the filtered outputs
        peaks, _ = find_peaks(filtered_outputs, height=0)

        # Determine window size for searching the highest peak in the original signal
        window_size = int((period / 2) / 0.001)  # Half period in terms of samples

        # Find the highest peak in the original signal near each detected peak in the filtered output
        true_peaks = []
        for peak in peaks:
            start = max(0, peak - window_size)
            end = min(len(valid_outputs), peak + window_size)
            true_peak = np.argmax(valid_outputs[start:end]) + start
            true_peaks.append(true_peak)

        # Plotting
        plt.figure(figsize=(25, 5))
        plt.plot(valid_outputs, label='Raw Output for Period: ' + str(period) + 's')
        plt.plot(true_peaks, valid_outputs[true_peaks], "x", label='Highest Peaks in Raw Output')
        plt.xlabel('Time Steps (after discard)')
        plt.ylabel('Model Activity')
        plt.title('Model Output for Period ' + str(period) + 's')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'plot_period_{period}.png'))  # Save plot
        plt.close()

        # Save data to files
        np.save(os.path.join(save_dir, f'valid_output_{period}.npy'), valid_outputs)
        np.save(os.path.join(save_dir, f'peaks_{period}.npy'), true_peaks)
        np.save(os.path.join(save_dir, f'hidden_states_{period}.npy'), np.array(valid_hidden_states))
        print(f'finished driving model and saving activity for period {period}')

def sample_exponential_skew(num_samples, lam=5):
    u = np.random.rand(num_samples)
    skewed_samples = np.exp(-lam * (1 - u))
    return skewed_samples

def main(config_path='config_IZRNN.yaml', model_type=None):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    model1 = PRNN(config['modelPRNN']['input_dim'], config['modelPRNN']['hidden_dim'], config['modelPRNN']['output_dim'], config['modelPRNN']['num_layers']).to(device)
    save_path1 = os.path.join(config['PRNN_dir']['save_folder'], config['PRNN_dir']['save_path'])
    print ("loading")
    model1.load_state_dict(torch.load(save_path1, map_location=device))
    print ("done")
    model1.eval()

    model2 = ZemlianovaRNN(config['modelZRNN']['input_dim'], config['modelZRNN']['hidden_dim'], config['modelZRNN']['output_dim'], config['modelZRNN']['dt'], config['modelZRNN']['tau'], config['modelZRNN']['excit_percent'], sigma_rec=config['modelZRNN']['sigma_rec']).to(device)
    save_path2 = os.path.join(config['ZRNN_dir']['save_folder'], config['ZRNN_dir']['save_path'])
    print ("loading")
    model2.load_state_dict(torch.load(save_path2, map_location=device))
    print ("done")
    model2.eval()
    
    # ADD a few extra randomly-generated periods
    drive_model_and_save_outputs(model1, model2, config['testing']['periods']+[round(v+0.1,3) for v in list(sample_exponential_skew(N_RANDOM_PERIODS))], device, config)

if __name__ == "__main__":
    import fire
    fire.Fire(main)