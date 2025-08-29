import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import yaml
import os
import re

matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting (e.g., on servers)

Ncycles = 50

def relu(x):
    return np.maximum(0, x)

def normalize(x):
    if x.size == 0:
        print("Warning: Attempted to normalize empty array.")
        return x
    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)
    range_val = max_val - min_val
    return np.where(range_val > 0, (x - min_val) / range_val, 0)

def load_and_process_data(periods, config, directory='model_outputs', save_dir='fig1_plotting'):
    # Create directory to save outputs if it doesn't exist
    save_path = os.path.join(config['testing']['output_folder'], save_dir)
    os.makedirs(save_path, exist_ok=True)
    
    for period in periods:
        hidden_state_path = os.path.join(config['testing']['output_folder'], directory, f'hidden_states_{period}.npy')
        peaks_path = os.path.join(config['testing']['output_folder'], directory, f'peaks_{period}.npy')

        if os.path.exists(hidden_state_path) and os.path.exists(peaks_path):
            hidden_states = np.squeeze(np.load(hidden_state_path))
            peaks = np.load(peaks_path)
            
            firing_rates = relu(hidden_states)
            normalized_firing_rates = normalize(firing_rates)
            if normalized_firing_rates.size == 0 or normalized_firing_rates.ndim < 2:
                print(f"Warning: normalized firing rates for period {period} are invalid. Skipping.")
                continue
                    
            num_neurons = normalized_firing_rates.shape[1]
            excitatory_indices = range(int(num_neurons * 0.8))
            inhibitory_indices = range(int(num_neurons * 0.8), num_neurons)
           
            distances = {}
            for neuron_index in range(num_neurons):
                neuron_firing_rates = normalized_firing_rates[:, neuron_index]
                neuron_peaks = []
                if neuron_firing_rates.max():
                    for i in range(1, len(peaks) - 1):
                        window_start = peaks[i] - (peaks[i] - peaks[i - 1]) // 2
                        window_end = peaks[i] + (peaks[i + 1] - peaks[i]) // 2

                        #print (len(neuron_firing_rates[window_start:window_end]))
                        segment = neuron_firing_rates[window_start:window_end]
                        if segment.size == 0:
                            continue   # or assign a default
                        max_index = np.argmax(segment) + window_start
                        neuron_peaks.append(max_index)
                else:
                    neuron_peaks = peaks[1:-1]-1

                # Calculate distances from defined peaks to actual neuron peaks
                distances[neuron_index] = np.median([p - peaks[i+1] for i, p in enumerate(neuron_peaks)])
            
            # Separate sorting for excitatory and inhibitory neurons
            def get_sorted_indices(indices):
                positive_sorted = []
                negative_sorted = []
                
                for idx in indices:
                    pos_distances = True if distances[idx] >= 0 else False
                    
                    if pos_distances:
                        positive_sorted.append((idx,distances[idx]))
                    else:
                        negative_sorted.append((idx,distances[idx]))
                
                # Sort both lists
                positive_sorted.sort(key=lambda x: x[1])  # Ascending order for positive
                negative_sorted.sort(key=lambda x: x[1])  # Descending order for negative
                
                # Combine and extract indices
                return [x[0] for x in positive_sorted + negative_sorted]
            
            excitatory_sorted = get_sorted_indices(excitatory_indices)
            inhibitory_sorted = get_sorted_indices(inhibitory_indices)
            
            # Combine sorted indices
            sorted_indices = excitatory_sorted + inhibitory_sorted
            
            # Visualization
            plt.figure(figsize=(20, 8))
            plt.imshow(normalized_firing_rates[10000:11000, sorted_indices].T, aspect='auto', cmap='viridis', interpolation='nearest')
            for peak in peaks:
                if 10000 < peak < 11000:  # Ensure peak is within the first 1000 samples
                    plt.axvline(x=peak-10000, color='k', linestyle='--')
            
            cbar = plt.colorbar()
            cbar.set_label('Normalized Firing Rate', fontsize=18)
            cbar.ax.tick_params(labelsize=18)

            plt.title(f'Firing Rates Sorted by Neuron Distance to Peak for Period {period}s', fontsize = 20)
            plt.xlabel('Time Steps', fontsize = 20)
            plt.ylabel('Neuron Index', fontsize = 20)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.axhline(y=len(excitatory_sorted), color='r', linestyle='--')  # Divide line between Excitatory and Inhibitory
            plt.savefig(os.path.join(save_path, f'firing_rate_plot_{period}.png'))  # Save plot

            print(f"Processed data for period {period}. Neurons sorted by distance to peaks.")
            np.save(os.path.join(save_path, f'neuron_sorting_indices_for_{period}.npy'), sorted_indices)

def calculate_cycle_lengths(pca_results, peaks):
    # Calculate distances between consecutive peaks in PCA space
    distances = []
    for i in range(1, len(peaks)):
        if peaks[i] < pca_results.shape[0]:
            dist = pca_results[peaks[i-1]:peaks[i]]
            dist = np.diff(dist,axis=0)
            dist = np.sqrt(np.sum(dist**2, axis=1))
        distances.append(np.sum(dist))
    return distances 

def load_data_compute_pca_and_plot(periods, config, directory='model_outputs', save_dir='fig1_plotting'):
    directory_path = os.path.join(config['testing']['output_folder'], directory)

    all_rates = []
    valid_periods = []
    means, std_devs = [], []

    # --- Load and preprocess hidden states ---
    for period in periods:
        hidden_state_path = os.path.join(directory_path, f'hidden_states_{period}.npy')
        if os.path.exists(hidden_state_path):
            hidden_states = np.squeeze(np.load(hidden_state_path))
            firing_rates = relu(hidden_states)
            firing_rates = normalize(firing_rates)
            all_rates.append(firing_rates)
            valid_periods.append(period)
        else:
            print(f"Warning: Missing hidden_states for period {period}, skipping.")

    if not all_rates:
        print("No valid hidden states found, exiting.")
        return

    combined_rates = np.vstack(all_rates)
    pca = PCA(n_components=3)
    pca.fit(combined_rates)

    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_periods)))

    # --- Create 2x2 figure ---
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(221, projection='3d')  # PCA
    ax2 = fig.add_subplot(222)  # Context cue vs intervals
    ax3 = fig.add_subplot(223)  # Mean vs Std
    ax4 = fig.add_subplot(224)  # Filtered Mean vs Std

    # --- PCA Trajectories with Peaks ---
    for i, period in enumerate(valid_periods):
        firing_rates = all_rates[i]
        pca_results = pca.transform(firing_rates)

        peaks_path = os.path.join(directory_path, f'peaks_{period}.npy')
        peaks = np.load(peaks_path) if os.path.exists(peaks_path) else []

        ax1.plot(pca_results[:, 0], pca_results[:, 1], pca_results[:, 2],
                 label=f'{1/period:.2f} Hz', color=colors[i])
        for peak in peaks:
            ax1.scatter(pca_results[peak, 0], pca_results[peak, 1], pca_results[peak, 2],
                        color='black', s=10)

        # Compute trajectory step lengths from peaks
        cycle_lengths = calculate_cycle_lengths(firing_rates, peaks)
        if cycle_lengths:
            means.append(np.mean(cycle_lengths))
            std_devs.append(np.std(cycle_lengths))

    ax1.set_xlabel('PC1', fontsize=18)
    ax1.set_ylabel('PC2', fontsize=18)
    ax1.set_zlabel('PC3', fontsize=18, labelpad = -25)
    ax1.set_title('PCA Trajectories', fontsize=18)
    ax1.tick_params(labelsize=14)
    ax1.legend(fontsize=14, loc='center left', bbox_to_anchor=(-0.3, 0.5), borderaxespad=0)

    # --- Context Cue vs Avg Inter-Peak Interval ---
    context_cues, average_intervals, point_colors = [], [], []
    filename_pattern = re.compile(r'peaks_([0-9]+\.[0-9]+)\.npy')

    for filename in os.listdir(directory_path):
        match = filename_pattern.match(filename)
        if match:
            period = float(match.group(1))
            peaks = np.load(os.path.join(directory_path, filename))

            intervals = np.diff(peaks)
            if intervals.size > 0:
                average_interval = np.mean(intervals)
                context_cue = 0.1 / period
                context_cues.append(context_cue)
                average_intervals.append(average_interval)

                if period in valid_periods:
                    point_colors.append('orange')
                else:
                    point_colors.append('gray')

    ax2.scatter(context_cues, average_intervals, c=point_colors)
    ax2.set_xlabel('Context Cue Size (0.1 / Period)', fontsize=18)
    ax2.set_ylabel('Average Inter-Peak Interval', fontsize=18)
    ax2.set_title('Context Cue vs Inter-Peak Interval', fontsize=18)
    ax2.tick_params(labelsize=14)
    ax2.grid(True)

    # --- Mean vs Std (all periods) ---
    ax3.scatter(means, std_devs, color='blue')
    for i, p in enumerate(valid_periods):
        ax3.annotate(f'{1/p:.2f} Hz', (means[i], std_devs[i]), fontsize=12)

    ax3.set_xlabel('Mean Trajectory Length', fontsize=18)
    ax3.set_ylabel('Std of Trajectory Length', fontsize=18)
    ax3.set_title('Mean vs Std of Trajectories', fontsize=18)
    ax3.tick_params(labelsize=14)

    # --- Mean vs Std (filtered periods <= 0.5s) ---
    filtered_means = [m for p, m in zip(valid_periods, means) if p <= 0.5]
    filtered_stds = [s for p, s in zip(valid_periods, std_devs) if p <= 0.5]
    filtered_periods = [p for p in valid_periods if p <= 0.5]

    ax4.scatter(filtered_means, filtered_stds, color='green')
    for i, p in enumerate(filtered_periods):
        ax4.annotate(f'{1/p:.2f} Hz', (filtered_means[i], filtered_stds[i]), fontsize=12)

    ax4.set_xlabel('Mean Trajectory Length', fontsize=18)
    ax4.set_ylabel('Std of Trajectory Length', fontsize=18)
    ax4.set_title('Mean vs Std of Trajectories (8-2Hz)', fontsize=18)
    ax4.tick_params(labelsize=14)

    plt.tight_layout()
    save_path = os.path.join(config['testing']['output_folder'], save_dir, "pca_and_stats_2x2.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

config_path='config_IZRNN.yaml'
with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

periods = [1, 0.93, 0.86, 0.75, 0.65, 0.500, 0.46, 0.333, 0.250, 0.200, 0.166, 0.143, 0.125] 
 
data = load_and_process_data(periods, config=config)
pca_model = load_data_compute_pca_and_plot(periods, directory='model_outputs', config=config)
