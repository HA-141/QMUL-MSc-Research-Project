from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from zrnn.models import ZemlianovaRNN, PRNN
from zrnn.datasets import generate_stimuli
import os


def relu(x: np.typing.ArrayLike) -> np.typing.ArrayLike:
    return np.maximum(x, 0)


def normalize(x: np.typing.ArrayLike) -> np.typing.ArrayLike:
    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)
    range_val = max_val - min_val
    return np.divide(x - min_val, range_val, out=np.zeros_like(x), where=range_val > 0)


def standardize(x: np.typing.ArrayLike) -> np.typing.ArrayLike:
    from scipy.stats import zscore
    return zscore(x, axis=0)


def load_model(model_path1: str | Path, model_path2: str | Path, config_path: Path | str = 'config_IZRNN.yaml', **kwargs) -> ZemlianovaRNN:
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = PRNN(**config['modelPRNN']).to(device)
    model1.load_state_dict(torch.load(model_path1, weights_only=True, map_location=device))
    model1.eval()

    model2 = ZemlianovaRNN(**config['modelZRNN']).to(device)
    map_location = kwargs.get('map_location', device)  
    model2.load_state_dict(torch.load(model_path2, weights_only=True, map_location=map_location))
    model2.eval()
    return model1, model2


def generate_stimuli_from_cc(context_cue_amplitude: float, t_onset_ms: float = .05, *args, **kwargs):
    period = .1 / context_cue_amplitude

    return torch.tensor(np.array([generate_stimuli(period, t_onset_ms, **kwargs, PRNN_training=False)[1]]),
                        dtype=torch.float)


def drive_model(model1: PRNN,
                model2: ZemlianovaRNN,
                context_cue_amplitude: float,
                time_steps_ms: int = 52_000,
                discard_steps: int = 2_000,
                initial_neuron_activity: torch.Tensor = None,
                t_onset_ms: float = .05,
                device: str = 'cpu',
                **kwargs):
    model1.to(device)
    model2.to(device)

    # Adjusted to fit whole IZRNN workflow
    # Generate signal with (PRNN) and without (ZRNN) clicks
    rnn_input_tensor_PRNN = generate_stimuli_from_cc(context_cue_amplitude,
                                                t_onset_ms=t_onset_ms,
                                                duration=52,
                                                I_stim_without_clicks=kwargs.get('I_stim_without_clicks', False)).to(device)

    rnn_input_tensor_ZRNN = generate_stimuli_from_cc(context_cue_amplitude,
                                                t_onset_ms=t_onset_ms,
                                                duration=52,
                                                I_stim_without_clicks=kwargs.get('I_stim_without_clicks', True)).to(device)
    
    # Set hidden states
    if initial_neuron_activity is None:
        neuron_state = model2.initHidden(1).to(device)
    else:
        neuron_state = initial_neuron_activity.to(device)

    hidden_state = tuple(h.to(device) for h in model1.initHidden(1))

    I_stim_PRNN = rnn_input_tensor_PRNN.unsqueeze(-1).to(device)

    I_stim_ZRNN = rnn_input_tensor_ZRNN.to(device)

    model_outputs, neuron_states = [], [] 

    # Run through IZRNN
    with torch.inference_mode():
            predicted_period = model1(I_stim_PRNN, hidden_state)
            i_cc = 0.1/predicted_period

            for t in range(time_steps_ms):
                i_stim_t = I_stim_ZRNN[:, t].unsqueeze(0).unsqueeze(-1) 
                i_cc_t = i_cc[:, t].unsqueeze(-1) 

                output, neuron_state = model2(i_stim_t, i_cc_t, neuron_state)

                model_outputs.append(output[-1]) 
                neuron_state = neuron_state[-1]
                neuron_states.append(neuron_state)

    # Process outputs
    model_outputs = torch.stack(model_outputs, dim=1).cpu()#.squeeze(0)
    neuron_states = torch.stack(neuron_states, dim=1).cpu()#.squeeze(0)

    model_outputs = np.array(model_outputs).squeeze()
    valid_outputs = model_outputs[discard_steps:]  # Discard the first 2000 steps
    
    neuron_states = np.array(neuron_states).squeeze()
    valid_neuron_state = neuron_states[discard_steps:]

    return valid_outputs, valid_neuron_state


def get_peaks(signal: np.ndarray | torch.Tensor) -> np.ndarray:
    # Since no noise in this part, no need for filtering
    from scipy.signal import find_peaks
    return find_peaks(signal, height=0)[0]


def get_neuron_phases_data(model_output: np.typing.ArrayLike, neurons_activity: np.typing.ArrayLike) -> pd.DataFrame:
    # Return the relative phase of each neuron wrt to the tap in the first ITI, along with other information

    # Normalize and retrieve inter tap interval indices
    normalized_neurons_activity = normalize(relu(neurons_activity))
    tap_indices = get_peaks(model_output)
    inter_tap_interval = slice(*tap_indices[1:3])
    inter_tap_interval_activity = normalized_neurons_activity[inter_tap_interval]
    # Calculate the phase by finding the shortest period between neuron max and tap time
    max_args = np.argmax(inter_tap_interval_activity, axis=0)
    neuron_phases = np.minimum(max_args, inter_tap_interval_activity.shape[0] - max_args)
    neuron_phases = neuron_phases / (inter_tap_interval.stop - inter_tap_interval.start)
    df = pd.DataFrame({'phase': neuron_phases,
                       'n_type': [1 if i < .8 * len(neuron_phases)
                                  else 0
                                  for i in range(len(neuron_phases))]})
    # Determine if tap or inter-tap neuron
    df['tap'] = ((df.phase > 0.) & (df.phase < .2)).astype(int)

    def _get_sub_pop(row: pd.Series) -> int:
        # Divides the neuron to subgroups. The outcome depends on the context cue - is that how it should be?
        if bool(row.tap) and bool(row.n_type):
            return 0
        if bool(row.tap) and not bool(row.n_type):
            return 1
        if not bool(row.tap) and not bool(row.n_type):
            return 2
        if not bool(row.tap) and bool(row.n_type):
            return 3

    # Determine the subgroup
    df['sub_pop'] = [_get_sub_pop(row) for _, row in df.iterrows()]
    return df


SUB_POPS = {0: 'tap_E', 1: 'tap_I', 2: 'inter_tap_I', 3: 'inter_tap_E'}


def generate_initial_neuron_activity(neuron_phases_data: pd.DataFrame, random_seed: int = 112023) -> torch.Tensor:
    generator = torch.Generator().manual_seed(random_seed)
    # for the general neuron population, draw numbers from uniform(-1, 1)
    initial_neuron_activity = 2 * torch.rand(len(neuron_phases_data), generator=generator) - 1
    # Single out the tap neuron population and draw from uniform (0, 1) to all of them:
    tap_neurons = neuron_phases_data['sub_pop'].isin([0, 1])
    initial_neuron_activity[tap_neurons] = torch.rand(1, generator=generator)
    return initial_neuron_activity.unsqueeze(0)


def get_principal_components(model1: PRNN,
                             model2: ZemlianovaRNN,
                             device: str = 'cpu',
                             context_cue_range: (float, float) = (-1, .2),
                             num_trajectories: int = 100, ############### -------- 100
                             phases_context_cue_amplitude: float = .5,
                             random_seed: int = 112023
                             ) -> (PCA, pd.DataFrame):
    # Calculate the activities for context cue in the oscillatory regime to get the subgroups
    model_output, neurons_activity = drive_model(model1, model2, phases_context_cue_amplitude, device=device)
    neuron_phases = get_neuron_phases_data(model_output, neurons_activity)
    # Now calculate the activities in the non-oscillatory regime
    all_activities = np.concat([drive_model(model1, model2, cc,
                                            time_steps_ms=500,
                                            discard_steps=0,
                                            device=device,
                                            initial_neuron_activity=generate_initial_neuron_activity(neuron_phases,
                                                                                                     random_seed))[1]
                                for cc in np.linspace(*context_cue_range, num_trajectories)])
    # Normalize and fit to the PCA. Not certain about the way of normalization here
    all_activities = standardize(all_activities)
    pca = PCA(n_components=len(neuron_phases))
    pca.fit(all_activities)
    return pca, neuron_phases


def get_vector_field(model1: PRNN,
                     model2: ZemlianovaRNN,
                     context_cue_amplitude: float,
                     initial_neuron_activity: torch.Tensor,
                     trained_pca: PCA,
                     span: (float, float) = (-5, 5),
                     center: (float, float) = (0, 0),
                     time_steps_ms: int = 500,
                     grid_res: int = 128,
                     at_time: int = None,
                     device: str = 'cpu') -> dict:
    from scipy.linalg import norm
    model1.to(device)
    model2 = model2.to('cpu')

    # PCA grid setup
    c0_points = torch.linspace(*span, grid_res).to(device) + center[0]
    c1_points = torch.linspace(*span, grid_res).to(device) + center[1]
    grid_c0, grid_c1 = torch.meshgrid(c0_points, c1_points, indexing='xy')

    # inputs
    stimulus_PRNN = generate_stimuli_from_cc(context_cue_amplitude,
                                        I_stim_without_clicks=True).to(device)

    stimulus_ZRNN = generate_stimuli_from_cc(context_cue_amplitude,
                                        I_stim_without_clicks=True).to(device)
    
    I_stim_PRNN = stimulus_PRNN.unsqueeze(-1).to(device)#[:, :, 0].unsqueeze(-1).to(device)
    I_stim_ZRNN = stimulus_ZRNN.unsqueeze(-1)#[:, :, 0] 

    #PRNN hidden state
    hidden_state_prnn = tuple(h.to(device) for h in model1.initHidden(1))    

    i_proj = torch.tensor(trained_pca.transform(initial_neuron_activity.cpu()), dtype=torch.float32).to(device)
    i_proj = torch.broadcast_to(i_proj, [grid_res, grid_res, 1, i_proj.shape[-1]]).clone()
    dc0, dc1, speeds = [], [], []

      # --- Predict period sequence with model1 ---
    with torch.inference_mode():
        period_prediction = model1(I_stim_PRNN, hidden_state_prnn, return_hidden = False)  # shape: (1, time_steps)
        I_cc = (0.1 / period_prediction).unsqueeze(-1).to(device)      

        for t in range(time_steps_ms):
            # broadcast the initial conditions to the grid on the projected plane
            i_proj[:, :, 0, :2] = torch.stack([grid_c0, grid_c1], dim=2)
            # transform back to the original RNN space and forward one step
            i_orig = _transform_tensor(i_proj, trained_pca, inverse=True).to(device)

            # one-step inputs
            i_stim_t = I_stim_ZRNN[:, t].unsqueeze(0).unsqueeze(-1).to(device)
            i_cc_t   = I_cc[:, t].unsqueeze(-1).to(device)

            _, hiddens = model2(i_stim_t.cpu(), i_cc_t.cpu(), i_orig.cpu())
            
            # extract tensor from a list of one
            i_orig = hiddens[-1]
        
            # Project back to the PC plane to get i_proj at t + 1, calculate the gradient, record and reassign
            i_proj_tp1 = _transform_tensor(i_orig, trained_pca)
            grad = (i_proj_tp1 - i_proj.cpu())
            dc0.append(grad[:, :, 0, 0].numpy()), dc1.append(grad[:, :, 0, 1].numpy())
            speeds.append(np.squeeze(norm(grad, axis=3), axis=2))
            i_proj = i_proj_tp1.to(device)

    out_slice = slice(None) if at_time is None else at_time
    return {"dc0": np.stack(dc0)[out_slice],
            "dc1": np.stack(dc1)[out_slice],
            "speed": np.stack(speeds)[out_slice]}



def _transform_tensor(tensor:torch.Tensor, pca: PCA, inverse: bool = False) -> torch.Tensor:
    # Performs PCA transformation on arbitrary tensor of vectors in the RNN / projected plane
    # (as sklearn is incompatible with tensors)
    size = tensor.shape
    shrank = [size[0] * size[0], size[-1]]
    out_func = lambda f: torch.tensor(f(tensor.reshape(shrank).cpu().detach().numpy()), dtype=torch.float32).reshape(size)
    return out_func(pca.transform) if not inverse else out_func(pca.inverse_transform)