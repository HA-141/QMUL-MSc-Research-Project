# QMUL-MSc-Research-Project

## Neural Clockworks: Decoding the Mechanisms of Temporal Dynamics with Recurrent Networks

### Problem & Motivation
Rhythmic timing is fundamental to human cognition, influencing everything from speech perception to musical performance. However, the neural mechanisms underlying how the brain maintains and perceives rhythm remain poorly understood. Prior work often relied on rule-based or entrainment-based models, which fail to capture the complexity of biological rhythmicity [1].

The original Zemlianova RNN (ZRNN) model [2] replicated oscillatory dynamics of rhythmic timing but required an external "context cue" signal, limiting its biological plausibility. This project aimed to address this gap by developing a fully integrated recurrent neural network capable of internally generating the necessary context cues for rhythmic perception.

### Approach
The project developed and integrated two key components:

Period-Predicting RNN (PRNN):

A Long Short-Term Memory (LSTM)-based architecture trained to infer the period of an input rhythm from its impulse timing.
Outputs a signal that predicts the periodicity of the input, enabling internal generation of the context cue.
Integrated ZRNN (IZRNN):

Combines the PRNN with the original ZRNN to create a fully self-sustaining model.
Requires only the rhythm's impulse times as input, eliminating the need for external context cues.
Key technical choices included:

Using LSTM architecture for its superior long-term memory capabilities in handling variable-length sequences.
Implementing biological constraints such as excitatory/inhibitory neuron dynamics to align with neurophysiological findings.
Employing PCA and raster plot analysis to study hidden layer dynamics and validate biological plausibility.


### Visualising PRNN model outputs

**Tracking training and validation loss:** training and validation losses (calculated using mean-squared error) tracked how well the PRNN model learned the patterns and generalise to different beat-frequencies. Also allows us to verify whether the model is under/over-fitting.

![Line graph showing training and validation losses over time (epochs).](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/training_validation_loss_period_prediction_lstm.png)

**Predicted vs True time period of a beat:** shows predictions of PRNN after testing, with a fitted regression line that can be compared against the ideal prediction line

![Scatter graph showing predictions vs actual value.](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/test_true_vs_predicted_period.png)

**Predicted period of a sample throughout whole duration:** maps out the PRNN's predicted vs true period of six samples during the whole duration of the sample to analyse change in prediction over time

![Line graphs showing prediction of the period for 6 samples over the whole sample duration](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/test_true_vs_predicted_period.png)

**Neuron activation analysis using raster plots:** raster plots showing neuron activity for the hidden layer. Colour coded to visualise positive and negative activations during synchronisation and continuation phase of the sample and samples with a period of 0.2 (Top), 0.6 (Middle), 0.95 (Bottom).

![Raster plot showing neuron activations of the hidden layer on a sample with a period of 0.2](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/neuron_raster_peaks_period_0.2.png)
![Raster plot showing neuron activations of the hidden layer on a sample with a period of 0.6](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/neuron_raster_peaks_period_0.6.png)
![Raster plot showing neuron activations of the hidden layer on a sample with a period of 0.95](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/neuron_raster_peaks_period_0.95.png)

**PCA Trajectory analysis**:
- **Left:** 2D PCA plot showing the trajectories of the PRNN's hidden layer neurons that is colour coded by stimulus frequency (from lowest to highest). Tap signals represented as black dots, yellow dots are the start of the sample trajectory, red dots represent end of sample trajectory. Dashed lines represent synchronisation phase and solid lines represent continuation phase.
- **Middle:** 3D PCA plot showing same trajectories as the 2D plot but in a higher dimensional space.
- **Right:** Line-graph displaying the relationship of the mean and standard-deviation of the hidden layer trajectories for each stimulus period.

<table style="width: 100%; border: none;">
  <tr>
    <td style="width: 33%; text-align: center; border: none; padding: none;">
      <img src="https://raw.githubusercontent.com/HA-141/QMUL-MSc-Research-Project/main/images/pca_trajectories_2d_vibrant.png" alt="2D PCA of neuronal trajectories" width="400" height="400">
    </td>
    <td style="width: 33%; text-align: center; border: none; padding: none;">
      <img src="https://raw.githubusercontent.com/HA-141/QMUL-MSc-Research-Project/main/images/pca_trajectories_3d_vibrant.png" alt="3D PCA of neuronal trajectories" width="400" height="400">
    </td>
    <td style="width: 34%; text-align: center; border: none; padding: none;">
      <img src="https://raw.githubusercontent.com/HA-141/QMUL-MSc-Research-Project/main/images/trajectory_mean_vs_std_vibrant.png" alt="Comparison of mean vs standard deviation of trajectory length" width="400" height="400">
    </td>
  </tr>
</table>


### Reference
[1] @article{zemlianova2024recurrent,
  title={A Recurrent Neural Network for Rhythmic Timing},
  author={Klavdia Zemlianova, Amitabha Bose, John Rinzel},
  journal={bioRxiv},
  year={2024},
  elocation-id={2024.05.15.594411},
  doi={10.1101/2024.05.15.594411},
  publisher={Cold Spring Harbor Laboratory}
}


