# QMUL-MSc-Research-Project

## Neural Clockworks: Decoding the Mechanisms of Temporal Dynamics with Recurrent Networks

## Problem & Motivation
Rhythmic timing is fundamental to human cognition, influencing everything from speech perception to musical performance. However, the neural mechanisms underlying how the brain maintains and perceives rhythm remain poorly understood. Prior work often relied on rule-based or entrainment-based models, which fail to capture the complexity of biological rhythmicity [[1]](#ref1).

The original Zemlianova RNN (ZRNN) model [[2]](#ref2) replicated oscillatory dynamics of rhythmic timing but required an external "context cue" signal, limiting its biological plausibility. This project aimed to address this gap by developing a fully integrated recurrent neural network capable of internally generating the necessary context cues for rhythmic perception.

## Approach
The project developed and integrated two key components:

**Period-Predicting RNN (PRNN):**
* A Long Short-Term Memory (LSTM)-based architecture trained to infer the period of an input rhythm from its impulse timing.
* Outputs a signal that predicts the periodicity of the input, enabling internal generation of the context cue.

**Integrated ZRNN (IZRNN):**
* Combines the PRNN with the original ZRNN to create a fully self-sustaining model.
* Requires only the rhythm's impulse times as input, eliminating the need for external context cues.

**Key technical choices included:**
* Using LSTM architecture for its superior long-term memory capabilities in handling variable-length sequences.
* Implementing biological constraints such as excitatory/inhibitory neuron dynamics to align with neurophysiological findings.
* Employing PCA and raster plot analysis to study hidden layer dynamics and validate biological plausibility.

## Results
The project achieved several key results, visualized through detailed figures and analyses:

### PRNN Performance
**Training & Validation Loss:** Tracking training and validation losses (calculated using mean-squared error) revealed how well the PRNN model learned patterns and generalized to different beat frequencies. The loss curves indicated effective learning without signs of overfitting.

**Key Metrics:** Mean Absolute Error = 0.0159, R² coefficient = 0.9972.

![Line graph showing training and validation losses over time (epochs).](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/training_validation_loss_period_prediction_lstm.png)

**Predicted vs True Period:** Scatter plots showed PRNN predictions against true values, with a fitted regression line closely aligning with the ideal prediction line (y=x). This demonstrated strong predictive accuracy across different periods.

![Scatter graph showing predictions vs actual value.](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/test_true_vs_predicted_period.png)

**Predicted Period Over Time:** Line graphs mapped out the PRNN's predicted vs true period for six samples throughout their full duration. These visualizations revealed consistent tracking of periodicity, even during extended continuation phases.

![Line graphs showing prediction of the period for 6 samples over the whole sample duration](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/period_comparison_overlay.png)

### Neuron Activation Analysis
Raster plots provided insights into hidden layer dynamics:

* Color-coded positive and negative activations highlighted rhythmic patterns during both synchronisation (input pulses) and continuation phases (no input).
* Example raster plots for periods of 0.2s, 0.6s, and 0.95s demonstrated how the model maintained stable internal representations across different temporal scales.

### PCA Trajectory Analysis
Principal Component Analysis (PCA) revealed the low-dimensional structure of hidden layer dynamics:

![Raster plot showing neuron activations of the hidden layer on a sample with a period of 0.2](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/neuron_raster_peaks_period_0.2.png)
![Raster plot showing neuron activations of the hidden layer on a sample with a period of 0.6](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/neuron_raster_peaks_period_0.6.png)
![Raster plot showing neuron activations of the hidden layer on a sample with a period of 0.95](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/neuron_raster_peaks_period_0.95.png)

### 2D & 3D PCA Plots:

* Colored trajectories showed stimulus frequency-dependent patterns, with black dots marking tap times.
* Yellow and red dots marked trajectory start and end points, respectively.
* Dashed lines represented synchronisation phase trajectories, while solid lines showed continuation phase dynamics.

<table style="width: 100%; border: none;">
  <tr>
    <td style="width: 33%; text-align: center; border: none; padding: none;">
      <img src="https://raw.githubusercontent.com/HA-141/QMUL-MSc-Research-Project/main/images/pca_trajectories_2d_vibrant.png" alt="2D PCA of neuronal trajectories" width="400" height="400">
    </td>
    <td style="width: 33%; text-align: center; border: none; padding: none;">
      <img src="https://raw.githubusercontent.com/HA-141/QMUL-MSc-Research-Project/main/images/pca_trajectories_3d_vibrant.png" alt="3D PCA of neuronal trajectories" width="400" height="400">
    </td>
  </tr>
</table>

### Mean vs Standard Deviation of Trajectories:

Line graphs displayed the relationship between mean trajectory length and standard deviation for different stimulus periods, highlighting consistent internal representations across frequencies.

<img src="https://raw.githubusercontent.com/HA-141/QMUL-MSc-Research-Project/main/images/trajectory_mean_vs_std_vibrant.png" alt="Comparison of mean vs standard deviation of trajectory length" width="400" height="400">

### IZRNN Dynamics
The integrated model replicated key features of the original ZRNN while operating with internally generated context cues.

![Firing activity of the IZRNN](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/IZRNN%20Activity%20map.png)

Hidden layer activity exhibited oscillatory patterns consistent with biological neural populations, including distinct excitatory and inhibitory subpopulations.

![ZRNN hidden layer of IZRNN portion and context cue size vs its inter-peak interval](https://github.com/HA-141/QMUL-MSc-Research-Project/blob/main/images/IZRRN%20pca.png)

### Limitations
* The PRNN exhibited a gradual overestimation drift during continuation phases, contrasting with biological underestimation trends.
* Current implementation focuses on isochronous rhythms; extension to more complex patterns remains an open challenge.

### Outcome
This work successfully addressed the limitation of the original ZRNN by developing a fully integrated model capable of internally generating context cues for rhythmic perception. The results demonstrate that:

**Biological Plausibility:**
* The IZRNN's dynamics align with known neural mechanisms, suggesting it could serve as a valuable tool for studying rhythmic timing in biological systems.

**Technical Advancements:**
* The integration of PRNN and ZRNN demonstrates the feasibility of creating distributed neural networks that mimic biological temporal processing.

**Future Directions:**
* Addressing the overestimation drift and extending the model to handle more complex rhythms are critical next steps.
* Exploring fully integrated training processes could improve biological realism and predictive accuracy.

## Setup & Usage

### Dependencies
Required libraries include:

* PyTorch (2.7.0+cu118)
* NumPy (2.2.6)
* SciPy (1.15.3)
* Matplotlib (3.10.3)
* Scikit-learn (1.6.1)

### Directory Structure

```bash
.
├── zrnn/
│   ├── models.py
│   ├── datasets.py
│   └── utils_IZRNN.py
├── config_IZRNN.yaml
├── config_PRNN.yaml
├── trainPRNN.py
├── testPRNN.py
├── trainZRNN.py
├── drive_model_and_save_outputs.py
├── plot_IZRNN_hidden_activity.py
└── plot_IZRNN_temporal_dynamics.py
```

## Training & Testing

### Train PRNN

```bash
python trainPRNN.py
```

### Test PRNN

```bash
python testPRNN.py
```

### Train ZRNN

```bash
python trainZRNN.py
```

### Generate & Visualize Outputs

```bash
python drive_model_and_save_outputs.py
```

### Plot Hidden Activity

```bash
python plot_IZRNN_hidden_activity.py
```

### Plot Temporal Dynamics (GPU Support)

```bash
python plot_IZRNN_temporal_dynamics.py --device cuda
```

> **Note:** If you do not have a CUDA-compatible GPU, omit the `--device cuda` flag or replace it with `--device cpu`.

---

## Conclusion

This project represents a significant step forward in understanding the neural mechanisms of rhythmic timing. By developing an integrated recurrent neural network capable of internally generating context cues, the model more closely aligns with biologically plausible neural computation.

The results demonstrate the feasibility of using deep learning to study complex temporal dynamics while providing a foundation for future research into rhythm perception, temporal prediction, and motor timing.

The accompanying codebase and documentation are intended to enable researchers to reproduce, validate, and extend this work, contributing to a deeper understanding of the neural basis of rhythmic timing.

---

## References

1. <a id="ref1"></a>Zemlianova, K., Bose, A., & Rinzel, J. (2024). *A Recurrent Neural Network for Rhythmic Timing*.  
   https://www.biorxiv.org/content/10.1101/2024.05.24.595797v1.abstract
2. <a id="ref2"></a>Roman, I. *Original ZRNN implementation*.  
   https://github.com/iranroman/ZemlianovaRNN
3. Relevant neurophysiological studies on rhythmic timing and temporal perception.




