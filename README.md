# QMUL-MSc-Research-Project
A modified ZRNN model based on [Iran Roman's ZRNN model](https://github.com/iranroman/ZemlianovaRNN)

This is a PyTorch implementation of the model described in the 2024 paper ["A Recurrent Neural Network for Rhythmic Timing"](https://www.biorxiv.org/content/10.1101/2024.05.24.595797v1.abstract) by Klavdia Zemlianova, Amit Bose, & John Rinzel [1]. 

The model has been adapted from Iran Roman's model to explore and demonstrate the neural mechanisms behind the generation of the context cue signal needed for internal rhythmic timing and see if it can also replicate similar patterns.

### Features: 
- A model, called the Period Predicting Recurrent Neural Network (PRNN) is used to count the period of a signal at each timestep, with its output being processed to produce the context cue signal
- A modified ZRNN (IZRNN) that integrates the PRNN with the ZRNN, enabling the whole model to use a single input (stimulus impulses) to give similar results and neural mechanisms to rhythmic timing
- Model paramters for both ZRNN and PRNN models can be configured to using their respective YAML files
- Scripts available to train and test ZRNN and PRNN models separately and then test the integrated IZRNN with the separately trained models
- GPU support available through CUDA, to enable efficient training and performance evaluations based on hardware

Python Libraries used: 
PyTorch: 2.7.0+cu118
NumPy: 2.2.6
Pandas: 2.2.3
SciPy: 1.15.3
Matplotlib: 3.10.3
Scikit-learn: 1.6.1
PyYAML: 6.0.2

You can install these libraries using this prompt:

```python
pip install -r requirements.txt
```


Consult ReadMe.txt to learn the order to how to run scripts

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

**PCA Trajectory analysis:**
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
  author={Zemlianova, Klavdia and Bose, Amitabha and Rinzel, John},
  journal={bioRxiv},
  pages={2024--05},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}


