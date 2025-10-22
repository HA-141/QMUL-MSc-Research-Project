# QMUL-MSc-Research-Project
A modified ZRNN model based on [Iran Roman's ZRNN model](https://github.com/iranroman/ZemlianovaRNN)

This is a PyTorch implementation of the model described in the 2024 paper ["A Recurrent Neural Network for Rhythmic Timing"](https://www.biorxiv.org/content/10.1101/2024.05.24.595797v1.abstract) by Klavdia Zemlianova, Amit Bose, & John Rinzel. 

The model has been adapted from Iran Roman's model to explore and demonstrate the neural mechanisms behind the generation of the context cue signal needed for internal rhythmic timing and see if it can also replicate similar patterns.

Features: 
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
bash`pip install -r requirements.txt`


Consult ReadMe.txt to learn how to run scripts


