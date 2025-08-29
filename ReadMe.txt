IZRNN Scripts
This directory contains the necessary scripts for training, testing, and visualizing the IZRNN model.

Directory Structure
To ensure the scripts run correctly, organise the directory like so:

.
├── zrnn/
│   ├── datasets.py
│   ├── models.py
│   └── utils_IZRNN.py
├── config_IZRNN.yaml
├── config_PRNN.yaml
├── trainPRNN.py
├── testPRNN.py
├── trainZRNN.py
├── drive_model_and_save_outputs.py
├── plot_IZRNN_hidden_activity.py
└── plot_IZRNN_temporal_dynamics.py

Running the Scripts
Follow the steps below to train, test, and analyze the model. It's crucial to run them in the specified order.

Train and Test the PRNN layer:

python trainPRNN.py
python testPRNN.py

Train the ZRNN layer

python trainZRNN.py

Generate and Visualize Outputs for the IZRNN model:

python drive_model_and_save_outputs.py
python plot_IZRNN_hidden_activity.py
python plot_IZRNN_temporal_dynamics.py

You can run plot_IZRNN_temporal_dynamics.py with GPU:

plot_IZRNN_temporal_dynamics.py --device cuda


