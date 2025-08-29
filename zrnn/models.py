import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, return_hidden = False):      

        #remove padding
        # Step 1: Calculate lengths (number of non-padding tokens per sequence)
        True_lengths = (x != -1).sum(dim=1)

        True_lengths = True_lengths.to(dtype = torch.long).cpu()  # Ensure shape [batch_size] and type Long

        True_lengths = True_lengths.squeeze(1)

        # Step 2: Replace padding tokens (-1) with 0
        x_clean = x.clone()
        x_clean[x_clean == -1] = 0

        # Step 3: pack_padded seq
        packed_x = pack_padded_sequence(x_clean, lengths=True_lengths, batch_first=True, enforce_sorted=False)

        # Step 4: Forward through RNN
        # LSTM outputs (output for each time step, and final hidden/cell states)
        outputs, (h_n, c_n) = self.lstm(packed_x, hidden)
        
        # Step 5: Unpack output sequence:
        padded_out, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=-1)

        # We take the output from the last time step and pass it through the linear layer
        period_prediction = self.fc(padded_out)  #h_n[-1])

        if return_hidden:
            return period_prediction, padded_out, (h_n, c_n) 
        return period_prediction
    
    def initHidden(self, batch_size):
        # LSTM returns (h_0, c_0), each of shape (num_layers, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device).contiguous()
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device).contiguous()
        return (h_0, c_0)

class ZemlianovaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dt, tau, excit_percent, sigma_rec=0.01, sigma_input = 0.01, isTraining = True):
        super().__init__()
        self.hidden_size = hidden_dim
        self.dt = dt
        self.tau = tau
        self.excit_percent = excit_percent
        self.sigma_rec = sigma_rec
        self.sigma_input = sigma_input
        self.training = isTraining

        # Initialize weights and biases for input to hidden layer
        self.w_ih = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_dim))

        # Initialize weights and biases for hidden to hidden layer
        self.w_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_dim))

        # Initialize weights and biases for hidden to output layer
        self.w_ho = nn.Parameter(torch.Tensor(output_dim, hidden_dim))
        self.b_ho = nn.Parameter(torch.Tensor(output_dim))

        # Initialize all weights and biases
        self.init_weights()

        # Create masks for zeroing diagonal of w_hh and contain EI ratio
        self.zero_diag_mask = torch.ones(hidden_dim, hidden_dim) - torch.eye(hidden_dim)
        self.EI_mask = torch.ones(hidden_dim).to(device)
        self.EI_mask[int(self.excit_percent * hidden_dim):] = -1


    def init_weights(self):
        # Initialize weights using xavier uniform and biases to zero
        nn.init.xavier_normal_(self.w_ih)
        nn.init.constant_(self.b_ih, 0)
        nn.init.xavier_normal_(self.w_hh, gain=0.1)
        self.w_hh = nn.Parameter(F.relu(self.w_hh))
        nn.init.constant_(self.b_hh, 0)
        nn.init.xavier_normal_(self.w_ho)
        nn.init.constant_(self.b_ho, 0)

    def forward(self, i_stim, i_cc, hidden):

        outputs = []
        hiddens = [] ## new

        inputs = torch.stack([i_stim, i_cc], dim=-1)

        for t in range(inputs.size(1)):  # process each time step

            current_input_step = inputs[:, t, :]

            # add noise to the input
            input_noise = torch.sqrt(torch.tensor(2 * (self.dt / self.tau) * (self.sigma_input ** 2))) * torch.randn_like(current_input_step)            
            if self.training:
                current_input_step = F.relu(current_input_step + input_noise)

            # compute recurrent noise
            rec_noise = torch.sqrt(torch.tensor(2 * (self.tau / self.dt) * (self.sigma_rec ** 2))) * torch.randn_like(hidden)

            # Zero out diagonal of w_hh each forward pass
            w_hh_no_diag = self.w_hh * self.zero_diag_mask.to(self.w_hh.device)
            w_hh_no_diag_p = torch.abs(w_hh_no_diag)
            w_hh_EI = w_hh_no_diag_p * self.EI_mask.to(self.w_hh.device)

            # Compute the hidden state
            hidden_input = torch.matmul(current_input_step, self.w_ih.t())
            hidden_hidden = torch.matmul(F.relu(hidden), w_hh_EI.t()) + self.b_hh + rec_noise
            hidden = hidden + (-hidden + hidden_input + hidden_hidden)  * (self.dt / self.tau)

            # Compute the output
            output = torch.matmul(F.relu(hidden), self.w_ho.t()) + self.b_ho
            outputs.append(output)
            hiddens.append(hidden) 

        return outputs, hiddens

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)



