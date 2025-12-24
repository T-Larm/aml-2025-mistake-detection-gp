import torch
from torch import nn

from core.models.blocks import fetch_input_dim, MLP


class ErLSTM(nn.Module):
    """
    LSTM-based baseline for error recognition.
    Processes the sequence of sub-segments within each step.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        input_dimension = fetch_input_dim(config)

        # LSTM parameters
        self.hidden_size = 512
        self.num_layers = 2
        self.bidirectional = True
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dimension,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.3 if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Output dimension after LSTM
        lstm_output_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # MLP decoder for final classification
        self.decoder = MLP(lstm_output_dim, 256, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_data):
        # Check for NaNs in input and replace them with zero
        input_data = torch.nan_to_num(input_data, nan=0.0, posinf=1.0, neginf=-1.0)

        # If input is 2D (batch_size, features), add a sequence dimension
        if len(input_data.shape) == 2:
            input_data = input_data.unsqueeze(1)  # (batch_size, 1, features)
        
        # LSTM forward pass
        # input_data shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(input_data)
        
        # Use the last hidden state for classification
        # For bidirectional LSTM, concatenate forward and backward hidden states
        if self.bidirectional:
            # hidden shape: (num_layers * 2, batch_size, hidden_size)
            # Take the last layer's forward and backward hidden states
            forward_hidden = hidden[-2, :, :]  # (batch_size, hidden_size)
            backward_hidden = hidden[-1, :, :]  # (batch_size, hidden_size)
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch_size, hidden_size * 2)
        else:
            # hidden shape: (num_layers, batch_size, hidden_size)
            final_hidden = hidden[-1, :, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        final_hidden = self.dropout(final_hidden)
        
        # Decode to get final prediction
        output = self.decoder(final_hidden)

        return output


class ErGRU(nn.Module):
    """
    GRU-based baseline for error recognition.
    Alternative to LSTM with fewer parameters.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        input_dimension = fetch_input_dim(config)

        # GRU parameters
        self.hidden_size = 512
        self.num_layers = 2
        self.bidirectional = True
        
        # GRU encoder
        self.gru = nn.GRU(
            input_size=input_dimension,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.3 if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Output dimension after GRU
        gru_output_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # MLP decoder for final classification
        self.decoder = MLP(gru_output_dim, 256, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_data):
        # Check for NaNs in input and replace them with zero
        input_data = torch.nan_to_num(input_data, nan=0.0, posinf=1.0, neginf=-1.0)

        # If input is 2D (batch_size, features), add a sequence dimension
        if len(input_data.shape) == 2:
            input_data = input_data.unsqueeze(1)  # (batch_size, 1, features)
        
        # GRU forward pass
        # input_data shape: (batch_size, seq_len, input_dim)
        gru_out, hidden = self.gru(input_data)
        
        # Use the last hidden state for classification
        # For bidirectional GRU, concatenate forward and backward hidden states
        if self.bidirectional:
            # hidden shape: (num_layers * 2, batch_size, hidden_size)
            # Take the last layer's forward and backward hidden states
            forward_hidden = hidden[-2, :, :]  # (batch_size, hidden_size)
            backward_hidden = hidden[-1, :, :]  # (batch_size, hidden_size)
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch_size, hidden_size * 2)
        else:
            # hidden shape: (num_layers, batch_size, hidden_size)
            final_hidden = hidden[-1, :, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        final_hidden = self.dropout(final_hidden)
        
        # Decode to get final prediction
        output = self.decoder(final_hidden)

        return output
