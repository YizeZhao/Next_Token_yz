# Model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Existing imports and classes...

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Optional: Add normalization or other layers if needed
        self.norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        self._init_weights()

        # Assign a name for identification
        self.name = "LSTM"

    def _init_weights(self):
        # Initialize weights for LSTM and FC layers
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
            elif 'fc' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
            y (Optional[torch.Tensor]): Target tensor for supervised training.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, output_size).
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Apply normalization (optional)
        out = self.norm(out)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # out: tensor of shape (batch_size, output_size)

        if y is not None:
            # Compute loss
            loss = F.cross_entropy(out, y, reduction='sum')
            self.last_loss = loss
        else:
            self.last_loss = None

        return out

    def forward_embedding(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract embeddings from the LSTM model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            np.ndarray: Embedding matrix.
        """
        with torch.no_grad():
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.norm(out)
            embeddings = out[:, -1, :].cpu().numpy()
        return embeddings
