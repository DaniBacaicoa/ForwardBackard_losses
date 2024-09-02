import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_p=0.0, bn = False, activation='relu'):
        super().__init__()

        # Create a list of layer sizes
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create a list of linear layers using ModuleList
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes)-1)
        ])
        for layer in self.layers:
            #nn.init.xavier_uniform_(layer.weight)
            nn.init.constant(layer.weight, 0.1)

        # Create a list of batch normalization layers using ModuleList
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(layer_sizes[i+1])
            for i in range(len(hidden_sizes))
        ])

        # Create a dropout layer
        self.dropout = nn.Dropout(dropout_p)
        self.bn = bn
        self.activation = activation

    def forward(self, x):
        # Iterate over the linear layers and apply them sequentially to the input
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.bn:
                x = self.batch_norms[i](x)
            activation_fn = getattr(nn.functional, self.activation)
            x = activation_fn(x)
            x = self.dropout(x)
        # Apply the final linear layer to get the output
        x = self.layers[-1](x)
        return x
