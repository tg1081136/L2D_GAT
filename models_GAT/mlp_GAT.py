import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, activation='tanh', norm_type='layernorm'):
        super(MultiLayerPerceptron, self).__init__()
        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")

        self.activation = F.tanh if activation == 'tanh' else F.relu
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norm_type = norm_type.lower()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

        for _ in range(num_layers - 1):
            if self.norm_type == 'batchnorm':
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == 'layernorm':
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.norms.append(nn.Identity())

    def forward(self, x):
        for layer, norm in zip(self.layers[:-1], self.norms):
            x = layer(x)
            x = norm(x) if self.norm_type != 'none' else x
            x = self.activation(x)
            if torch.isnan(x).any():
                print(f"NaN detected in layer output: {x}")
        x = self.layers[-1](x)  # 最後一層無激活
        if torch.isnan(x).any():
            print(f"NaN detected in final output: {x}")
        return x