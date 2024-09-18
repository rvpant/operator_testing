import torch
import torch.nn as nn

class LegendreKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(LegendreKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.legendre_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.legendre_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        x = torch.tanh(x)  # Normalize input to [-1, 1] for stability in Legendre polynomial calculation

        # Initialize Legendre polynomial tensors
        legendre = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        legendre[:, :, 0] = 1  # P_0(x) = 1
        if self.degree > 0:
            legendre[:, :, 1] = x  # P_1(x) = x

        # Compute Legendre polynomials using the recurrence relation
        for n in range(2, self.degree + 1):
           # Recurrence relation without in-place operations
            legendre[:, :, n] = ((2 * (n-1) + 1) / (n)) * x * legendre[:, :, n-1].clone() - ((n-1) / (n)) * legendre[:, :, n-2].clone()

        # Compute output using matrix multiplication
        y = torch.einsum('bid,iod->bo', legendre, self.legendre_coeffs)
        y = y.view(-1, self.outdim)
        return y
        
        
class GeneralLegendreKAN(nn.Module):
    def __init__(self, layer_dims, degree=4, norm=False):
        """
        layer_dims: List of integers representing the dimensions of each layer.
                    e.g., [input_dim, hidden_dim1, hidden_dim2, ..., output_dim]
        degree: Degree of the Legendre polynomials.
        """
        ## do we need layerNorm at all for our problems?s
        super(GeneralLegendreKAN, self).__init__()
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            self.layers.append(LegendreKANLayer(layer_dims[i], layer_dims[i + 1], degree))
            if norm and (i < len(layer_dims) - 2):  
                self.norm_layers.append(nn.LayerNorm(layer_dims[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.norm_layers):
                x = self.norm_layers[i](x)
        return x
        
if __name__ == '__main__':
    input_dim = 64
    layer_dims = [64, 32, 16, 10]  # Custom layer dimensions
    degree = 4
    
    model = GeneralLegendreKAN(layer_dims, degree)
    x = torch.randn(8, input_dim)
    output = model(x)
    print("Output shape:", output.shape)
    
    expected_output_dim = layer_dims[-1]
    assert output.shape == (8, expected_output_dim), f"Expected shape (8, {expected_output_dim}), but got {output.shape}"

    print("Test passed!")