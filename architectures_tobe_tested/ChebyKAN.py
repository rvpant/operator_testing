## code roiginally from https://github.com/SynodicMonth/ChebyKAN.git
## I slightly modified the original code in order for it to work on our problem beyond the MNIST set
## Several things we may need to play with in order to find the optimal setup for our problems, listed below
## 1. see if activation other than cos and acos should be used for best performance
## 2. not sure if we still need to do LayerNorm for our problems, I'd suggest try having it and then remove it, see which one is better
## 3. play a little bit with the polynomial degree and see which works best, I am setting it to 4 now but no reason that is gonna work
## 4. other changes you can think of if needed......



import torch
import torch.nn as nn



# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y
        
class GeneralChebyKAN(nn.Module):
    def __init__(self, layer_dims, degree = 4):
        """
        layer_dims: List of integers representing the dimensions of each layer.
                    e.g., [input_dim, hidden_dim1, hidden_dim2, ..., output_dim]
        degree: Degree of the Chebyshev polynomials.
        """
        # do we need LayerNorm at all?
        super(GeneralChebyKAN, self).__init__()
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            self.layers.append(ChebyKANLayer(layer_dims[i], layer_dims[i + 1], degree))
            if i < len(layer_dims) - 2: 
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
    
    model = GeneralChebyKAN(layer_dims, degree)
    x = torch.randn(8, input_dim)
    output = model(x)
    print("Output shape:", output.shape)
    
    expected_output_dim = layer_dims[-1]
    assert output.shape == (8, expected_output_dim), f"Expected shape (8, {expected_output_dim}), but got {output.shape}"

    print("Test passed!")