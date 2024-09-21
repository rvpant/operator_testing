## code oiginally from https://github.com/SpaceLearner/JacobiKAN.git
## I slightly modified the original code in order for it to work on our problem beyond the MNIST set
## Several things we may need to play with in order to find the optimal setup for our problems, listed below
## 1. see if activation other than tanh should be used for best performance
## 2. not sure if we still need to do LayerNorm for our problems, I'd suggest try having it and then remove it, see which one is better
## 3. play a little bit with the polynomial degree and see which works best, I am setting it to 4 now but no reason that is gonna work
## 4. other changes you can think of if needed, such as changing the values for a,b......


import torch
import torch.nn as nn
import numpy as np

# This is inspired by Kolmogorov-Arnold Networks but using Jacobian polynomials instead of splines coefficients
class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim   = output_dim
        self.a        = a
        self.b        = b
        self.degree   = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0: ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a-self.b) + (self.a+self.b+2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k  = (2*i+self.a+self.b)*(2*i+self.a+self.b-1) / (2*i*(i+self.a+self.b))
            theta_k1 = (2*i+self.a+self.b-1)*(self.a*self.a-self.b*self.b) / (2*i*(i+self.a+self.b)*(2*i+self.a+self.b-2))
            theta_k2 = (i+self.a-1)*(i+self.b-1)*(2*i+self.a+self.b) / (i*(i+self.a+self.b)*(2*i+self.a+self.b-2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :, i - 2].clone()  # 2 * x * jacobi[:, :, i - 1].clone() - jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

class GeneralJacobiKAN(nn.Module):
    def __init__(self, layer_dims, degree=4, a=1.0, b=1.0, norm=False):
        """
        layer_dims: List of integers representing the dimensions of each layer.
                    e.g., [input_dim, hidden_dim1, hidden_dim2, ..., output_dim]
        degree: Degree of the Jacobi polynomials.
        a, b: Parameters for the Jacobi polynomials.
        """
        super(GeneralJacobiKAN, self).__init__()
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            self.layers.append(JacobiKANLayer(layer_dims[i], layer_dims[i + 1], degree, a, b))
            if norm and (i < len(layer_dims) - 2):  # No LayerNorm after the last layer
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
    
    model = GeneralJacobiKAN(layer_dims, degree)
    x = torch.randn(8, input_dim)
    output = model(x)
    print("Output shape:", output.shape)
    
    expected_output_dim = layer_dims[-1]
    assert output.shape == (8, expected_output_dim), f"Expected shape (8, {expected_output_dim}), but got {output.shape}"

    print("Test passed!")