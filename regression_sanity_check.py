import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from architectures_tobe_tested.ChebyKAN import ChebyKANLayer
from architectures_tobe_tested.JacobiKAN import JacobiKANLayer
from architectures_tobe_tested.LegendreKAN import LegendreKANLayer
from efficient_kan.kan import KANLinear 


# Define target function
def target_function(x):
    y = np.zeros_like(x)
    mask1 = x < 0.5
    y[mask1] = np.sin(20 * np.pi * x[mask1]) + x[mask1] ** 2
    mask2 = (0.5 <= x) & (x < 1.5)
    y[mask2] = 0.5 * x[mask2] * np.exp(-x[mask2]) + np.abs(np.sin(5 * np.pi * x[mask2]))
    mask3 = x >= 1.5
    y[mask3] = np.log(x[mask3] - 1) / np.log(2) - np.cos(2 * np.pi * x[mask3])

    # add noise
    noise = np.random.normal(0, 0.2, y.shape)
    y += noise
    
    return y

# Define MLP and ChebyKAN
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.layers(x-1) # centralize the input


class ChebyKAN(nn.Module):
    def __init__(self):
        super(ChebyKAN, self).__init__()
        self.chebykan1 = ChebyKANLayer(1, 8, 8)
        self.chebykan2 = ChebyKANLayer(8, 1, 8)

    def forward(self, x):
        x = self.chebykan1(x)
        x = self.chebykan2(x)
        return x

class JacobiKAN(nn.Module):
    def __init__(self):
        super(JacobiKAN, self).__init__()
        self.jacobikan1 = JacobiKANLayer(1, 8, 8)
        self.jacobikan2 = JacobiKANLayer(8, 1, 8)

    def forward(self, x):
        x = self.jacobikan1(x)
        x = self.jacobikan2(x)
        return x

class LegendreKAN(nn.Module):
    def __init__(self):
        super(LegendreKAN, self).__init__()
        self.legendrekan1 = LegendreKANLayer(1, 8, 8)
        self.legendrekan2 = LegendreKANLayer(8, 1, 8)

    def forward(self, x):
        x = self.legendrekan1(x)
        x = self.legendrekan2(x)
        return x

class layer2EfficientKAN(nn.Module):
    def __init__(self):
        super(layer2EfficientKAN, self).__init__()
        self.kan1 = KANLinear(1,8,grid_size=5,spline_order=3)
        self.kan2 = KANLinear(8,1,grid_size=5,spline_order=3)

    def forward(self, x):
        x = self.kan1(x)
        x = self.kan2(x)
        return x
        
        
        
# Generate sample data
x_train = torch.linspace(0, 2, steps=500).unsqueeze(1)
y_train = torch.tensor(target_function(x_train))

# Instantiate models
cheby_model = ChebyKAN()
ekan_model = layer2EfficientKAN()
mlp_model = SimpleMLP()
jacobi_model = JacobiKAN()  # New model: Jacobikan
legendre_model = LegendreKAN()  # New model: LegendreKAN

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer_cheby = torch.optim.Adam(cheby_model.parameters(), lr=0.01)
criterion_ekan = nn.MSELoss()
optimizer_ekan = torch.optim.Adam(ekan_model.parameters(), lr=0.01)
optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=0.03)
optimizer_jacobi = torch.optim.Adam(jacobi_model.parameters(), lr=0.01)  # Optimizer for Jacobikan
optimizer_legendre = torch.optim.Adam(legendre_model.parameters(), lr=0.01)  # Optimizer for LegendreKAN

mlp_losses = []
ekan_losses = []
cheby_losses = []
jacobi_losses = []  # Losses for Jacobikan
legendre_losses = []  # Losses for LegendreKAN

# Train the models
epochs = 5000
for epoch in range(epochs):
    
    # Train ChebyKAN
    optimizer_cheby.zero_grad()
    outputs_cheby = cheby_model(x_train)
    loss_cheby = criterion(outputs_cheby, y_train)
    loss_cheby.backward()
    optimizer_cheby.step()
    
    # Train EfficientKAN
    optimizer_ekan.zero_grad()
    outputs_ekan = ekan_model(x_train)
    loss_ekan = criterion(outputs_ekan, y_train)
    loss_ekan.backward()
    optimizer_ekan.step()
    
    # Train MLP
    optimizer_mlp.zero_grad()
    outputs_mlp = mlp_model(x_train)
    loss_mlp = criterion(outputs_mlp, y_train)
    loss_mlp.backward()
    optimizer_mlp.step()
    
    # Train Jacobikan
    optimizer_jacobi.zero_grad()
    outputs_jacobi = jacobi_model(x_train)
    loss_jacobi = criterion(outputs_jacobi, y_train)
    loss_jacobi.backward()
    optimizer_jacobi.step()

    # Train LegendreKAN
    optimizer_legendre.zero_grad()
    outputs_legendre = legendre_model(x_train)
    loss_legendre = criterion(outputs_legendre, y_train)
    loss_legendre.backward()
    optimizer_legendre.step()

    if epoch % 100 == 0:
        cheby_losses.append(loss_cheby.item())
        ekan_losses.append(loss_ekan.item())
        mlp_losses.append(loss_mlp.item())
        jacobi_losses.append(loss_jacobi.item())  # Record Jacobikan loss
        legendre_losses.append(loss_legendre.item())  # Record LegendreKAN loss
        print(f'Epoch {epoch + 1}/{epochs},  ChebyKAN Loss: {loss_cheby.item():.4f}, eKAN Loss: {loss_ekan.item():.4f}, MLP Loss: {loss_mlp.item():.4f}, Jacobikan Loss: {loss_jacobi.item():.4f}, LegendreKAN Loss: {loss_legendre.item():.4f}')

# Test the models
x_test = torch.linspace(0, 2, steps=400).unsqueeze(1)
y_pred_cheby = cheby_model(x_test).detach()
y_pred_ekan = ekan_model(x_test).detach()
y_pred_mlp = mlp_model(x_test).detach()
y_pred_jacobi = jacobi_model(x_test).detach()  # Predictions for Jacobikan
y_pred_legendre = legendre_model(x_test).detach()  # Predictions for LegendreKAN

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original Data')
plt.plot(x_test.numpy(), y_pred_ekan.numpy(), 'y-', label='eKAN')
plt.plot(x_test.numpy(), y_pred_mlp.numpy(), 'g--', label='MLP')
plt.plot(x_test.numpy(), y_pred_cheby.numpy(), 'm-', label='ChebyKAN')
plt.plot(x_test.numpy(), y_pred_jacobi.numpy(), 'c-', label='Jacobikan')  # Jacobikan plot
plt.plot(x_test.numpy(), y_pred_legendre.numpy(), 'b--', label='LegendreKAN')  # LegendreKAN plot
plt.title('Comparison of ChebyKAN, EfficientKAN, MLP, Jacobikan, and LegendreKAN Interpolations f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.savefig('plot1.png',dpi=400,bbox_inches='tight')
plt.show()

# Plot the convergence speed
plt.figure(figsize=(10, 5))
plt.plot(range(0, epochs, 100), ekan_losses, 'y-', label='eKAN')
plt.plot(range(0, epochs, 100), mlp_losses, 'g--', label='MLP')
plt.plot(range(0, epochs, 100), cheby_losses, 'm-', label='ChebyKAN')
plt.plot(range(0, epochs, 100), jacobi_losses, 'c-', label='Jacobikan')  # Jacobikan losses
plt.plot(range(0, epochs, 100), legendre_losses, 'b--', label='LegendreKAN')  # LegendreKAN losses
plt.title('Convergence Speed Comparison Between kAN models, MLP, Jacobikan, and LegendreKAN')
plt.xlim(1000, epochs)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig('plot2.png',dpi=400,bbox_inches='tight')
plt.show()