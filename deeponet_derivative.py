import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import kan #This is the import of the original kan implementation from PyKAN.
import efficient_kan #This should import the efficient KAN implementation from the directory.

#Add imports of other architectures.
from architectures_tobe_tested.ChebyKAN import GeneralChebyKAN
from architectures_tobe_tested.JacobiKAN import GeneralJacobiKAN
from architectures_tobe_tested.LegendreKAN import GeneralLegendreKAN

## should probably simplify these 2 functions into 1
# Define the Branch network
class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,  activ='relu'):
        super(BranchNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        if activ == 'silu':
            self.activation = nn.SiLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

#Define a similar class for a branch network using KANs
class KANBranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, modeltype='original_kan', layernorm=None):
        super(KANBranchNet, self).__init__()
        if modeltype=='original_kan':
            self.branch = kan.KAN(width=[input_dim, hidden_dim, output_dim], grid=5, k=3, seed=0)
        elif modeltype=='cheby_kan':
            self.branch = GeneralChebyKAN(layer_dims=[input_dim, hidden_dim, output_dim], norm=layernorm) #default degree 4
        elif modeltype=='jacobi_kan':
            self.branch = GeneralJacobiKAN(layer_dims=[input_dim, hidden_dim, output_dim], norm=layernorm) #def deg 4
        elif modeltype=='legendre_kan':
            self.branch = GeneralLegendreKAN(layer_dims=[input_dim, hidden_dim, output_dim], norm=layernorm) #def deg 4
        else:
            self.branch = efficient_kan.KAN(layers_hidden = [input_dim] + [hidden_dim]*1 + [output_dim])

    def forward(self, x):
        
        return self.branch(x)

# Define the Trunk network
class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activ='relu'):
        super(TrunkNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        if activ == 'silu':
            self.activation = nn.SiLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, y):
        y = self.activation(self.fc1(y))
        y = self.activation(self.fc2(y))
        y = self.fc3(y)
        return y

#Define a similar class again for a trunk network using KANs
class KANTrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, modeltype='original_kan', layernorm=None):
        super(KANTrunkNet, self).__init__()
        if modeltype == 'original_kan':
            self.trunk = kan.KAN(width=[input_dim, hidden_dim, output_dim], grid=5, k=3, seed=0)
        elif modeltype=='cheby_kan':
            self.trunk = GeneralChebyKAN(layer_dims=[input_dim, hidden_dim, output_dim], norm=layernorm) #default degree 4
        elif modeltype=='jacobi_kan':
            self.trunk = GeneralJacobiKAN(layer_dims=[input_dim, hidden_dim, output_dim], norm=layernorm) #def deg 4
        elif modeltype=='legendre_kan':
            self.trunk = GeneralLegendreKAN(layer_dims=[input_dim, hidden_dim, output_dim], norm=layernorm) #def deg 4
        else:
            self.trunk = efficient_kan.KAN(layers_hidden = [input_dim] + [hidden_dim]*1 + [output_dim])

    def forward(self, y):
        
        return self.trunk(y)
        
# Define the DeepONet
class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net, label):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.label = label

    def forward(self, x, y):
        branch_out = self.branch_net(x)
        trunk_out = self.trunk_net(y)
        out = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
        return out
        
# Generate training data using cubic polynomials, use other functions if needed
def generate_data(num_samples, num_points):
    x_data = []
    y_data = []
    z_data = []

    for _ in range(num_samples):
        # Coefficients for cubic polynomial: a*x^3 + b*x^2 + c*x + d
        coeffs = np.random.randn(4)
        f = lambda x: coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x + coeffs[3]
        df = lambda x: 3 * coeffs[0] * x**2 + 2 * coeffs[1] * x + coeffs[2]

        x = np.linspace(-1, 1, num_points)
        y = np.random.uniform(-1, 1, 1)
        z = df(y)

        x_data.append(f(x))
        y_data.append(y)
        z_data.append(z)

    return torch.tensor(np.array(x_data), dtype=torch.float32), torch.tensor(np.array(y_data), dtype=torch.float32), torch.tensor(np.array(z_data), dtype=torch.float32)
    
def plot_results(deeponet, num_points=100):
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))

    lbl = deeponet.label #This tells us if we have MLP, KAN, efficient_kan, etc in the ONet.

    for i in range(2):
        # Generate a new test case
        coeffs = np.random.randn(4)
        f_test = lambda x: coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x + coeffs[3]
        df_test = lambda x: 3 * coeffs[0] * x**2 + 2 * coeffs[1] * x + coeffs[2]

        x_test = np.linspace(-1, 1, num_points)
        x_test_data = torch.tensor(np.array([f_test(x_test)]), dtype=torch.float32)

        # Compute the true derivative across the domain
        true_derivative = df_test(x_test)

        # Compute the learned derivative across the domain
        learned_derivative = []
        with torch.no_grad():
            for y in x_test:
                y_test_data = torch.tensor(np.array([[y]]), dtype=torch.float32)
                z_pred = deeponet(x_test_data, y_test_data)
                learned_derivative.append(z_pred.item())

        # Plotting the original function
        axs[i].plot(x_test, f_test(x_test), label="Cubic Polynomial", color='blue')
        
        # Plotting the true derivative
        axs[i].plot(x_test, true_derivative, label="True Derivative", color='green')
        
        # Plotting the learned derivative
        axs[i].plot(x_test, learned_derivative, label="Learned Derivative", color='red', linestyle='--')

        axs[i].legend()
        axs[i].set_title(f'Example {i+1}')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('f(x) / f\'(x)')

    fig.savefig(f'results_compare_{deeponet.label}.png', dpi=500,bbox_inches='tight')


def model_compare(onets):
    '''Takes in a list of models to plot comparatively.'''
    # n = len(onets)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for i in range(3):

        # Generate a new test case
        coeffs = np.random.randn(4)
        f_test = lambda x: coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x + coeffs[3]
        df_test = lambda x: 3 * coeffs[0] * x**2 + 2 * coeffs[1] * x + coeffs[2]

        x_test = np.linspace(-1, 1, num_points)
        x_test_data = torch.tensor(np.array([f_test(x_test)]), dtype=torch.float32)

        # Compute the true derivative across the domain
        true_derivative = df_test(x_test)

        # Compute the learned derivative across the domain
        for onet in onets:
            learned_derivative = []
            lbl = onet.label #This tells us if we have MLP, KAN, efficient_kan, etc in the ONet.
            with torch.no_grad():
                for y in x_test:
                    y_test_data = torch.tensor(np.array([[y]]), dtype=torch.float32)
                    z_pred = onet(x_test_data, y_test_data)
                    learned_derivative.append(z_pred.item())
            axs[i].plot(x_test, learned_derivative, label=f"{lbl} Learned Derivative", linestyle='--')

        # Plotting the original function
        # axs[i].plot(x_test, f_test(x_test), label="Cubic Polynomial", color='blue')
        
        # Plotting the true derivative
        axs[i].plot(x_test, true_derivative, label="True Derivative", color='green')
        
        # Plotting the learned derivative
        # axs[i].plot(x_test, learned_derivative, label="Learned Derivative", color='red', linestyle='--')

        axs[i].legend()
        axs[i].set_title(f'Example {i+1}')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('f(x) / f\'(x)')

    fig.savefig(f'model_compare.png', dpi=500,bbox_inches='tight')


def main():
    # Hyperparameters
    input_dim = 20  # Number of points in the function
    hidden_dim = 50
    hidden_dim_kan = 2*input_dim + 1 #Change this for different architectures!
    output_dim = 50
    num_epochs = 1e4
    learning_rate = 1e-4
    num_samples = 1000
    num_points = 20

    # Initialize networks
    branch_net = BranchNet(input_dim, hidden_dim, output_dim)
    trunk_net = TrunkNet(1, hidden_dim, output_dim)
    deeponet = DeepONet(branch_net, trunk_net, label='MLP')

    # Initialize the KAN networks
    branch_net_kan = KANBranchNet(input_dim, hidden_dim_kan, output_dim, modeltype='original_kan')
    trunk_net_kan = KANTrunkNet(1, 3, output_dim, modeltype='original_kan')
    deeponet_kan = DeepONet(branch_net_kan, trunk_net_kan, label='original_kan')

    criterion = nn.MSELoss(); criterion_kan = nn.MSELoss()
    optimizer = optim.Adam(deeponet.parameters(), lr=learning_rate)
    optimizer_kan = optim.Adam(deeponet_kan.parameters(), lr=learning_rate)
    # Generate data, here I am using fixed data for sensors and the solutions, maybe need changing
    x_train, y_train, z_train = generate_data(num_samples, num_points)
    print("y_train shape: ", np.shape(y_train), type(y_train))

    mlp_losses = []
    # Training loop for MLP model.
    print(('-----TRAINING MLP MODEL-----'))
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = deeponet(x_train, y_train)
        loss = criterion(outputs, z_train)
        mlp_losses.append(loss.detach().numpy())

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    kan_losses = []
    # Training loop for KAN model.
    print('-----TRAINING KAN MODEL-----')
    for epoch in range(num_epochs):
        optimizer_kan.zero_grad()

        # Forward pass
        outputs = deeponet_kan(x_train, y_train)
        loss = criterion_kan(outputs, z_train)
        kan_losses.append(loss.detach().numpy())

        # Backward pass and optimize
        loss.backward()
        optimizer_kan.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('-----ALL TRAINING COMPLETE-----')

    # Test on a new cubic polynomial
    with torch.no_grad():
        coeffs = np.random.randn(4)
        f_test = lambda x: coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x + coeffs[3]
        df_test = lambda x: 3 * coeffs[0] * x**2 + 2 * coeffs[1] * x + coeffs[2]

        x_test = np.linspace(-1, 1, num_points)
        y_test = np.random.uniform(-1, 1, 1)

        x_test_data = torch.tensor(np.array([f_test(x_test)]), dtype=torch.float32)
        y_test_data = torch.tensor(np.array([y_test]), dtype=torch.float32)

        z_pred = deeponet(x_test_data, y_test_data)
        z_pred_kan = deeponet_kan(x_test_data, y_test_data)
        z_true = torch.tensor(np.array([df_test(y_test)]), dtype=torch.float32)

        print(f'Predicted derivative ({deeponet.label}): {z_pred.item():.4f}')
        print(f'Predicted derivative ({deeponet_kan.label}): {z_pred_kan.item():.4f}')
        print(f'True derivative: {z_true.item():.4f}')
        
    # Call the plotting function
    plot_results(deeponet, num_points=num_points)
    plot_results(deeponet_kan, num_points=num_points)

    #Call cross-model plotting function
    model_compare([deeponet, deeponet_kan])

    # print(kan_losses), len(kan_losses)
    #And: save a loss trajectory figure for each model.
    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(np.arange(num_epochs), kan_losses, label=f'{deeponet_kan.label} loss')
    ax.plot(np.arange(num_epochs), mlp_losses, label='MLP loss')
    ax.set_title("Loss Trajectories for each model")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.legend(loc='best')
    fig.savefig('loss_trajectories.jpg')

    # #Finally, if we are using original KAN implementation: call their native plotting functions.
    # fkan = plt.figure()
    # if deeponet_kan.label == 'original_kan':
    #     branch_net_kan.branch.plot('./branch_net_plots')
    #     # trunk_net_kan.trunk.plot('./trunk_net_plots')
    return None

if __name__ == '__main__':
    main()
