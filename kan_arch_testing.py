import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import efficient_kan
import kan #This is from the pip install pykan.
from deeponet_derivative import generate_data, DeepONet, BranchNet, TrunkNet, KANBranchNet, KANTrunkNet

def train_kans():
    # Hyperparameters -- taken from other file, should tweak if needed.
    input_dim = 50  # Number of points in the function
    input_dim_trunk = 1 #Trunk net input size.
    hidden_dim = 100
    hidden_dim_kan = 2*input_dim + 1 #Change this for different architectures!
    hidden_dim_kan_trunk = 2*input_dim_trunk + 1
    output_dim = 50
    num_epochs = 10000
    learning_rate = 1e-4
    num_samples = 10000
    num_points = input_dim

    #Adding batching here.
    batch_size = 64
    num_batches = num_samples // batch_size

    mlp_onet = DeepONet(BranchNet(input_dim, hidden_dim, output_dim),
                         TrunkNet(input_dim_trunk, hidden_dim, output_dim),
                         label='MLP')
    kan_onet = DeepONet(KANBranchNet(input_dim, hidden_dim_kan, output_dim,modeltype='efficient_kan'),
                         KANTrunkNet(input_dim_trunk, hidden_dim_kan_trunk, output_dim, modeltype='efficient_kan'),
                         label='KAN')
    cheby_onet = DeepONet(KANBranchNet(input_dim, hidden_dim, output_dim, modeltype='cheby_kan'),
                         KANTrunkNet(input_dim_trunk, hidden_dim, output_dim, modeltype='cheby_kan'),
                         label='ChebyKAN')
    jacobi_onet = DeepONet(KANBranchNet(input_dim, hidden_dim, output_dim, modeltype='jacobi_kan'),
                         KANTrunkNet(input_dim_trunk, hidden_dim, output_dim, modeltype='jacobi_kan'),
                         label='JacobiKAN')
    leg_onet = DeepONet(KANBranchNet(input_dim, hidden_dim, output_dim, modeltype='legendre_kan'),
                         KANTrunkNet(input_dim_trunk, hidden_dim, output_dim, modeltype='legendre_kan'),
                         label='LegendreKAN')
    
    #Initialize the figure to which we will plot the loss curves.
    f = plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.title('Model Loss Comparison')
    
    #Initialize what's needed for training.
    x_train, y_train, z_train = generate_data(num_samples, num_points)
    onets = [mlp_onet, kan_onet, cheby_onet, jacobi_onet, leg_onet]
    # onet_losses = []
    for onet in onets:
        print(f"Training {onet.label}.")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(onet.parameters(), lr=learning_rate)
        losses = []
        train_dataset = TensorDataset(x_train, y_train, z_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for e in range(num_epochs):
            epoch_losses = []
            for x,y,z in train_dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = onet(x, y)
                loss = criterion(outputs, z)
                epoch_losses.append(loss.detach().numpy())

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            mean_loss = np.mean(epoch_losses)
            losses.append(mean_loss)
            if (e + 1) % 100 == 0:
                print(f'Epoch [{e + 1}/{num_epochs}], Loss: {mean_loss.item():.5f}')

        # onet_losses.append(losses)
        plt.plot(np.arange(num_epochs), losses, label=onet.label)
        print("Saving model...")
        torch.save(onet.state_dict, f'./trained_architectures/{onet.label}_derivative.pt')
        print("Models saved.")
        print("Training complete.")
    print(f"All {len(onets)} models trained to {num_epochs} iterations.")
    plt.legend(loc='best')
    plt.savefig('kan_arch_comparison.png')
    return None

#Obsolete function?
def test_plot(onet_losses):
    pass

def main():
    print("Main run.")
    train_kans()
    print("Complete.")
    return None

if __name__ == '__main__':
    main()

