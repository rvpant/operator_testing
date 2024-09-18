import os  # **CHANGE**: Import os to manage directories.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import efficient_kan
import kan  # This is from the pip install pykan.
from deeponet_derivative import generate_data, DeepONet, BranchNet, TrunkNet, KANBranchNet, KANTrunkNet

def train_kans():
    # Hyperparameters
    input_dim = 50  # Number of points in the function
    input_dim_trunk = 1  # Trunk net input size.
    hidden_dim = 100
    hidden_dim_kan = 2 * input_dim + 1  # Change this for different architectures!
    hidden_dim_kan_trunk = 2 * input_dim_trunk + 1
    output_dim = 50
    num_epochs = 5000
    learning_rate = 5e-4
    num_samples = 1000
    num_val_samples = 100
    num_points = input_dim

    batch_size = 256
    num_batches = num_samples // batch_size

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models and move them to the appropriate device
    mlp_onet = DeepONet(BranchNet(input_dim, hidden_dim, output_dim),
                        TrunkNet(input_dim_trunk, hidden_dim, output_dim),
                        label='MLP').to(device)
    kan_onet = DeepONet(KANBranchNet(input_dim, hidden_dim_kan, output_dim, modeltype='efficient_kan'),
                        KANTrunkNet(input_dim_trunk, hidden_dim_kan_trunk, output_dim, modeltype='efficient_kan'),
                        label='KAN').to(device)
    cheby_onet = DeepONet(KANBranchNet(input_dim, hidden_dim, output_dim, modeltype='cheby_kan'),
                          KANTrunkNet(input_dim_trunk, hidden_dim, output_dim, modeltype='cheby_kan'),
                          label='ChebyKAN').to(device)
    jacobi_onet = DeepONet(KANBranchNet(input_dim, hidden_dim, output_dim, modeltype='jacobi_kan'),
                           KANTrunkNet(input_dim_trunk, hidden_dim, output_dim, modeltype='jacobi_kan'),
                           label='JacobiKAN').to(device)
    leg_onet = DeepONet(KANBranchNet(input_dim, hidden_dim, output_dim, modeltype='legendre_kan'),
                        KANTrunkNet(input_dim_trunk, hidden_dim, output_dim, modeltype='legendre_kan'),
                        label='LegendreKAN').to(device)

    # **CHANGE**: Create the directory if it doesn't exist
    if not os.path.exists('./trained_architectures'):  # **CHANGE**
        os.makedirs('./trained_architectures')  # **CHANGE**

    # Initialize the figure for plotting the loss curves
    f = plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.title('Model Loss Comparison')

    # Generate training data and move to the appropriate device
    x_train, y_train, z_train = generate_data(num_samples, num_points)
    x_train, y_train, z_train = x_train.to(device), y_train.to(device), z_train.to(device)

    # Generate validation data and move to the appropriate device
    x_val, y_val, z_val = generate_data(num_val_samples, num_points)
    x_val, y_val, z_val = x_train.to(device), y_train.to(device), z_train.to(device)

    onets = [mlp_onet, kan_onet, cheby_onet, jacobi_onet, leg_onet]

    for onet in onets:
        print(f"Training {onet.label}.")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(onet.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        losses = []; val_losses = []
        train_dataset = TensorDataset(x_train, y_train, z_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(x_val, y_val, z_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for e in range(num_epochs):
            onet.train() #adding this along with the .eval() later in case we include layernorm, dropout etc in any models.
            epoch_losses = []
            epoch_val_losses = []
            for x, y, z in train_dataloader:
                x, y, z = x.to(device), y.to(device), z.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = onet(x, y)
                loss = criterion(outputs, z)
                epoch_losses.append(loss.detach().cpu().numpy())

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            mean_loss = np.mean(epoch_losses)
            losses.append(mean_loss)
            if (e + 1) % 100 == 0:
                print(f'Epoch [{e + 1}/{num_epochs}], Loss: {mean_loss:.5f}')
            
            #Validation mode.
            onet.eval() #see comment above regarding the train and validation modes.
            for x, y, z in val_dataloader:
                x, y, z = x.to(device), y.to(device), z.to(device)
                with torch.no_grad():
                    val_output = onet(x,y)
                    val_loss = criterion(val_output, z)
                    epoch_val_losses.append(val_loss.detach().cpu().numpy())

            mean_val_loss = np.mean(epoch_val_losses)
            val_losses.append(mean_val_loss)
            
            scheduler.step()

        # Plot the losses for each model
        plt.plot(np.arange(num_epochs), losses, label=onet.label)
        print("Saving model...")
        torch.save(onet.state_dict(), f'./trained_architectures/{onet.label}_derivative.pt')  # **CHANGE**: Save in the created directory
        print("Model saved.")
        print("Training complete.")

    print(f"All {len(onets)} models trained for {num_epochs} iterations.")
    plt.legend(loc='best')
    plt.savefig('kan_arch_comparison.png')

def main():
    print("Main run.")
    train_kans()
    print("Complete.")

if __name__ == '__main__':
    main()
