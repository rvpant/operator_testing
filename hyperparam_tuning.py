from deeponet_derivative import generate_data, DeepONet, KANBranchNet, KANTrunkNet, BranchNet, TrunkNet
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import kan
import efficient_kan
import argparse
import os
import glob

'''Similar to the tuning.py file, except this is a comprehensive script meaning to allow for the optimization
of any of the five models, eventually over three different problems (Burger's, derivative, antiderivative.)
We provide two arguments from the commmand line: -model specifies one of {MLP, efficient KAN, ChebyKAN, JacobiKAN, LegendreKAN},
and -problem specifies the specific operator problem to optimize over.'''

parser = argparse.ArgumentParser()
parser.add_argument('-model', dest='model', type=str, help='Model to optimize.',
                     choices=['MLP', 'efficient', 'cheby', 'jacobi', 'legendre'])
parser.add_argument('-problem', dest='problem', type=str, help='Problem to use during optimization.',
                    choices=['burgers', 'derivative', 'integral'],
                    default='derivative')
model = parser.parse_args().model; problem = parser.parse_args().problem
print(f"Hyperparameter optimization of {model} on {problem} problem.")


#Defining some default parameters that will be stored.
input_dim = 50  # Number of points in the function
input_dim_trunk = 1
hidden_dim = 100
hidden_dim_kan = 2*input_dim + 1 #Change this for different architectures!
output_dim = 50
num_epochs = int(5e3)
learning_rate = 1e-3
num_samples = 1000
num_val_samples = num_samples // 10
num_points = 50

#Define the output directory for the loss plots.
output_dir = './hyperparam_tuning'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#The below clears existing stuff in the directory ... not sure if needed.
# os.chmod(output_dir, 0o700)
# existing_files = glob.glob(os.path.join(output_dir, '*'))
# for f in existing_files:
#     os.remove(f)

def train_model(model, train_dl, val_dl, n_epochs, learning_rate):
    '''Function that stores training loops.'''
    print(f"Training {model.label}.")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []; val_losses = []

    #Handling the device correcly.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    for e in range(n_epochs):
        model.train()
        epoch_losses = []; epoch_val_losses = []
        for x,y,z in train_dl:
            x,y,z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x, y)
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
        model.eval() #see comment above regarding the train and validation modes.
        for x, y, z in val_dl:
            x, y, z = x.to(device), y.to(device), z.to(device)
            with torch.no_grad():
                val_output = model(x,y)
                val_loss = criterion(val_output, z)
                epoch_val_losses.append(val_loss.detach().cpu().numpy())

            epoch_val_losses.append(val_loss)
        mean_val_loss = np.mean(epoch_val_losses)
        val_losses.append(mean_val_loss)

    return model, losses, val_losses

# Generate training and validation data.
x_train, y_train, z_train = generate_data(num_samples, num_points)
x_val, y_val, z_val = generate_data(num_val_samples, num_points)
train_data = TensorDataset(x_train, y_train, z_train)
val_data = TensorDataset(x_val, y_val, z_val)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)


if model == "MLP":

    #Store the minimum achieved loss, and save new model if it has less avg loss over final epochs.
    min_loss = 1e4 #initialize to an improbably large number.
    best_model = '' #String that will print the best model.

    activations = ['relu', 'silu', 'leaky'] #These are the implemented options in deeponet_derivative.py for the BranchNet and TrunkNet
    lrs = [1e-4]
    hidden_dims = [50, 100, 200]

    f, ax = plt.subplots(2, 1,figsize=(20,10))
    f.suptitle("MLP models")
    ax[0].set_title("Training Loss Curves")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss (log scale)")
    ax[0].set_yscale('log')
    ax[1].set_title("Validation Loss Curves")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Loss (log scale)")
    ax[1].set_yscale('log')
    for activation in activations:
        for lr in lrs:
            for dim in hidden_dims:
                onet = DeepONet(BranchNet(input_dim, dim, output_dim, activ=activation),
                        TrunkNet(input_dim_trunk, dim, output_dim, activ=activation),
                        label='MLP')
                onet, losses, val_losses = train_model(onet, train_dataloader, val_dataloader, num_epochs, lr)
                ax[0].plot(range(num_epochs), losses, label=f'{onet.label}_{activation}_hidden{dim}_{lr}')
                ax[1].plot(range(num_epochs), val_losses, label=f'{onet.label}_{activation}_hidden{dim}_{lr}')
                torch.save(onet, f'{output_dir}/{onet.label}_{activation}_hidden{dim}_{lr}_{problem}.pt')
                if np.mean(val_losses[(len(val_losses)//2):]) < min_loss:
                    torch.save(onet, f'{output_dir}/best_MLP_model_{activation}_hidden{dim}_{lr}_{problem}.pt')
                    best_model = f'{onet.label}_{activation}_hidden{dim}_{lr}_{problem}'
                    
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    f.savefig(f'{output_dir}/{model}_hyperparam_tuning.png')
    print(f"Best MLP model is {best_model}.")
                
elif model == 'efficient':

    #Store the minimum achieved loss, and save new model if it has less avg loss over final epochs.
    min_loss = 1e4 #initialize to an improbably large number.
    best_model = '' #String that will print the best model. 

    n = input_dim
    p = input_dim_trunk
    lrs = [1e-4]
    hidden_dims = [n, (3*n)//2, 2*n+1, 2*n+5]
    hidden_dims_trunk = [p, 2*p+1, 2*p+5]

    f, ax = plt.subplots(2,1,figsize=(20,10))
    f.suptitle("efficient KAN models")
    ax[0].set_title("Training Loss Curves")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss (log scale)")
    ax[0].set_yscale('log')
    ax[1].set_title("Validation Loss Curves")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Loss (log scale)")
    ax[1].set_yscale('log')
    for lr in lrs:
        for dim in hidden_dims:
            for trunk_dim in hidden_dims_trunk:
                onet = DeepONet(KANBranchNet(input_dim, dim, output_dim, modeltype='efficient_kan'),
                        KANTrunkNet(input_dim_trunk, trunk_dim, output_dim, modeltype='efficient_kan'),
                        label='efficient_KAN')
                onet, losses, val_losses = train_model(onet, train_dataloader, val_dataloader, num_epochs, lr)
                ax[0].plot(range(num_epochs), losses, label=f'{onet.label}_hidden{dim}_hidden_trunk_{trunk_dim}_{lr}')
                ax[1].plot(range(num_epochs), val_losses, label=f'{onet.label}_hidden{dim}_hidden_trunk_{trunk_dim}_{lr}')
                torch.save(onet, f'{output_dir}/{onet.label}_hidden{dim}_hidden_trunk_{trunk_dim}_{lr}.pt')
                if np.mean(val_losses[(len(val_losses)//2):]) < min_loss:
                    torch.save(onet, f'{output_dir}/best_efficientKAN_model_hidden{dim}_trunk{trunk_dim}_{lr}_{problem}.pt')
                    best_model = f'{onet.label}_hidden{dim}_trunk{trunk_dim}_{lr}_{problem}'

    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    f.savefig(f'{output_dir}/{model}_hyperparam_tuning.png')
    print(f"Best efficientKAN model is {best_model}.")

elif model == 'cheby':

    #Store the minimum achieved loss, and save new model if it has less avg loss over final epochs.
    min_loss = 1e4 #initialize to an improbably large number.
    best_model = '' #String that will print the best model.

    layernorm_bools = [True, False]
    degrees = [2, 3, 4, 5, 8]
    hidden_dims = [50, 100, 200]
    lrs = [1e-4]

    f, ax = plt.subplots(2, 1, figsize=(20,10))
    f.suptitle("chebyKAN models")
    ax[0].set_title("Training Loss Curves")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss (log scale)")
    ax[0].set_yscale('log')
    ax[1].set_title("Validation Loss Curves")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Loss (log scale)")
    ax[1].set_yscale('log')
    for lr in lrs:
        for dim in hidden_dims:
            for deg in degrees:
                for b in layernorm_bools:
                    onet = DeepONet(KANBranchNet(input_dim, dim, output_dim, modeltype='cheby_kan', layernorm=b),
                            KANTrunkNet(input_dim_trunk, dim, output_dim, modeltype='cheby_kan', layernorm=b),
                            label='chebyKAN')
                    onet, losses, val_losses = train_model(onet, train_dataloader, val_dataloader, num_epochs, lr)
                    ax[0].plot(losses, label=f'{onet.label}_hidden{dim}_degree{deg}_{lr}')
                    ax[1].plot(val_losses, label=f'{onet.label}_hidden{dim}_degree{deg}_{lr}')
                    torch.save(onet, f'{output_dir}/{onet.label}_hidden{dim}_degree{deg}_{lr}.pt')
                    if np.mean(val_losses[(len(val_losses)//2):]) < min_loss:
                        torch.save(onet, f'{output_dir}/best_chebyKAN_model_hidden{dim}_degree{deg}_{lr}_{problem}.pt')
                        best_model = f'{onet.label}_hidden{dim}_degree{deg}_{lr}_{problem}'

    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    f.savefig(f'{output_dir}/{model}_hyperparam_tuning.png')
    print(f"Best chebyKAN model is {best_model}.")

elif model == 'jacobi':

    #Store the minimum achieved loss, and save new model if it has less avg loss over final epochs.
    min_loss = 1e4 #initialize to an improbably large number.
    best_model = '' #String that will print the best model.

    layernorm_bools = [True, False]
    degrees = [2, 3, 4, 5, 8]
    hidden_dims = [50, 100, 200]
    lrs = [1e-4]

    f, ax = plt.subplots(2, 1, figsize=(20,10))
    f.suptitle("jacobiKAN models")
    ax[0].set_title("Training Loss Curves")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss (log scale)")
    ax[0].set_yscale('log')
    ax[1].set_title("Validation Loss Curves")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Loss (log scale)")
    ax[1].set_yscale('log')
    for lr in lrs:
        for dim in hidden_dims:
            for deg in degrees:
                for b in layernorm_bools:
                    onet = DeepONet(KANBranchNet(input_dim, dim, output_dim, modeltype='jacobi_kan', layernorm=b),
                            KANTrunkNet(input_dim_trunk, dim, output_dim, modeltype='jacobi_kan', layernorm=b),
                            label='jacobiKAN')
                    onet, losses, val_losses = train_model(onet, train_dataloader, val_dataloader, num_epochs, lr)
                    ax[0].plot(losses, label=f'{onet.label}_hidden{dim}_degree{deg}_{lr}')
                    ax[1].plot(val_losses, label=f'{onet.label}_hidden{dim}_degree{deg}_{lr}')
                    torch.save(onet, f'{output_dir}/{onet.label}_hidden{dim}_degree{deg}_{lr}.pt')
                    if np.mean(val_losses[(len(val_losses)//2):]) < min_loss:
                        torch.save(onet, f'{output_dir}/best_jacobiKAN_model_hidden{dim}_degree{deg}_{lr}_{problem}.pt')
                        best_model = f'{onet.label}_hidden{dim}_degree{deg}_{lr}_{problem}'
                    
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    f.savefig(f'{output_dir}/{model}_hyperparam_tuning.png')
    print(f"Best jacobiKAN model is {best_model}.")

elif model == 'legendre':

    #Store the minimum achieved loss, and save new model if it has less avg loss over final epochs.
    min_loss = 1e4 #initialize to an improbably large number.
    best_model = '' #String that will print the best model.

    layernorm_bools = [True, False]
    degrees = [2, 3, 4, 5, 8]
    hidden_dims = [50, 100, 200]
    lrs = [1e-4]

    f, ax = plt.subplots(2, 1, figsize=(20,10))
    f.suptitle("legendreKAN models")
    ax[0].set_title("Training Loss Curves")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss (log scale)")
    ax[0].set_yscale('log')
    ax[1].set_title("Validation Loss Curves")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Loss (log scale)")
    ax[1].set_yscale('log')
    for lr in lrs:
        for dim in hidden_dims:
            for deg in degrees:
                for b in layernorm_bools:
                    onet = DeepONet(KANBranchNet(input_dim, dim, output_dim, modeltype='legendre_kan', layernorm=b),
                            KANTrunkNet(input_dim_trunk, dim, output_dim, modeltype='legendre_kan', layernorm=b),
                            label='legendreKAN')
                    onet, losses, val_losses = train_model(onet, train_dataloader, val_dataloader, num_epochs, lr)
                    ax[0].plot(losses, label=f'{onet.label}_hidden{dim}_degree{deg}_{lr}')
                    ax[1].plot(val_losses, label=f'{onet.label}_hidden{dim}_degree{deg}_{lr}')
                    torch.save(onet, f'{output_dir}/{onet.label}_hidden{dim}_degree{deg}_{lr}.pt')
                    if np.mean(val_losses[(len(val_losses)//2):]) < min_loss:
                        torch.save(onet, f'{output_dir}/best_legendreKAN_model_hidden{dim}_degree{deg}_{lr}_{problem}.pt')
                        best_model = f'{onet.label}_hidden{dim}_degree{deg}_{lr}_{problem}'
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    f.savefig(f'{output_dir}/{model}_hyperparam_tuning.png')
    print(f"Best legendreKAN model is {best_model}.")
else:
    print(f"Invalid input mode.")
