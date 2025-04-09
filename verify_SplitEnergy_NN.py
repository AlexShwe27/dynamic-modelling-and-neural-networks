import numpy
import numpy as np
import torch.nn as nn
import torch
import random
import torch.optim as optim
from torchdiffeq import odeint
import pandas as pd
import matplotlib.pyplot as plt

# ===================================================================================================================
# SYSTEM
# ===================================================================================================================

# Parameters
m = 1.0
l = 1.0
g = 9.81


# -------------------------------------------------------------------------------------------------------------------
# Energy Functions
# -------------------------------------------------------------------------------------------------------------------


def kinetic_energy(trajectory):
    # Initialize an array or list to store the kinetic energy values
    KE = []

    # Iterate over the true_trajectory
    for state in trajectory:
        # Extract states
        q = state[:1].detach().numpy()
        p = state[1:].detach().numpy()

        T = 0.5 * (p ** 2) / (m * (l ** 2))

        KE.append(T)

    # Convert the list to a numpy array for further analysis
    KE = numpy.array(KE).squeeze()

    return KE


def potential_energy(trajectory):
    # Initialize an array or list to store the potential energy values
    PE = []

    # Iterate over the true_trajectory
    for state in trajectory:
        # Extract states
        q = state[:1].detach().numpy()
        p = state[1:].detach().numpy()

        V = m * g * l * (1 - np.cos(q))

        PE.append(V)

    # Convert the list to a numpy array for further analysis
    PE = numpy.array(PE).squeeze()

    return PE


# ===================================================================================================================
# DATA PREPARATION
# ===================================================================================================================

# -------------------------------------------------------------------------------------------------------------------
# Load True Data
# -------------------------------------------------------------------------------------------------------------------

# Load simulation data
sim_data = pd.read_csv('pendulum_verification_data.csv')
sim_data = sim_data.iloc[:-1]

# Process the data
data_points = 10000
sim_data = sim_data.iloc[::int(len(sim_data)/data_points)].reset_index(drop=True)

# Extract the variables from the DataFrame
sim_time = sim_data.iloc[:, 0].to_numpy(dtype=np.float32)
sim_angle = sim_data.iloc[:, 1].to_numpy(dtype=np.float32)
sim_angular_momentum = sim_data.iloc[:, 2].to_numpy(dtype=np.float32)

sim_kinetic_energy = sim_data.iloc[:, 3].to_numpy(dtype=np.float32)
sim_potential_energy = sim_data.iloc[:, 4].to_numpy(dtype=np.float32)
sim_total_energy = sim_data.iloc[:, 5].to_numpy(dtype=np.float32)

# Time
sim_time = torch.tensor(sim_time, dtype=torch.float32)

# True Trajectory
true_trajectory = np.column_stack((sim_angle, sim_angular_momentum))
true_trajectory = torch.tensor(true_trajectory, dtype=torch.float32)

# Initial Conditions
init_cond = true_trajectory[0, :]

# -------------------------------------------------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------------------------------------------------
# Data parameters
data_size = len(sim_data)
batch_count = 8
batch_time = 10

# Simulation time
nsteps = data_size


# ===================================================================================================================
# NEURAL ORDINARY DIFFERENTIAL EQUATION
# ===================================================================================================================

# -------------------------------------------------------------------------------------------------------------------
# Neural Networks
# -------------------------------------------------------------------------------------------------------------------


class kinetic_energy_net(nn.Module):

    def __init__(self):
        super(kinetic_energy_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, state):
        return self.net(state)


class potential_energy_net(nn.Module):

    def __init__(self):
        super(potential_energy_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, state):
        return self.net(state)


# -------------------------------------------------------------------------------------------------------------------
# Neural ODE
# -------------------------------------------------------------------------------------------------------------------

def pendulum_splitenergy_neuralODE(t, state):

    state = state.requires_grad_(True)

    if state.dim() == 2:
        q = state[:, 0].unsqueeze(1)
        p = state[:, 1].unsqueeze(1)
    else:
        q = state[0].unsqueeze(0)
        p = state[1].unsqueeze(0)

    # Obtain the kinetic energy and potential energy from Neural Network
    T = kinetic_neural_network(state)
    V = potential_neural_network(q)

    # Calculate the gradients of kinetic and potential energy w.r.t the respective states
    dT = torch.autograd.grad(outputs=T, inputs=state, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    dVdq = torch.autograd.grad(outputs=V, inputs=q, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    if state.dim() == 2:
        # Calculate dqdt and dpdt
        dqdt = dT[:, 1]
        dpdt = -dT[:, 0] - dVdq.squeeze()
        # Combine dqdt and dpdt into a single tensor
        dstate = torch.stack([dqdt, dpdt], dim=1)

    else:
        # Calculate dqdt and dpdt
        dqdt = dT[1]
        dpdt = -dT[0] - dVdq
        # Combine dqdt and dpdt into a single tensor
        dstate = torch.stack([dqdt, dpdt.squeeze()])

    return dstate


# -------------------------------------------------------------------------------------------------------------------
# Load Train Model
# -------------------------------------------------------------------------------------------------------------------

kinetic_neural_network = kinetic_energy_net()
kinetic_neural_network.load_state_dict(torch.load("pendulum_kinetic_energy_NN.pth", weights_only=True))
kinetic_neural_network.eval()

potential_neural_network = potential_energy_net()
potential_neural_network.load_state_dict(torch.load("pendulum_potential_energy_NN.pth", weights_only=True))
potential_neural_network.eval()

# -------------------------------------------------------------------------------------------------------------------
# Predict Trajectory
# -------------------------------------------------------------------------------------------------------------------

pred_trajectory = odeint(pendulum_splitenergy_neuralODE, init_cond, sim_time, method='rk4')

# -------------------------------------------------------------------------------------------------------------------
# Energy values
# -------------------------------------------------------------------------------------------------------------------

pred_kinetic_energy = kinetic_neural_network(true_trajectory).detach().numpy()
pred_potential_energy = potential_neural_network(true_trajectory[:, 0].unsqueeze(1)).detach().numpy()

# offset in energy
index = random.randint(1, data_size-1)
KE_offset = sim_kinetic_energy - pred_kinetic_energy

loss_in_T = np.abs(sim_kinetic_energy - pred_kinetic_energy)
loss_in_V = np.abs(sim_potential_energy - pred_potential_energy)
loss_in_total_energy = np.abs(sim_total_energy - (pred_kinetic_energy + pred_potential_energy))

# ===================================================================================================================
# PLOTS
# ===================================================================================================================

# prepare data for plot
pred_trajectory = pred_trajectory.detach().numpy()

# -------------------------------------------------------------------------------------------------------------------
# Compare true and predicted states
# -------------------------------------------------------------------------------------------------------------------

# Plots
plt.figure(1, figsize=(10, 10))

# Plot the angle of the pendulum
plt.subplot(2, 1, 1)
plt.plot(sim_time, sim_angle, label=r'true $\theta$')
plt.plot(sim_time, pred_trajectory[:, 0], 'r--', label=r'predicted $\theta$')
plt.title(r'Angle of the Pendulum ($\theta$)')
plt.xlabel(r'Time [ $s$ ]')
plt.ylabel(r'Angle [ $rad$ ]')
plt.legend()

# Plot the angular momentum
plt.subplot(2, 1, 2)
plt.plot(sim_time, sim_angular_momentum, label=r'true $p_\theta$')
plt.plot(sim_time, pred_trajectory[:, 1], 'r--', label=r'predicted $p_\theta$')
plt.title(r'Angular Momentum of the Pendulum ($p_\theta$)')
plt.xlabel(r'Time [ $s$ ]')
plt.ylabel(r'Angular Momentum [ $kgm^{2}s^{-1}$ ]')
plt.legend()

plt.figure(2, figsize=(10, 15))

# Plot the kinetic energy
plt.subplot(2, 1, 1)
plt.plot(sim_time, sim_kinetic_energy, label=r'true T')
plt.plot(sim_time, pred_kinetic_energy, 'r--', label=r'predicted T')
plt.title('Kinetic energy of the system')
plt.xlabel(r'Time [ $s$ ]')
plt.ylabel(r'Kinetic Energy [ $J$ ]')
plt.legend()

# Plot the potential energy
plt.subplot(2, 1, 2)
plt.plot(sim_time, sim_potential_energy, label=r'true V')
plt.plot(sim_time, pred_potential_energy, 'r--', label=r'predicted V')
plt.title('Potential energy of the system')
plt.xlabel(r'Time [ $s$ ]')
plt.ylabel(r'Potential Energy [ $J$ ]')
plt.legend()

# Plot the total energy
plt.figure(3)
plt.plot(sim_time, sim_total_energy, label=r'true total energy')
plt.plot(sim_time, pred_kinetic_energy+pred_potential_energy, 'r--', label=r'predicted total energy')
plt.title('Total energy of the system')
plt.xlabel(r'Time [ $s$ ]')
plt.ylabel(r'Total Energy [ $J$ ]')
plt.legend()

plt.show()
