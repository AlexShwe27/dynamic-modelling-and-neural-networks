import numpy as np
import torch.nn as nn
import torch
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

        T = 0.5 * (p**2) / (m * (l**2))

        KE.append(T)

    # Convert the list to a numpy array for further analysis
    KE = np.array(KE).squeeze()

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
    PE = np.array(PE).squeeze()

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
# Neural Network
# -------------------------------------------------------------------------------------------------------------------


class baseline_net(nn.Module):

    def __init__(self):
        super(baseline_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
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

def pendulum_baseline_neuralODE(t, state):

    state = state.requires_grad_(True)

    # Obtain the state derivatives directly from the neural network
    dstate = neural_network(state)

    return dstate


# -------------------------------------------------------------------------------------------------------------------
# Load Trained Model
# -------------------------------------------------------------------------------------------------------------------

neural_network = baseline_net()
neural_network.load_state_dict(torch.load("pendulum_baseline_NN.pth", weights_only=True))
neural_network.eval()

# -------------------------------------------------------------------------------------------------------------------
# Predict Trajectory
# -------------------------------------------------------------------------------------------------------------------
pred_trajectory = odeint(pendulum_baseline_neuralODE, init_cond, sim_time)

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

plt.show()
