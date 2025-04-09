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
mc = 5.0
mp = 2.0
l = 0.75
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
        q = state[:2].detach().numpy()
        p = state[2:].detach().numpy()

        T = (0.5*((mc+mp)/(mc**2))*(p[1]**2)) - ((1/(mc*l))*p[0]*p[1]*np.cos(q[0])) + (0.5*(1/(mp*(l**2)))*(p[0]**2))

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

        V = mp*g*l*np.cos(q[0])

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
sim_data = pd.read_csv('cartpole_verification_data.csv')
sim_data = sim_data.iloc[:-1]

# Process the data
data_points = 10000
sim_data = sim_data.iloc[::int(len(sim_data)/data_points)].reset_index(drop=True)

# Extract the variables from the DataFrame
sim_time = sim_data.iloc[:, 0].to_numpy(dtype=np.float32)
sim_angle = sim_data.iloc[:, 1].to_numpy(dtype=np.float32)
sim_position = sim_data.iloc[:, 2].to_numpy(dtype=np.float32)
sim_angular_momentum = sim_data.iloc[:, 3].to_numpy(dtype=np.float32)
sim_linear_momentum = sim_data.iloc[:, 4].to_numpy(dtype=np.float32)

sim_kinetic_energy = sim_data.iloc[:, 5].to_numpy(dtype=np.float32)
sim_potential_energy = sim_data.iloc[:, 6].to_numpy(dtype=np.float32)
sim_total_energy = sim_data.iloc[:, 7].to_numpy(dtype=np.float32)

# Time
sim_time = torch.tensor(sim_time, dtype=torch.float32)

# True Trajectory
true_trajectory = np.column_stack((sim_angle, sim_position, sim_angular_momentum, sim_linear_momentum))
true_trajectory = torch.tensor(true_trajectory, dtype=torch.float32)

# Initial Conditions
init_cond = true_trajectory[0, :]


# ===================================================================================================================
# NEURAL ORDINARY DIFFERENTIAL EQUATION
# ===================================================================================================================

# -------------------------------------------------------------------------------------------------------------------
# Neural Network
# -------------------------------------------------------------------------------------------------------------------

class hamiltonian_net(nn.Module):

    def __init__(self):
        super(hamiltonian_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 576),
            nn.Tanh(),
            nn.Linear(576, 576),
            nn.Tanh(),
            nn.Linear(576, 1)
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

def cartpole_hamiltonian_neuralODE(t, state):

    print(t)

    state = state.requires_grad_(True)

    # Obtain Hamiltonian from Neural Network
    H = neural_network(state)

    # Calculate the gradients of Hamiltonian w.r.t. the coordinates
    dH = torch.autograd.grad(outputs=H, inputs=state, grad_outputs=torch.ones_like(H), create_graph=True)[0]

    # Calculate the sate derivatives from the gradients of Hamiltonian
    if dH.dim() == 2:
        # Calculate dqdt and dpdt
        dqdt = dH[:, 2:4]
        dpdt = -dH[:, 0:2]
        # Combine dqdt and dpdt into a single tensor
        dstate = torch.cat([dqdt, dpdt], dim=1)

    else:
        # Calculate dqdt and dpdt
        dqdt = dH[2:4]
        dpdt = -dH[0:2]
        # Combine dqdt and dpdt into a single tensor
        dstate = torch.cat([dqdt, dpdt])

    return dstate


# -------------------------------------------------------------------------------------------------------------------
# Load Train Model
# -------------------------------------------------------------------------------------------------------------------

neural_network = hamiltonian_net()
neural_network.load_state_dict(torch.load("cartpole_hamiltonian_NN.pth", weights_only=True))
neural_network.eval()

# -------------------------------------------------------------------------------------------------------------------
# Predict Trajectory
# -------------------------------------------------------------------------------------------------------------------

pred_trajectory = odeint(cartpole_hamiltonian_neuralODE, init_cond, sim_time, method='fixed_adams')

# ===================================================================================================================
# PLOTS
# ===================================================================================================================

# prepare data for plot
true_trajectory = true_trajectory.detach().numpy()
pred_trajectory = pred_trajectory.detach().numpy()

# -------------------------------------------------------------------------------------------------------------------
# Compare true and predicted states
# -------------------------------------------------------------------------------------------------------------------

# Plots
plt.figure(1)

# Plot the angle of the pendulum
plt.subplot(2, 2, 1)
plt.plot(sim_time, sim_angle, label=r'true $\theta$')
plt.plot(sim_time, pred_trajectory[:, 0], 'r--', label=r'predicted $\theta$')
plt.title(r'Angle of the Pendulum ($\theta$)')
plt.xlabel(r'Time [ $s$ ]')
plt.ylabel(r'Angle [ $rad$ ]')
plt.legend()

# Plot the position of the cart
plt.subplot(2, 2, 2)
plt.plot(sim_time, sim_position, label=r'true $x$')
plt.plot(sim_time, pred_trajectory[:, 1], 'r--', label=r'predicted $x$')
plt.title(r'position of the cart ($x$)')
plt.xlabel(r'Time [ $s$ ]')
plt.ylabel(r'Distance [ $m$ ]')
plt.legend()

# Plot the angular momentum
plt.subplot(2, 2, 3)
plt.plot(sim_time, sim_angular_momentum, label=r'true $p_\theta$')
plt.plot(sim_time, pred_trajectory[:, 2], 'r--', label=r'predicted $p_\theta$')
plt.title(r'Angular Momentum of the Pendulum ($p_\theta$)')
plt.xlabel(r'Time [ $s$ ]')
plt.ylabel(r'Angular Momentum [ $kgm^{2}s^{-1}$ ]')
plt.legend()

# Plot the linear momentum
plt.subplot(2, 2, 4)
plt.plot(sim_time, sim_linear_momentum, label=r'true $p_x$')
plt.plot(sim_time, pred_trajectory[:, 3], 'r--', label=r'predicted $p_x$')
plt.title(r'Angular Momentum of the Pendulum ($p_x$)')
plt.xlabel(r'Time [ $s$ ]')
plt.ylabel(r'Linear Momentum [ $kgms^{-1}$ ]')
plt.ylim([-0.05, 0.05])
plt.legend()

plt.show()
