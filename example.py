import numpy as np

from ato_simulator.ato import ATOSingleTrain
from ato_simulator.aux_functions import plot_trajectory

dt = 0.01
n_steps = 100
train = ATOSingleTrain(
    a_davis_formula=0.5,
    b_davis_formula=0.1,
    c_davis_formula=0.01,
    train_mass=100000,
    maximum_traction_force=500000,
    maximum_braking_force=500000,
)

# Initial state: [position, velocity, acceleration]
state = np.array([0, 0]) 
all_states = np.zeros((n_steps+1, 2))
all_actions = np.zeros((n_steps, 1))
for i in range(100):
    # Compute the action
    action = 1.0  # Full throttle
    # Save data
    all_states[i] = state
    all_actions[i] = action
    # Update the state of the train
    state, info = train.step(state, action, dt, {}, {}, {})
    # print(f"Step {i}: State: {state}")
    # print(f"Step {i}: Info: {info}")
# Save the last state
all_states[-1] = state

# Plot the trajectory
plot_trajectory(all_states, all_actions)