import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(
    states: np.ndarray,
    actions: np.ndarray,
) -> None:
    """
    Plot the trajectory of the train.

    Parameters
    ----------
    states : np.ndarray
        The states of the train.
    actions : np.ndarray
        The actions taken by the train.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot position
    plt.subplot(4, 1, 1)
    plt.plot(states[:, 0], label='Position (m)')
    plt.title('Train Position (over time)')
    plt.xlabel('Step')
    plt.ylabel('Position (m)')
    plt.grid()
    
    # Plot velocity
    plt.subplot(4, 1, 2)
    plt.plot(states[:, 1], label='Velocity (m/s)', color='orange')
    plt.title('Train Velocity (over time)')
    plt.xlabel('Step')
    plt.ylabel('Velocity (m/s)')
    plt.grid()
    
    # Plot acceleration
    plt.subplot(4, 1, 3)
    plt.plot(actions, label='Inputs (%)', color='green')
    plt.title('Train Inputs (over time)')
    plt.xlabel('Step')
    plt.ylabel('Inputs (%)')
    plt.grid()
    
    # Plot velocity over position
    plt.subplot(4, 1, 4)
    plt.plot(states[:, 0], states[:, 1], label='Velocity vs Position', color='red')
    plt.title('Train Velocity (over position)')
    plt.xlabel('Position (m)')
    plt.ylabel('Velocity (m/s)')
    plt.grid()
    
    # Show the plot
    plt.tight_layout()
    plt.show()