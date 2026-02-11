import numpy as np
import matplotlib.pyplot as plt

env_names = ["highway-v0", "merge-v0", "intersection-v0", "roundabout-v0"]
mean_returns = []
cum_steps = []

for i in range(4):
    random_returns = "random_returns_" + env_names[i] + ".npy"
    random_steps = "random_steps_" + env_names[i] + ".npy"
    returns = np.load(random_returns)
    steps = np.load(random_steps)

    window = 5
    mean_returns.append(
        np.convolve(
        returns, np.ones(window)/window, mode="valid"   # Mean over a window of epi_returns Vs Steps experienced => 
        # For a random (or expert) policy with no learning, should be approx flat, with decreasing var as more experience is collected.
    )
    )
    cum_steps.append(steps[window-1:])

    # plt.plot(steps[window-1:], mean_returns)
    # plt.xlabel("Cumulative Environment Steps")
    # plt.ylabel("Mean Episode Return")
    # plt.title("Random Agent Performance on highway-v0")
    # plt.grid(True)
    # plt.savefig("random_agent_performance.png")
    # plt.show()


fig, axs = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
# Top-left plot
axs[0, 0].plot(cum_steps[0], mean_returns[0], 'tab:blue')
axs[0, 0].set_title('Highway-v')
axs[0, 0].set_xlabel('Cumulative Environment Steps')
axs[0, 0].set_ylabel('Mean Episode Return')
axs[0, 0].grid(True)

# Top-right plot
axs[0, 1].plot(cum_steps[1], mean_returns[1], 'tab:orange')
axs[0, 1].set_title('Merge-v0')
axs[0, 1].set_xlabel('Cumulative Environment Steps')
axs[0, 1].set_ylabel('Mean Episode Return')
axs[0, 1].grid(True)

# Bottom-left plot
axs[1, 0].plot(cum_steps[2], mean_returns[2], 'tab:green')
axs[1, 0].set_title('Intersection-v0')
axs[1, 0].set_xlabel('Cumulative Environment Steps')
axs[1, 0].set_ylabel('Mean Episode Return')
axs[1, 0].grid(True)

# Bottom-right plot
axs[1, 1].plot(cum_steps[2], mean_returns[2], 'tab:red')
axs[1, 1].set_title('Roundabout-v0')
axs[1, 1].set_xlabel('Cumulative Environment Steps')
axs[1, 1].set_ylabel('Mean Episode Return')
axs[1, 1].grid(True)

fig.suptitle('Random Agent Performance', fontsize=16)

plt.savefig("random_agent_performance_all_env.png")
plt.show()