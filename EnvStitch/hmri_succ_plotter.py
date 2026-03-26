import pandas as pd
import matplotlib.pyplot as plt

# Load CSV results
ppo = pd.read_csv("ppo_results.csv")
dqn = pd.read_csv("dqn_results.csv")
random = pd.read_csv("random_results.csv")

# Success metrics we care about
success_metrics = ["success/highway", "success/merge", "success/roundabout", "success/intersection", "success/full"]
metric_labels = ["Highway", "Merge", "Roundabout", "Intersection", "Full Episode"]
colors = {"PPO": "blue", "DQN": "green", "Random": "red"}

plt.figure(figsize=(15, 10))

for i, metric in enumerate(success_metrics, 1):
    plt.subplot(3, 2, i)
    # PPO
    if metric in ppo.columns:
        plt.plot(ppo[metric], label="PPO", color=colors["PPO"])
    # DQN
    if metric in dqn.columns:
        plt.plot(dqn[metric], label="DQN", color=colors["DQN"])
    # Random
    if metric in random.columns:
        plt.plot(random[metric], label="Random", color=colors["Random"])

    plt.title(f"{metric_labels[i-1]} Success Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.grid(True)
    if i == 1:
        plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("all_success_metrics.png")
plt.show()