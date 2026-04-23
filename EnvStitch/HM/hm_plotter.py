import pandas as pd
import matplotlib.pyplot as plt

ppo = pd.read_csv("ppo_results.csv")
dqn = pd.read_csv("dqn_results.csv")
rnd = pd.read_csv("random_results.csv")

plt.plot(ppo["return"], label="PPO")
plt.plot(dqn["return"], label="DQN")
plt.plot([rnd["return"].mean()]*len(ppo), label="Random")

plt.xlabel("Episodes")
plt.ylabel("Return")
plt.legend()
plt.show()