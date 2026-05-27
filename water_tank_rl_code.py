# ============================================================
# SMART WATER TANK MANAGEMENT USING Q-LEARNING
# ============================================================

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import os

# ==============================
# SETUP
# ==============================
os.makedirs("plots", exist_ok=True)

MAX_LEVEL = 100
ACTIONS = [0, 1, 2, 3]
DEMANDS = [0, 1]

# ==============================
# FUNCTIONS
# ==============================

def get_level(value):
    if value <= 30:
        return 0
    elif value <= 70:
        return 1
    else:
        return 2

def get_reward(tankA, tankB, action, time):
    reward = 0

    if tankA > MAX_LEVEL or tankB > MAX_LEVEL:
        reward -= 10
    if tankA < 10 or tankB < 10:
        reward -= 10

    if 30 <= tankA <= 70:
        reward += 5
    if 30 <= tankB <= 70:
        reward += 5

    if action != 0:
        if time == 1:
            reward -= 5
        else:
            reward -= 2

    return reward

def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ==============================
# Q-TABLE
# ==============================
Q = np.zeros((3, 3, 2, 2, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 2000
steps = 50

reward_history = []
pump_usage_history = []

# ==============================
# TRAINING
# ==============================
print("Training...")

for ep in range(episodes):

    tankA = random.randint(20, 80)
    tankB = random.randint(20, 80)

    total_reward = 0
    pump_count = 0

    for step in range(steps):

        demand = random.choice(DEMANDS)
        time = step % 2

        state = (get_level(tankA), get_level(tankB), demand, time)

        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = np.argmax(Q[state])

        if action != 0:
            pump_count += 1

        # Apply action
        if action == 1:
            tankA += 20
        elif action == 2:
            tankB += 20
        elif action == 3:
            tankA += 20
            tankB += 20

        # Realistic different consumption
        if demand == 0:
            tankA -= random.randint(4, 6)
            tankB -= random.randint(5, 7)
        else:
            tankA -= random.randint(8, 12)
            tankB -= random.randint(9, 13)

        tankA = max(0, min(120, tankA))
        tankB = max(0, min(120, tankB))

        reward = get_reward(tankA, tankB, action, time)
        total_reward += reward

        next_state = (get_level(tankA), get_level(tankB), demand, time)

        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

    reward_history.append(total_reward)
    pump_usage_history.append(pump_count)

print("Training Done")

np.save("q_table.npy", Q)

# ==============================
# PLOTS
# ==============================

# Reward Plot
plt.figure()
plt.plot(reward_history, alpha=0.3, label="Raw")
plt.plot(moving_average(reward_history), label="Smoothed")
plt.legend()
plt.title("Reward vs Episodes")
plt.savefig("plots/reward_plot.png")
plt.close()

# Pump Usage Plot
plt.figure()
plt.plot(pump_usage_history, alpha=0.3, label="Raw")
plt.plot(moving_average(pump_usage_history), label="Smoothed")
plt.legend()
plt.title("Pump Usage")
plt.savefig("plots/pump_usage.png")
plt.close()


# Q-TABLE VISUALIZATION
policy = np.argmax(Q[:, :, 0, 0, :], axis=2)

plt.figure()
sns.heatmap(policy, annot=True, cmap="coolwarm")

plt.title("Optimal Policy Heatmap")
plt.xlabel("Tank B Level")
plt.ylabel("Tank A Level")

plt.savefig("plots/q_table_policy.png")
plt.close()



