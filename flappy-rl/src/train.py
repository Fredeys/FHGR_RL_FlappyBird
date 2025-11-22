from src.env import FlappyEnv
from src.agent import QLearningAgent
import numpy as np
import pickle
import os


def main():
    print("train.py wurde gestartet")

    env = FlappyEnv()
    agent = QLearningAgent()

    episodes = 10000 
    rewards = []

    # Make sure 'runs' exists
    os.makedirs("runs", exist_ok=True)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1

            # Break against infinity loops
            if steps > 2000:
                print(f"Warnung: Episode {ep} überschreitet 2000 Schritte – breche ab.")
                done = True

        rewards.append(total_reward)

        # Message the user every 100 iterations
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}/{episodes}: Return = {total_reward:.1f}, epsilon = {agent.epsilon:.3f}")

    print("Training fertig, speichere Ergebnisse...")

    # Save Q-Table
    q_table_path = os.path.join("runs", "q_table.pkl")
    with open(q_table_path, "wb") as f:
        pickle.dump(agent.Q, f)

    # Save Rewards
    rewards_path = os.path.join("runs", "rewards.csv")
    np.savetxt(rewards_path, rewards)

    print(f"Fertig! Q-Table: {q_table_path}, Rewards: {rewards_path}")


if __name__ == "__main__":
    main()