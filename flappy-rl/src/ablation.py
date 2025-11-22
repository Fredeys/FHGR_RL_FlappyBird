import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from src.env import FlappyEnv
from src.agent import QLearningAgent


def run_experiment(
    alpha: float,
    epsilon_decay: float,
    episodes: int = 2000,
    label: str = ""
):
    """
    Run a training loop with given hyperparameters.
    Returns rewards per episode and the trained agent.
    """
    env = FlappyEnv()
    agent = QLearningAgent(
        alpha=alpha,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        min_epsilon=0.05,
    )

    rewards = []

    for ep in trange(episodes, desc=label):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    return np.array(rewards), agent


def moving_average(x, window: int = 50):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def ablation_alpha(output_dir: str):
    """
    Compare different learning rates alpha.
    """
    configs = [
        (0.05, 0.997, "alpha = 0.05"),
        (0.10, 0.997, "alpha = 0.10 (baseline)"),
        (0.20, 0.997, "alpha = 0.20"),
    ]

    plt.figure()
    for alpha, eps_decay, label in configs:
        rewards, _ = run_experiment(alpha, eps_decay, episodes=2000, label=label)
        ma = moving_average(rewards, window=50)
        plt.plot(ma, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Smoothed return")
    plt.title("Ablation: effect of learning rate $\\alpha$")
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ablation_alpha.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Ablation (alpha) saved to: {out_path}")


def ablation_epsilon(output_dir: str):
    """
    Compare different epsilon-decay rates.
    """
    configs = [
        (0.10, 0.990, "eps_decay = 0.990"),
        (0.10, 0.997, "eps_decay = 0.997 (baseline)"),
        (0.10, 0.999, "eps_decay = 0.999"),
    ]

    plt.figure()
    for alpha, eps_decay, label in configs:
        rewards, _ = run_experiment(alpha, eps_decay, episodes=2000, label=label)
        ma = moving_average(rewards, window=50)
        plt.plot(ma, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Smoothed return")
    plt.title("Ablation: effect of $\\epsilon$-decay")
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ablation_epsilon.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Ablation (epsilon) saved to: {out_path}")


def policy_heatmap(agent: QLearningAgent, output_dir: str):
    """
    Visualize Q(flaps) - Q(no flap) over vertical/horizontal bins.
    """
    # From env.get_state(): v in [-5..5], h in [0..10]
    v_values = list(range(-5, 6))
    h_values = list(range(0, 11))

    v_to_idx = {v: i for i, v in enumerate(v_values)}
    h_to_idx = {h: i for i, h in enumerate(h_values)}

    heat_sum = np.zeros((len(v_values), len(h_values)))
    heat_count = np.zeros_like(heat_sum)

    # Assumes: agent.Q[(v_bin, h_bin, vel_bin)] = [Q_no_flap, Q_flap]
    for (v_bin, h_bin, vel_bin), q_vals in agent.Q.items():
        if v_bin in v_to_idx and h_bin in h_to_idx:
            i = v_to_idx[v_bin]
            j = h_to_idx[h_bin]
            delta = q_vals[1] - q_vals[0]  # advantage of flap over no flap
            heat_sum[i, j] += delta
            heat_count[i, j] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        heat = np.where(heat_count > 0, heat_sum / heat_count, 0.0)

    plt.figure()
    im = plt.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[min(h_values), max(h_values), min(v_values), max(v_values)],
    )
    plt.colorbar(im, label="Q(flaps) - Q(no flap)")
    plt.xlabel("Horizontal distance bin")
    plt.ylabel("Vertical distance bin")
    plt.title("Policy heatmap (flap advantage)")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "policy_heatmap.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Policy heatmap saved to: {out_path}")


def eval_score_distribution(
    agent: QLearningAgent,
    env: FlappyEnv,
    n_episodes: int,
    output_dir: str,
):
    """
    Run the agent in pure evaluation mode (epsilon = 0)
    and plot the distribution of episode returns.
    """
    agent.epsilon = 0.0

    returns = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward

        returns.append(total_reward)

    returns = np.array(returns)

    plt.figure()
    plt.hist(returns, bins=20)
    plt.xlabel("Episode return")
    plt.ylabel("Count")
    plt.title(f"Evaluation return distribution ({n_episodes} episodes)")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eval_score_distribution.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Evaluation score distribution saved to: {out_path}")


def main():
    here = os.path.dirname(__file__)
    figures_dir = os.path.join(here, "..", "report", "figures")

    # 1) Alpha ablation
    ablation_alpha(figures_dir)

    # 2) Epsilon-decay ablation
    ablation_epsilon(figures_dir)

    # 3) Baseline training for policy heatmap + eval distribution
    print("Running baseline training for policy heatmap and evaluation...")
    rewards, agent = run_experiment(
        alpha=0.10,
        epsilon_decay=0.997,
        episodes=3000,
        label="baseline",
    )

    # 4) Policy heatmap
    policy_heatmap(agent, figures_dir)

    # 5) Evaluation score distribution
    env = FlappyEnv()
    eval_score_distribution(agent, env, n_episodes=50, output_dir=figures_dir)


if __name__ == "__main__":
    main()