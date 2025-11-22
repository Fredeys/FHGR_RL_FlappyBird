import numpy as np
import matplotlib.pyplot as plt
import os

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")


def main():
    rewards_path = os.path.join(RUNS_DIR, "rewards.csv")
    rewards = np.loadtxt(rewards_path)

    # Moving average for smoothing
    window = 50
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        x_smooth = np.arange(window - 1, window - 1 + len(smoothed))
    else:
        smoothed = rewards
        x_smooth = np.arange(len(rewards))

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "learning_curve.png")

    plt.figure(figsize=(6, 4))
    plt.plot(rewards, alpha=0.3, label="Return per Episode")
    plt.plot(x_smooth, smoothed, label=f"Smoothing (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning Curve of the Q-Learning Agent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Learning curve saved to: {out_path}")


if __name__ == "__main__":
    main()