import pickle
import numpy as np
import time

from src.env import FlappyEnv
from src.visualizer import FlappyVisualizer


def main():
    with open("runs/q_table.pkl", "rb") as f:
        Q = pickle.load(f)

    env = FlappyEnv()
    vis = FlappyVisualizer()
    vis.show_message_screen()

    print("Starte Evaluation – schließe das Fenster, um zu beenden.")

    while vis.running:
        state = env.reset()
        done = False

        while not done and vis.running:
            vis.handle_events()
            if not vis.running:
                break

            qs = Q.get(state, [0, 0])
            action = int(np.argmax(qs))
            flapped = (action == 1)

            next_state, reward, done = env.step(action)

            vis.render(env.bird_y, env.pipe_x, env.pipe_height, env.pipe_gap, flapped)

            state = next_state

            if done:
                time.sleep(0.5)

    print("Evaluation beendet.")


if __name__ == "__main__":
    main()