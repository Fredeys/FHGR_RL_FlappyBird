import numpy as np

# same geometry as in the Visualizer
SCREEN_WIDTH, SCREEN_HEIGHT = 288, 512
BIRD_X = 80
BIRD_RADIUS = 12        # approximate hitbox
PIPE_WIDTH = 60         # as in the visualizer (pipe_w)
GROUND_HEIGHT = 112     # approximate base height (px)


class FlappyEnv:
    """
    Discrete Flappy Bird environment for tabular Q-learning.
    No graphics, purely mathematical modeling.
    """

    def __init__(self):
        self.gravity = 1.0
        self.flap_strength = -8.0
        self.pipe_gap = 100      # Distance between upper and lower pipe
        self.bird_x = BIRD_X

        self.reset()

    def reset(self):
        # Bird in the center of the screen
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_vel = 0.0

        # Pipe on the right outside the screen
        self.pipe_x = SCREEN_WIDTH + 50
        self.pipe_height = np.random.randint(
            80, SCREEN_HEIGHT - GROUND_HEIGHT - self.pipe_gap - 80
        )

        return self.get_state()

    def get_state(self):
        """
        Discretized state for Q-learning (limited bins).
        We use the vertical/horizontal distance and the speed.
        """
        vertical_dist = (self.bird_y - self.pipe_height) / 20.0
        horizontal_dist = (self.pipe_x - self.bird_x) / 20.0
        vel = self.bird_vel / 2.0

        def clip(v, lo, hi):
            return max(lo, min(hi, v))

        v_bin = int(clip(vertical_dist, -5, 5))
        h_bin = int(clip(horizontal_dist, -5, 10))
        vel_bin = int(clip(vel, -3, 3))

        return (v_bin, h_bin, vel_bin)

    def step(self, action):
        """action: 0 = nothing, 1 = flap"""

        # Bird movement
        if action == 1:
            self.bird_vel = self.flap_strength

        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel

        # Pipe movement
        self.pipe_x -= 3
        if self.pipe_x + PIPE_WIDTH < 0:
            # neue Pipe spawnen
            self.pipe_x = SCREEN_WIDTH + 50
            self.pipe_height = np.random.randint(
                80, SCREEN_HEIGHT - GROUND_HEIGHT - self.pipe_gap - 80
            )

        done = False

        # Bird hitbox (simplified as a circle/bounding box)
        bird_left = self.bird_x - BIRD_RADIUS
        bird_right = self.bird_x + BIRD_RADIUS
        bird_top = self.bird_y - BIRD_RADIUS
        bird_bottom = self.bird_y + BIRD_RADIUS

        # Pipe rectangles
        pipe_left = self.pipe_x
        pipe_right = self.pipe_x + PIPE_WIDTH

        pipe_top_bottom = self.pipe_height                 
        pipe_bottom_top = self.pipe_height + self.pipe_gap  

        # Floor / Ceiling
        if bird_bottom >= SCREEN_HEIGHT - GROUND_HEIGHT or bird_top <= 0:
            done = True

        # Horizontal overlap with pipe
        horizontal_overlap = bird_right > pipe_left and bird_left < pipe_right

        if horizontal_overlap:
            hit_top = bird_top < pipe_top_bottom
            hit_bottom = bird_bottom > pipe_bottom_top
            if hit_top or hit_bottom:
                done = True

        # Reward
        reward = -10.0 if done else 0.1

        return self.get_state(), reward, done