import pygame
import os

ASSET_PATH = os.path.join(os.path.dirname(__file__), "..", "assets")

SCREEN_WIDTH, SCREEN_HEIGHT = 288, 512


class FlappyVisualizer:
    def __init__(self, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT):
        pygame.init()

        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Flappy Bird – RL Evaluation")

        self.clock = pygame.time.Clock()
        self.running = True

        self._load_assets()

        # Bottom-Scroll
        self.base_x = 0

        # Bird-Animationsstate
        self.flap_timer = 0          # Duration of animation after flap
        self.up_frame = 0
        self.mid_frame = 1
        self.down_frame = 2

    def _load_assets(self):
        # Background
        bg_raw = pygame.image.load(os.path.join(ASSET_PATH, "background-day.png")).convert()
        self.background = pygame.transform.scale(bg_raw, (self.width, self.height))

        # Base
        self.base = pygame.image.load(os.path.join(ASSET_PATH, "base.png")).convert_alpha()
        self.base_width = self.base.get_width()
        self.base_height = self.base.get_height()

        # Pipes
        pipe = pygame.image.load(os.path.join(ASSET_PATH, "pipe-green.png")).convert_alpha()
        pipe_w = 60
        pipe_h = int(pipe.get_height() * (pipe_w / pipe.get_width()))
        self.pipe_bottom = pygame.transform.scale(pipe, (pipe_w, pipe_h))
        self.pipe_top = pygame.transform.flip(self.pipe_bottom, False, True)
        self.pipe_width = pipe_w
        self.pipe_height = pipe_h

        # Bird frames
        bird_up = pygame.image.load(os.path.join(ASSET_PATH, "bluebird-upflap.png")).convert_alpha()
        bird_mid = pygame.image.load(os.path.join(ASSET_PATH, "bluebird-midflap.png")).convert_alpha()
        bird_down = pygame.image.load(os.path.join(ASSET_PATH, "bluebird-downflap.png")).convert_alpha()

        scale = 1.1
        def scale_bird(img):
            w = int(img.get_width() * scale)
            h = int(img.get_height() * scale)
            return pygame.transform.scale(img, (w, h))

        self.bird_frames = [
            scale_bird(bird_up),
            scale_bird(bird_mid),
            scale_bird(bird_down)
        ]
        self.bird_width = self.bird_frames[0].get_width()
        self.bird_height = self.bird_frames[0].get_height()
        self.bird_x = 80

        # Message screen
        msg_raw = pygame.image.load(os.path.join(ASSET_PATH, "message.png")).convert_alpha()
        msg_w = int(self.width * 0.8)
        sf = msg_w / msg_raw.get_width()
        msg_h = int(msg_raw.get_height() * sf)
        self.message_img = pygame.transform.scale(msg_raw, (msg_w, msg_h))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def draw_base(self):
        y = self.height - self.base_height

        self.screen.blit(self.base, (self.base_x, y))
        self.screen.blit(self.base, (self.base_x + self.base_width, y))

        self.base_x -= 2
        if self.base_x <= -self.base_width:
            self.base_x = 0

    def _select_bird_frame(self):
        """
        Selects dependent on flap_timer the matching frame:
        - after Flap: upflap
        - shortly after: midflap
        - later: downflap (gliding/falling)
        """
        if self.flap_timer > 4:
            return self.bird_frames[self.up_frame]
        elif self.flap_timer > 2:
            return self.bird_frames[self.mid_frame]
        elif self.flap_timer > 0:
            return self.bird_frames[self.down_frame]
        else:
            # kein aktueller Flap → Standardpose (leicht fallend)
            return self.bird_frames[self.down_frame]

    def render(self, bird_y, pipe_x, pipe_height, pipe_gap, flapped: bool):
        # Update Flap-Status
        if flapped:
            self.flap_timer = 6
        elif self.flap_timer > 0:
            self.flap_timer -= 1

        # Background
        self.screen.blit(self.background, (0, 0))

        # Pipes
        self.screen.blit(self.pipe_top, (pipe_x, pipe_height - self.pipe_height))
        self.screen.blit(self.pipe_bottom, (pipe_x, pipe_height + pipe_gap))

        # Bird
        clamped_y = max(0, min(self.height - self.base_height - self.bird_height, bird_y))
        bird_surf = self._select_bird_frame()
        self.screen.blit(bird_surf, (self.bird_x, int(clamped_y)))

        # Bottom
        self.draw_base()

        pygame.display.flip()
        self.clock.tick(30)

    def show_message_screen(self):
        """Displays Start Message and waits for click."""
        waiting = True

        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False

            self.screen.blit(self.background, (0, 0))

            msg_x = (self.width - self.message_img.get_width()) // 2
            msg_y = int(self.height * 0.15)
            self.screen.blit(self.message_img, (msg_x, msg_y))

            self.draw_base()

            pygame.display.flip()
            self.clock.tick(30)