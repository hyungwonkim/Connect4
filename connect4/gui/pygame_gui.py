import pygame

from connect4.board import Board, ROWS, COLS, EMPTY, P1, P2
from connect4.players.human import HumanPlayer
from connect4.players.random_player import RandomPlayer
from connect4.players.greedy_player import GreedyPlayer
from connect4.players.epsilon_greedy_player import EpsilonGreedyPlayer
from connect4.players.rl.alphazero.alphazero_player import AlphaZeroPlayer
from connect4.players.rl.ppo.ppo_player import PPOPlayer
from connect4.players.rl.dqn.dqn_player import DQNPlayer

# Layout
CELL_SIZE = 100
HEADER_HEIGHT = 100
WINDOW_WIDTH = COLS * CELL_SIZE
WINDOW_HEIGHT = HEADER_HEIGHT + ROWS * CELL_SIZE
PIECE_RADIUS = CELL_SIZE // 2 - 8

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 200)
RED = (220, 20, 20)
YELLOW = (220, 220, 0)
GRAY = (60, 60, 60)
LIGHT_GRAY = (100, 100, 100)
OVERLAY_COLOR = (0, 0, 0, 150)

PLAYER_COLORS = {P1: RED, P2: YELLOW}

# States
STATE_START = 0
STATE_CHOOSE_FIRST = 1
STATE_PLAYING = 2
STATE_GAME_OVER = 3

# Animation
ANIM_SPEED = 25  # pixels per frame

OPPONENT_KEYS = ["Human", "Random", "Greedy", "Epsilon-Greedy", "AlphaZero", "PPO", "DQN"]


class PygameGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Connect 4")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont(None, 64)
        self.font_medium = pygame.font.SysFont(None, 40)
        self.font_small = pygame.font.SysFont(None, 32)

        self.state = STATE_START
        self.board = None
        self.current_player = P1
        self.human_player_id = P1
        self.opponent_key = None
        self.ai_player = None
        self.winner = None

        # Animation state
        self.animating = False
        self.anim_col = 0
        self.anim_row = 0
        self.anim_player = P1
        self.anim_y = 0.0
        self.anim_target_y = 0.0

        # AI turn pending (set after human move animation completes)
        self.ai_pending = False

        # Button rects (built in draw methods)
        self.start_buttons = []
        self.first_buttons = []
        self.play_again_rect = None
        self.quit_rect = None

    def _make_opponent(self, key, player_id):
        builders = {
            "Human": lambda: HumanPlayer("Player 2"),
            "Random": lambda: RandomPlayer(),
            "Greedy": lambda: GreedyPlayer(player_id),
            "Epsilon-Greedy": lambda: EpsilonGreedyPlayer(player_id),
            "AlphaZero": lambda: AlphaZeroPlayer(player_id, "checkpoints/alphazero/best.pt"),
            "PPO": lambda: PPOPlayer(player_id, "checkpoints/ppo/best.pt"),
            "DQN": lambda: DQNPlayer(player_id, "checkpoints/dqn/best.pt"),
        }
        return builders[key]()

    def _is_human_turn(self):
        return self.current_player == self.human_player_id or self.opponent_key == "Human"

    def _select_opponent(self, key):
        self.opponent_key = key
        if key == "Human":
            self._start_game(human_first=True)
        else:
            self.state = STATE_CHOOSE_FIRST

    def _start_game(self, human_first):
        self.human_player_id = P1 if human_first else P2
        ai_player_id = P2 if human_first else P1
        self.board = Board()
        self.current_player = P1
        self.winner = None
        self.animating = False
        self.ai_pending = False
        if self.opponent_key != "Human":
            self.ai_player = self._make_opponent(self.opponent_key, ai_player_id)
        else:
            self.ai_player = None
        self.state = STATE_PLAYING
        if not self._is_human_turn():
            self.ai_pending = True

    def _execute_move(self, col):
        row = self.board.drop_piece(col, self.current_player)
        self.animating = True
        self.anim_col = col
        self.anim_row = row
        self.anim_player = self.current_player
        self.anim_y = 0.0
        self.anim_target_y = float(HEADER_HEIGHT + row * CELL_SIZE + CELL_SIZE // 2)

    def _finish_move(self):
        winner = self.board.check_winner()
        if winner:
            self.winner = winner
            self.state = STATE_GAME_OVER
        elif self.board.is_draw():
            self.winner = None
            self.state = STATE_GAME_OVER
        else:
            self.current_player = P2 if self.current_player == P1 else P1
            if not self._is_human_turn():
                self.ai_pending = True

    def _col_from_mouse(self, x):
        col = x // CELL_SIZE
        if 0 <= col < COLS:
            return col
        return None

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    running = self._handle_click(event.pos)

            self._update()
            self._draw()
            self.clock.tick(60)
        pygame.quit()

    def _handle_click(self, pos):
        x, y = pos

        if self.state == STATE_START:
            for i, rect in enumerate(self.start_buttons):
                if rect.collidepoint(x, y):
                    self._select_opponent(OPPONENT_KEYS[i])
                    break

        elif self.state == STATE_CHOOSE_FIRST:
            for i, rect in enumerate(self.first_buttons):
                if rect.collidepoint(x, y):
                    self._start_game(human_first=(i == 0))
                    break

        elif self.state == STATE_PLAYING:
            if not self.animating and not self.ai_pending and self._is_human_turn():
                col = self._col_from_mouse(x)
                if col is not None and self.board.is_valid_move(col):
                    self._execute_move(col)

        elif self.state == STATE_GAME_OVER:
            if self.play_again_rect and self.play_again_rect.collidepoint(x, y):
                self.state = STATE_START
            elif self.quit_rect and self.quit_rect.collidepoint(x, y):
                return False  # signal quit

        return True  # keep running

    def _update(self):
        if self.animating:
            self.anim_y += ANIM_SPEED
            if self.anim_y >= self.anim_target_y:
                self.anim_y = self.anim_target_y
                self.animating = False
                self._finish_move()

        elif self.ai_pending and self.state == STATE_PLAYING:
            self.ai_pending = False
            # Draw "Thinking..." before blocking AI call
            self._draw()
            pygame.display.flip()
            col = self.ai_player.choose_move(self.board)
            self._execute_move(col)

    def _draw(self):
        self.screen.fill(BLACK)
        if self.state == STATE_START:
            self._draw_start_screen()
        elif self.state == STATE_CHOOSE_FIRST:
            self._draw_choose_first()
        elif self.state == STATE_PLAYING:
            self._draw_playing()
        elif self.state == STATE_GAME_OVER:
            self._draw_game_over()
        pygame.display.flip()

    # -- Start Screen --

    def _draw_start_screen(self):
        title = self.font_large.render("Connect 4", True, WHITE)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 40))

        subtitle = self.font_small.render("Select your opponent:", True, WHITE)
        self.screen.blit(subtitle, (WINDOW_WIDTH // 2 - subtitle.get_width() // 2, 120))

        self.start_buttons = []
        btn_w, btn_h = 300, 50
        start_y = 180
        mx, my = pygame.mouse.get_pos()

        for i, key in enumerate(OPPONENT_KEYS):
            rect = pygame.Rect(WINDOW_WIDTH // 2 - btn_w // 2, start_y + i * (btn_h + 12), btn_w, btn_h)
            hovered = rect.collidepoint(mx, my)
            color = LIGHT_GRAY if hovered else GRAY
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            label = self.font_small.render(key, True, WHITE)
            self.screen.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2))
            self.start_buttons.append(rect)

    # -- Choose First Screen --

    def _draw_choose_first(self):
        title = self.font_large.render("Connect 4", True, WHITE)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 40))

        vs_text = self.font_medium.render(f"vs {self.opponent_key}", True, WHITE)
        self.screen.blit(vs_text, (WINDOW_WIDTH // 2 - vs_text.get_width() // 2, 120))

        subtitle = self.font_small.render("Who goes first?", True, WHITE)
        self.screen.blit(subtitle, (WINDOW_WIDTH // 2 - subtitle.get_width() // 2, 200))

        self.first_buttons = []
        btn_w, btn_h = 300, 60
        labels = ["You (Red)", f"{self.opponent_key} (Red)"]
        start_y = 280
        mx, my = pygame.mouse.get_pos()

        for i, label in enumerate(labels):
            rect = pygame.Rect(WINDOW_WIDTH // 2 - btn_w // 2, start_y + i * (btn_h + 20), btn_w, btn_h)
            hovered = rect.collidepoint(mx, my)
            color = LIGHT_GRAY if hovered else GRAY
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            txt = self.font_small.render(label, True, WHITE)
            self.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))
            self.first_buttons.append(rect)

    # -- Playing Screen --

    def _draw_playing(self):
        # Header
        if self.ai_pending:
            text = "Thinking..."
        elif self.animating:
            text = ""
        elif self._is_human_turn():
            color_name = "Red" if self.current_player == P1 else "Yellow"
            text = f"Your turn ({color_name}) — click a column"
        else:
            text = f"{self.opponent_key}'s turn..."

        if text:
            header = self.font_medium.render(text, True, WHITE)
            self.screen.blit(header, (WINDOW_WIDTH // 2 - header.get_width() // 2, HEADER_HEIGHT // 2 - header.get_height() // 2))

        # Hover indicator
        if not self.animating and not self.ai_pending and self._is_human_turn():
            mx, _ = pygame.mouse.get_pos()
            col = self._col_from_mouse(mx)
            if col is not None and self.board.is_valid_move(col):
                cx = col * CELL_SIZE + CELL_SIZE // 2
                cy = HEADER_HEIGHT // 2
                color = PLAYER_COLORS[self.current_player]
                hover_surface = pygame.Surface((PIECE_RADIUS * 2, PIECE_RADIUS * 2), pygame.SRCALPHA)
                pygame.draw.circle(hover_surface, (*color, 120), (PIECE_RADIUS, PIECE_RADIUS), PIECE_RADIUS)
                self.screen.blit(hover_surface, (cx - PIECE_RADIUS, cy - PIECE_RADIUS))

        self._draw_board()

    def _draw_board(self):
        # Blue board background
        board_rect = pygame.Rect(0, HEADER_HEIGHT, WINDOW_WIDTH, ROWS * CELL_SIZE)
        pygame.draw.rect(self.screen, BLUE, board_rect)

        for r in range(ROWS):
            for c in range(COLS):
                cx = c * CELL_SIZE + CELL_SIZE // 2
                cy = HEADER_HEIGHT + r * CELL_SIZE + CELL_SIZE // 2

                # Skip animated cell — draw it separately
                if self.animating and r == self.anim_row and c == self.anim_col:
                    pygame.draw.circle(self.screen, BLACK, (cx, cy), PIECE_RADIUS)
                    continue

                cell = self.board.grid[r][c]
                if cell == EMPTY:
                    pygame.draw.circle(self.screen, BLACK, (cx, cy), PIECE_RADIUS)
                else:
                    pygame.draw.circle(self.screen, PLAYER_COLORS[cell], (cx, cy), PIECE_RADIUS)

        # Draw animated piece
        if self.animating:
            cx = self.anim_col * CELL_SIZE + CELL_SIZE // 2
            ay = int(self.anim_y)
            pygame.draw.circle(self.screen, PLAYER_COLORS[self.anim_player], (cx, ay), PIECE_RADIUS)

    # -- Game Over Screen --

    def _draw_game_over(self):
        self._draw_board()

        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill(OVERLAY_COLOR)
        self.screen.blit(overlay, (0, 0))

        # Result text
        if self.winner:
            if self.opponent_key == "Human":
                name = "Red" if self.winner == P1 else "Yellow"
            elif self.winner == self.human_player_id:
                name = "You"
            else:
                name = self.opponent_key
            result_text = f"{name} win!" if name == "You" else f"{name} wins!"
            color = PLAYER_COLORS[self.winner]
        else:
            result_text = "Draw!"
            color = WHITE

        result = self.font_large.render(result_text, True, color)
        self.screen.blit(result, (WINDOW_WIDTH // 2 - result.get_width() // 2, WINDOW_HEIGHT // 2 - 80))

        # Buttons
        btn_w, btn_h = 200, 50
        self.play_again_rect = pygame.Rect(WINDOW_WIDTH // 2 - btn_w - 10, WINDOW_HEIGHT // 2 + 20, btn_w, btn_h)
        self.quit_rect = pygame.Rect(WINDOW_WIDTH // 2 + 10, WINDOW_HEIGHT // 2 + 20, btn_w, btn_h)

        mx, my = pygame.mouse.get_pos()
        for rect, label in [(self.play_again_rect, "Play Again"), (self.quit_rect, "Quit")]:
            hovered = rect.collidepoint(mx, my)
            pygame.draw.rect(self.screen, LIGHT_GRAY if hovered else GRAY, rect, border_radius=8)
            txt = self.font_small.render(label, True, WHITE)
            self.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))
