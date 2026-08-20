import json

from interaction.game_state_server import start_game_state_server

RED = 'red'
BLUE = 'blue'
NEUTRAL = 'neutral'
ASSASSIN = 'assassin'

TOTAL_BLUE = 8
TOTAL_RED = 7
INITIAL_RED = 2

CARD_DESCRIPTIONS_PATH = "../assets/card_descriptions.json"


class GameState:
    def __init__(self, board):
        self.board = board  # list of card filenames
        self.card_descriptions = json.load(open(CARD_DESCRIPTIONS_PATH))
        self.revealed = {}  # idx -> blue | red | neutral | assassin
        self.history = []  # structured guess history
        self.confidence_history = []  # confidence level per turn
        self.turn = 0
        self.game_over = False
        self.win = None

        start_game_state_server(self)

    def is_revealed(self, idx):
        return idx in self.revealed

    def reveal_card(self, idx, team):
        """
        team: 'red' | 'blue' | 'neutral' | 'assassin'
        """
        if self.game_over:
            return False

        if not self.is_valid_team(team):
            return False

        self.revealed[idx] = team
        print(f"[GAMESTATE] Card {idx} revealed as {team}")
        return True

    @staticmethod
    def is_valid_team(team):
        return team in {RED, BLUE, NEUTRAL, ASSASSIN}

    def unreveal_card(self, idx):
        if idx in self.revealed:
            prev = self.revealed[idx]
            del self.revealed[idx]
            print(f"[GAMESTATE] Card {idx} unrevealed (was {prev})")
            return True
        return False

    def are_initial_red_cards_placed(self):
        red_placed = sum(1 for color in self.revealed.values() if color == RED)
        return red_placed == INITIAL_RED
