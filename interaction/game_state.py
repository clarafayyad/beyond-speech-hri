
class GameState:
    def __init__(self, board, card_descriptions):
        self.board = board                      # list of card filenames
        self.card_descriptions = card_descriptions
        self.revealed = {}                      # idx -> blue | red | neutral | assassin
        self.history = []                       # structured guess history
        self.turn = 0
        self.game_over = False
        self.win = None

    def is_revealed(self, idx):
        return idx in self.revealed