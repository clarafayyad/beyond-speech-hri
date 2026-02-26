
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
        return team in {'red', 'blue', 'assassin', 'neutral'}

    def unreveal_card(self, idx):
        if idx in self.revealed:
            prev = self.revealed[idx]
            del self.revealed[idx]
            print(f"[GAMESTATE] Card {idx} unrevealed (was {prev})")
            return True
        return False