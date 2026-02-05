class Spymaster:
    def __init__(self, board, map_config):
        self.board = board
        self.map = map_config
        self.blue_indices = map_config["blue"]
        self.assassin_index = map_config["assassin"]
        self.red_indices = map_config["red"]
        self.neutral_indices = map_config["neutral"]

    def is_blue(self, idx):
        return idx in self.blue_indices

    def is_assassin(self, idx):
        return idx == self.assassin_index

    def is_red(self, idx):
        return idx in self.red_indices

    def is_neutral(self, idx):
        return idx in self.neutral_indices

    def get_board_summary(self):
        summary = {}
        for i, card in enumerate(self.board):
            if i in self.blue_indices:
                typ = "blue"
            elif i in self.red_indices:
                typ = "red"
            elif i == self.assassin_index:
                typ = "assassin"
            else:
                typ = "neutral"
            summary[card] = {"index": i, "type": typ}
        return summary