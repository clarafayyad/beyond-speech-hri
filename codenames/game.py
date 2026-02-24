import os
import json
import random
from PIL import Image

from codenames.spymaster import Spymaster

CONFIG_DIR = "../assets/configs"
CARDS_DIR = "../assets/cards"
BOARD_DIR = "../assets"


def _load_random_config():
    files = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
    if not files:
        raise RuntimeError("No board configurations found")

    chosen = random.choice(files)

    with open(os.path.join(CONFIG_DIR, chosen), "r", encoding="utf-8") as f:
        return json.load(f)


class CodenamesGame:
    def __init__(self):
        print("\nSETTING UP GAME")

        config = _load_random_config()

        self.board_id = config["board_id"]
        self.board = config["cards"]
        self.map_name = config["map_name"]
        self.map = config["map"]
        self.board_image = config["board_image"]

        # Show the chosen configuration
        Image.open(os.path.join(BOARD_DIR, self.board_image)).show()

        self.spymaster = Spymaster(self.board, self.map)

        print(f"Loaded board {self.board_id}")
        print(f"Board image: {self.board_image}")
        print("GAME SETUP COMPLETE!")

    def get_card_path(self, idx):
        return os.path.join(CARDS_DIR, self.board[idx])

    def reveal_board(self):
        print("Board (row-wise indices 0–19):")
        for idx, card in enumerate(self.board):
            print(f"{idx}: {card}")

    def get_spymaster_view(self):
        return self.spymaster.blue_indices, self.spymaster.assassin_index