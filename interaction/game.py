import os
import json
import random
from PIL import Image

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


def _load_config(config_number):
    config_str = str(config_number)
    if config_number < 10:
        config_str = "0" + config_str
    config_file_name = "config_" + config_str + ".json"
    with open(os.path.join(CONFIG_DIR, config_file_name), "r", encoding="utf-8") as f:
        return json.load(f)


class CodenamesGame:
    def __init__(self, config_number=None):
        print("\nSETTING UP GAME")

        if not config_number:
            print("Loading random board configuration...")
            config = _load_random_config()
        else:
            print(f"Loading board configuration #{config_number}...")
            config = _load_config(config_number)

        self.board_id = config["board_id"]
        self.board = config["cards"]
        self.map_name = config["map_name"]
        self.map = config["map"]
        self.board_image = config["board_image"]

        # Show the chosen configuration
        print("Displaying board image...")
        Image.open(os.path.join(BOARD_DIR, self.board_image)).show()

        print(f"Loaded board {self.board_id}")
        print(f"Board image: {self.board_image}")
        print("GAME SETUP COMPLETE!")

    def get_card_path(self, idx):
        return os.path.join(CARDS_DIR, self.board[idx])

    def reveal_board(self):
        print("Board (row-wise indices 0–19):")
        for idx, card in enumerate(self.board):
            print(f"{idx}: {card}")
