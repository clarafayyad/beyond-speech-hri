import os
import json
import random
from PIL import Image

from codenames.spymaster import Spymaster

CARDS_DIR = "../assets/cards"
MAPS_DIR = "../assets/maps"
MAPS_FILE = "../assets/map_configurations.json"
BOARD_OUTPUT_PATH = "board.png"

ROWS = 4
COLS = 5
NUM_CARDS = ROWS * COLS
CARD_SIZE = (300, 300)
PADDING = 20
MARGIN = 30


class CodenamesGame:
    def __init__(self):
        print("\n SETTING UP GAME")
        self.cards = self._load_cards()  # just file names
        self.board = self._pick_board()   # list of file names
        self.map_name, self.map = self._pick_map()
        self.spymaster = Spymaster(self.board, self.map)

        # generate and save board image
        self.generate_board_image()

        print("\n GAME SETUP COMPLETE!")

    def _load_cards(self):
        """Load only card file names, no images."""
        files = sorted(f for f in os.listdir(CARDS_DIR) if f.lower().endswith(".png"))
        if len(files) < NUM_CARDS:
            raise ValueError(f"Need at least {NUM_CARDS} cards")
        return files

    def _pick_board(self):
        selected = random.sample(self.cards, NUM_CARDS)
        random.shuffle(selected)
        return selected

    def _pick_map(self):
        with open(MAPS_FILE, "r", encoding="utf-8") as f:
            maps = json.load(f)
        name = random.choice(list(maps.keys()))
        return name, maps[name]

    def get_card_path(self, idx):
        """Return the full file path of the card at a given board index."""
        if idx < 0 or idx >= NUM_CARDS:
            raise IndexError(f"Card index {idx} out of range")
        return os.path.join(CARDS_DIR, self.board[idx])

    def generate_board_image(self):
        """Load images only when generating the board image."""
        images = [
            Image.open(self.get_card_path(idx)).convert("RGBA").resize(CARD_SIZE)
            for idx in range(NUM_CARDS)
        ]

        # Load map image
        map_path = os.path.join(MAPS_DIR, f"{self.map_name}.png")
        map_img = Image.open(map_path).convert("RGBA")
        map_aspect = map_img.width / map_img.height
        map_height = MARGIN * 2 + ROWS * CARD_SIZE[1] + (ROWS - 1) * PADDING
        map_width = int(map_aspect * map_height)
        map_img = map_img.resize((map_width, map_height))

        # Board dimensions
        board_width = MARGIN * 2 + COLS * CARD_SIZE[0] + (COLS - 1) * PADDING
        board_height = MARGIN * 2 + ROWS * CARD_SIZE[1] + (ROWS - 1) * PADDING

        # Canvas
        total_width = board_width + 20 + map_width
        total_height = max(board_height, map_height)
        canvas = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))

        # Paste cards
        for idx, img in enumerate(images):
            row, col = divmod(idx, COLS)
            x = MARGIN + col * (CARD_SIZE[0] + PADDING)
            y = MARGIN + row * (CARD_SIZE[1] + PADDING)
            canvas.paste(img, (x, y))

        # Paste map
        map_x = board_width + 20
        map_y = (total_height - map_height) // 2
        canvas.paste(map_img, (map_x, map_y))

        canvas.show()
        canvas.save(BOARD_OUTPUT_PATH)
        print(f"Board + map saved to {BOARD_OUTPUT_PATH} (map: {self.map_name})")

    def reveal_board(self):
        print("Board (row-wise indices 0-19):")
        for idx, card in enumerate(self.board):
            print(f"{idx}: {card}")

    def get_spymaster_view(self):
        return self.spymaster.blue_indices, self.spymaster.assassin_index


if __name__ == "__main__":
    game = CodenamesGame()
    game.reveal_board()

    blue, assassin = game.get_spymaster_view()
    print("\nSpymaster view:")
    print("Blue indices:", blue)
    print("Assassin index:", assassin)

    print("\nBoard summary:")
    for card, info in game.spymaster.get_board_summary().items():
        print(f"{card}: {info}")