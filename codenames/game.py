import os
import json
import random
from PIL import Image

from codenames.spymaster import Spymaster

CARDS_DIR = "../assets/cards"
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
        self.cards = self._load_cards()
        self.board = self._pick_board()
        self.map_name, self.map = self._pick_map()
        self.spymaster = Spymaster(self.board, self.map)

        # generate and save board image
        self.generate_board_image()

    def _load_cards(self):
        files = sorted(f for f in os.listdir(CARDS_DIR) if f.lower().endswith(".png"))
        if len(files) < NUM_CARDS:
            raise ValueError("Need at least 20 cards")
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

    def generate_board_image(self):
        # Load card images
        # TODO: this is taking too long!
        images = [
            Image.open(os.path.join(CARDS_DIR, card)).convert("RGBA").resize(CARD_SIZE)
            for card in self.board
        ]

        # Load map image
        map_path = os.path.join("../assets/maps", f"{self.map_name}.png")
        map_img = Image.open(map_path).convert("RGBA")
        # Resize map to match board height while keeping aspect ratio
        map_aspect = map_img.width / map_img.height
        map_height = MARGIN * 2 + ROWS * CARD_SIZE[1] + (ROWS - 1) * PADDING
        map_width = int(map_aspect * map_height)
        map_img = map_img.resize((map_width, map_height))

        # Calculate board size
        board_width = MARGIN * 2 + COLS * CARD_SIZE[0] + (COLS - 1) * PADDING
        board_height = MARGIN * 2 + ROWS * CARD_SIZE[1] + (ROWS - 1) * PADDING

        # Total canvas size (board + spacing + map)
        total_width = board_width + 20 + map_width  # 20 px spacing
        total_height = max(board_height, map_height)
        canvas = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))

        # Paste board
        for idx, img in enumerate(images):
            row = idx // COLS
            col = idx % COLS
            x = MARGIN + col * (CARD_SIZE[0] + PADDING)
            y = MARGIN + row * (CARD_SIZE[1] + PADDING)
            canvas.paste(img, (x, y))

        # Paste map to the right of the board
        map_x = board_width + 20
        map_y = (total_height - map_height) // 2
        canvas.paste(map_img, (map_x, map_y))

        # canvas.save("board_with_map.png")
        canvas.show()
        print(f"Board + map saved to board_with_map.png (map: {self.map_name})")

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
