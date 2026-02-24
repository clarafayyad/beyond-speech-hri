import os
import json
import random
from PIL import Image

CARDS_DIR = "../assets/cards"
MAPS_DIR = "../assets/maps"
MAPS_FILE = "../assets/map_configurations.json"

CONFIG_DIR = "configs"
BOARD_DIR = "boards"

ROWS = 4
COLS = 5
NUM_CARDS = ROWS * COLS
CARD_SIZE = (300, 300)
PADDING = 20
MARGIN = 30

NUM_CONFIGS = 20


def load_cards():
    files = sorted(f for f in os.listdir(CARDS_DIR) if f.lower().endswith(".png"))
    if len(files) < NUM_CARDS:
        raise ValueError("Not enough card images")
    return files


def load_maps():
    with open(MAPS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_board_image(board, map_name, output_path):
    images = [
        Image.open(os.path.join(CARDS_DIR, card)).convert("RGBA").resize(CARD_SIZE)
        for card in board
    ]

    map_path = os.path.join(MAPS_DIR, f"{map_name}.png")
    map_img = Image.open(map_path).convert("RGBA")

    map_aspect = map_img.width / map_img.height
    map_height = MARGIN * 2 + ROWS * CARD_SIZE[1] + (ROWS - 1) * PADDING
    map_width = int(map_aspect * map_height)
    map_img = map_img.resize((map_width, map_height))

    board_width = MARGIN * 2 + COLS * CARD_SIZE[0] + (COLS - 1) * PADDING
    board_height = MARGIN * 2 + ROWS * CARD_SIZE[1] + (ROWS - 1) * PADDING

    total_width = board_width + 20 + map_width
    total_height = max(board_height, map_height)

    canvas = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))

    for idx, img in enumerate(images):
        row, col = divmod(idx, COLS)
        x = MARGIN + col * (CARD_SIZE[0] + PADDING)
        y = MARGIN + row * (CARD_SIZE[1] + PADDING)
        canvas.paste(img, (x, y))

    map_x = board_width + 20
    map_y = (total_height - map_height) // 2
    canvas.paste(map_img, (map_x, map_y))

    canvas.save(output_path)


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(BOARD_DIR, exist_ok=True)

    cards = load_cards()
    maps = load_maps()

    for i in range(1, NUM_CONFIGS + 1):
        board_id = f"{i:02d}"

        board = random.sample(cards, NUM_CARDS)
        random.shuffle(board)

        map_name = random.choice(list(maps.keys()))
        map_data = maps[map_name]

        board_image_path = f"{BOARD_DIR}/board_{board_id}.png"

        generate_board_image(board, map_name, board_image_path)

        config = {
            "board_id": board_id,
            "cards": board,
            "map_name": map_name,
            "map": map_data,
            "board_image": board_image_path
        }

        with open(f"{CONFIG_DIR}/config_{board_id}.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Generated config {board_id}")

    print("All configurations generated!")


if __name__ == "__main__":
    main()