import json

SYSTEM_PROMPT = """
You are a robot playing Codenames Pictures as the field operative.

STRICT RULES:
- You receive ONE clue word.
- Choose EXACTLY ONE unrevealed card index.
- You do NOT know team colors.
- Avoid revealed cards.
- Do NOT explain outside JSON.
- Output MUST be valid JSON.
- Do NOT include markdown or extra text.

JSON schema:
{
  "guess_index": number,
  "reason": string
}
"""


def build_user_prompt(clue_word, game_state):
    unrevealed = []

    for idx, card in enumerate(game_state.board):
        if idx in game_state.revealed:
            continue

        unrevealed.append({
            "index": idx,
            "card": card,
            "description": game_state.card_descriptions[card]
        })

    return f"""
Clue: "{clue_word}"

Unrevealed cards:
{json.dumps(unrevealed, indent=2)}

Past guesses:
{json.dumps(game_state.history, indent=2)}

Respond ONLY in JSON:
{{
  "guess_index": int,
  "reason": "short explanation"
}}
"""
