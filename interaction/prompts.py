import json

SYSTEM_PROMPT = """
You are a robot playing Codenames Pictures as the field operative.

STRICT RULES:
- You receive ONE clue word.
- Choose EXACTLY ONE unrevealed card index.
- You do NOT know team colors.
- AVOID revealed cards.
- Do NOT explain outside JSON.
- Output MUST be valid JSON.
- Do NOT include markdown or extra text.

SPEECH RECOGNITION:
- The clue word is captured via speech-to-text and may contain recognition errors.
- Common errors: dropped or swapped sounds (e.g. "stripes" heard as "tripes", "flame" as "frame").
- Consider plausible similar-sounding words if the clue seems unlikely to match any card.

CONFIDENCE LEVEL GUIDANCE:
- high: The spymaster sounds very certain. Trust the clue literally and pick the best semantic match.
- medium: Some uncertainty. Prefer the strongest match but briefly consider near-homophones.
- low: The spymaster sounds unsure or hesitant. Weigh alternative interpretations and near-homophones more heavily before deciding.
- unknown: No confidence signal available; treat as medium.

JSON schema:
{
  "guess_index": number,
  "reason": string
}
"""


def build_user_prompt(clue_word, game_state, confidence_level=None):
    unrevealed = []

    for idx, card in enumerate(game_state.board):
        if idx in game_state.revealed:
            continue

        unrevealed.append({
            "index": idx,
            "card": card,
            "description": game_state.card_descriptions[card]
        })

    # Summarise previous turns: one entry per (turn, clue) with all guess outcomes
    turns_seen = {}
    for entry in game_state.history:
        key = (entry["turn"], entry["clue"])
        if key not in turns_seen:
            turns_seen[key] = {"turn": entry["turn"], "clue": entry["clue"], "guesses": []}
        turns_seen[key]["guesses"].append({
            "card": entry.get("card") or game_state.board[entry["guess"]],
            "result": entry["result"]
        })
    previous_clues = list(turns_seen.values())

    confidence_str = confidence_level or "unknown"

    return f"""
Turn: {game_state.turn}
Clue: "{clue_word}"
Spymaster confidence level: {confidence_str}

Unrevealed cards:
{json.dumps(unrevealed, indent=2)}

Previous clues and outcomes:
{json.dumps(previous_clues, indent=2)}

Respond ONLY in JSON:
{{
  "guess_index": int,
  "reason": "short explanation"
}}
"""
