import json

INNER_DISCUSSION_SYSTEM_PROMPT = """
You are simulating a short two-player inner consultation in a Codenames Pictures game.

Players:
- Player 1 (Robot): asks a tentative question about a candidate card.
- Player 2 (Clara, consultant): raises a safety concern (e.g. assassin or wrong-team risk) and recommends a safer alternative.
- Final decision: Robot commits to the card Clara recommended.

STRICT RULES:
- Output MUST be valid JSON.
- Do NOT include markdown or extra text.
- player1_line: Robot asks about one candidate card (e.g. "Could it be 'Seal'?").
- player2_line: Clara expresses a concern about that card and names a safer alternative (e.g. "Maybe, but that risks the assassin. 'Bridge' is safer.").
- final_decision: Robot commits to the alternative card Clara suggested (e.g. "I'll go with 'Bridge'.").
- guess_index: The index of the card chosen in the final decision (the alternative Clara recommended). Must be an unrevealed card index.

JSON schema:
{
  "player1_line": string,
  "player2_line": string,
  "final_decision": string,
  "guess_index": number
}
"""


def build_inner_discussion_prompt(clue_word, game_state):
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

Generate a short inner consultation exchange: Player 1 proposes one candidate, Clara warns about its risk and recommends a safer card, then Robot commits to Clara's recommendation.
Respond ONLY in JSON.
"""


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

REASONING STYLE — adapt the "reason" field to the confidence level:
The "reason" field must be a natural spoken sentence (1–3 sentences) that the robot will say aloud.
Be creative and vary your phrasing across turns.
- high confidence: Use short, decisive language. State the choice directly with minimal justification. Do NOT include an interpretation statement.
  Examples: "Nice, that's clear. Bridge fits best—I'll go with that.",
            "Got it. This strongly points to Apple—selecting it.",
            "Easy. River is the obvious match here—going with that."
- medium confidence: Mention 2 candidate options and briefly justify your final choice. Then add one sentence stating your interpretation of the clue (e.g. "I think you meant X and Y with this clue."). No questions.
  Examples: "I'm between Bridge and Stream since both relate to River. I think you meant Bridge and Stream with this clue. Bridge feels stronger—I'll go with that.",
            "This could be Apple or Orange as both are fruits. I think you meant Apple and Orange with this clue. Apple seems more central here—I'll go with Apple."
- low confidence: Mention 2–3 hypotheses, express uncertainty and risk, add a slight hesitation phrase, and include a sentence stating your best guess at the intended interpretation (e.g. "I think you might be pointing to X and Y."). No questions.
  Examples: "This is tricky… I see Seal, Bridge, and Stream. I think you might be pointing to Bridge and Stream, but Seal makes it ambiguous. I'll cautiously choose Bridge.",
            "Hmm, not fully sure. It could be Bank or Bridge. I think you're leaning toward Bridge, but Bank is risky. I'll go with Bridge."
- unknown: Use a fixed, non-adaptive style regardless of difficulty. State your choice and the reason why you think it relates to the clue.

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
Current turn: {game_state.turn}
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
