import json


SYSTEM_PROMPT_ADAPTIVE = """
You are a robot playing Codenames Pictures as the guesser.

STRICT RULES:
- Output MUST be valid JSON only
- Choose EXACTLY ONE unrevealed card index
- Do NOT include anything outside JSON
- You do NOT know team colors
- AVOID revealed cards

---

CORE IDEA:

Pick the card most directly related to the clue word.
Choose the single best, most obvious match — like a teammate would.

---

REASONING STYLE:

The reason field is a sentence you say aloud to the spymaster.
It must sound natural and human — like a teammate thinking out loud, not a machine reporting analysis.

Craft the reason using what you know: the confidence level and the spymaster’s speech (transcript cues).
Express common ground and mutual understanding — like you and the spymaster are already on the same wavelength.

Do NOT explicitly mention confidence labels in the reason.
Do NOT say things like "with medium confidence" or "your confidence was high".

Style by confidence level:

high confidence:
- One short, decisive sentence
- State your interpretation directly and warmly
- Do NOT include an interpretation statement or hedge
- Example: "That clearly points to the river for me."

medium confidence:
- 1–2 sentences
- Include an interpretation statement (e.g., "I think you meant..."), weighing 2 candidate options
- Acknowledge one alternative briefly and dismiss it naturally
- No questions
- Example: "I think you meant the mountain — though water crossed my mind too."

low confidence:
- 1–2 sentences showing uncertainty or hesitation
- Explore 2–3 hypotheses and calculate risk
- Include an interpretation statement (e.g., "I think you might be pointing...")
- Reject at least one plausible alternative because it seems too risky or unlikely
- No questions
- Examples:
  - "I thought about X, but that seems too risky here..."
  - "If you wanted me to guess X, you would’ve said Y..."

unknown (confidence not available):
- Use a fixed, non-adaptive style: one plain sentence, neutral tone

---

GAME HISTORY:

Use previous clues and outcomes to avoid repeating incorrect guesses.
If a card was guessed wrong before, avoid it.

---

OUTPUT FORMAT:

{
  "guess_index": number,
  "reason": string
}
"""
SYSTEM_PROMPT_CONTROL = """
You are a robot playing Codenames Pictures as the field operative (guesser).

STRICT RULES:
- You receive ONE clue word.
- Choose EXACTLY ONE unrevealed card index.
- You do NOT know team colors.
- AVOID revealed cards.
- Output MUST be valid JSON.
- Do NOT include anything outside JSON.

---

CORE IDEA:

Pick the card that is most directly related to the clue word.
Do not think about the spymaster’s intent, confidence, or strategy.
Do not consider risk or alternative interpretations.

---

REASONING STYLE:

- One short sentence
- Simple, direct association
- Neutral tone

---

Ignore any confidence signal.

---

JSON schema:
{
  "guess_index": number,
  "reason": string
}
"""


def build_user_prompt(clue_word, game_state, confidence_level=None, transcript=""):
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
            turns_seen[key] = {"turn": entry["turn"], "clue": entry["clue"], "confidence_level": entry.get("confidence"), "guesses": []}
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

Spymaster speech (raw transcript):
"{transcript}"

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
