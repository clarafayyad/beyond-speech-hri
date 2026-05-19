# 🤖 Beyond Speech HRI

A research project exploring how a social robot can leverage non-verbal cues to dynamically adapt its behavior during collaborative human–robot interaction (HRI).

---

## Overview

This project investigates how a social robot can move beyond spoken language by using non-verbal signals (timing, prosody, turn-taking, contextual awareness) to adapt its behavior during collaborative tasks with humans.

The system integrates conversational AI, speech synthesis, and real-time interaction services to support adaptive, socially aware robot behavior.

---

## Repository Structure, Sub-repositories, and Runtime Services

### Key folders

- `src/`: entry points (for example `src/main.py`) and launch scripts.
- `agents/`: conversational robot logic (dialog, guessing, and speech behavior).
- `interaction/`: game loop, turn handling, state management, logging, prompts, and the game-state web UI server.
- `multimodal_perception/`: audio/video feature extraction and confidence classification.
- `assets/`: cards, boards, maps, and board configuration files (`assets/configs`).
- `tests/`: automated tests for parsing, prompts, logging, and interaction behavior.
- `logs/`: experiment and utterance logs.
- `sic_applications/`: local runtime files used by Social Interaction Cloud services (for example Redis config).

### Sub-repositories

This repo includes the `RealtimeSTT/` git submodule (`KoljaB/RealtimeSTT`) for streaming speech-to-text support.

### External services and APIs

The system depends on external services and APIs that run together during experiments:

- **OpenAI API** (LLM-based reasoning/guessing)
- **ElevenLabs API** (text-to-speech voice synthesis)
- **Redis** (required by Social Interaction Cloud services)
- **Social Interaction Cloud services** such as `run-dialogflow` and `run-gpt` (optional, depending on setup)

---
## A. Setup Guide

Follow the steps below in order to ensure a correct setup.


### 1. OpenAI API Key
 
Generate a personal OpenAI API key:
   https://platform.openai.com/api-keys

### 2. ElevenLabs API Key

Generate an ElevenLabs API key from your ElevenLabs account.

### 3. Storing API Keys

Store the API keys in an .env file:
1. Create the file:
   config/.env

2. Add the following lines:

   OPENAI_API_KEY="your-openai-key-here"

   ELEVENLABS_API_KEY="your-elevenlabs-key-here"

### 4. Start Required Services

All services below must be running at the same time.

#### 4.1 Install Dependencies
```bash
pip install --upgrade social_interaction_cloud[dialogflow,openai-gpt]
```

#### 4.2 Start Redis
```bash
cd sic_applications
conf/redis/redis-server.exe conf/redis/redis.conf
```

#### 4.3 Start Interaction Services (Optional)
(Each command in a new terminal)
```bash
run-dialogflow
```
```bash
run-gpt
```
---

## B. Calibration 
This is the calibration phase used to extract participant-specific feature distribution and improve robustness. 
1.  Run `python -m sounddevice` to list devices and find the correct index for your microphone
2. Run the [calibration script](interaction/run_calibration.py) by setting the appropriate `participant_id` and `device_index`
3. To quickly test Whisper transcription quality, run:
   - `python interaction/run_whisper_transcriber.py --device-index <MIC_INDEX>`
   - or pass an existing file: `python interaction/run_whisper_transcriber.py --audio-path /path/to/audio.wav`

This will save extracted features to a file per participant under `multimodal_perception/data/calibration_phase`.
These features will then be used for normalization before confidence estimation during the actual experiment.

---
## C. Running the Experiment
1. Configure the interaction settings [in the main script](src/main.py):
    - **participant_id**: unique identifier for the participant 
    - **is_adaptive**: experiment condition 
    - **use_keyboard_input**: set `True` to type spymaster inputs in the terminal (no STT listening)
    - **stt_mic_device_index**: index of the microphone used for speech-to-text (STT) processing 
   - **audio_features_mic_device_index**: index of the microphone used for audio feature extraction (worn by the participant)
   - **board_config_number**: configuration number for the game (see `assets/configs` for details)
   - **robot_ip**: IP address of the robot 
2. Run the system: `python main.py`
3. You can access the Game UI at `http://127.0.0.1:8765/ui`
4. Spymaster and robot utterances are saved in `logs/utterances_<participant_id>_<YYYYMMDD>.txt`

## D. How the game works (brief)

1. A Codenames board configuration is loaded from `assets/configs`, and `GameState` starts a local Flask server for state/UI endpoints.
2. At the beginning, the spymaster places the initial red cards and confirms they are ready.
3. For each turn, the spymaster gives a clue (`word + number`), the robot confirms it, then the robot makes guesses.
4. The game continues turn-by-turn until win/loss conditions are reached (all blue found, assassin hit, or red-card limit reached).

### Controlling game state from the UI

Open `http://127.0.0.1:8765/ui` during a run:

- Select a color (`Red`, `Blue`, `Neutral`, `Assassin`) and click a card to reveal it with that outcome.
- Use `Unreveal` to remove a previously applied outcome from a card.
- The UI syncs with server state (`/state`) and sends updates to `/reveal` and `/unreveal`.
