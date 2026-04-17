# 🤖 Beyond Speech HRI

A research project exploring how a social robot can leverage non-verbal cues to dynamically adapt its behavior during collaborative human–robot interaction (HRI).

---

## Overview

This project investigates how a social robot can move beyond spoken language by using non-verbal signals (timing, prosody, turn-taking, contextual awareness) to adapt its behavior during collaborative tasks with humans.

The system integrates conversational AI, speech synthesis, and real-time interaction services to support adaptive, socially aware robot behavior.

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
3. After each recording, enter the participant self-report confidence (`low`, `medium`, or `high`) when prompted. It is stored in the calibration CSV as `self_report` for post-hoc calibration during inference.

This will save extracted features to a file per participant under `multimodal_perception/data/calibrartion_phase`.
These features will then be used for normalization before confidence estimation during the actual experiment.

---
## C. Running the Experiment
1. Configure the interaction settings [in the main script](src/main.py):
   - **participant_id**: unique identifier for the participant 
   - **is_adaptive**: experiment condition 
   - **stt_mic_device_index**: index of the microphone used for speech-to-text (STT) processing 
   - **audio_features_mic_device_index**: index of the microphone used for audio feature extraction (worn by the participant)
   - **board_config_number**: configuration number for the game (see `assets/configs` for details)
   - **robot_ip**: IP address of the robot 
2. Run the system: `python main.py`
3. You can access the Game UI at `http://127.0.0.1:8765/ui`
4. Spymaster and robot utterances are saved in `logs/utterances_<participant_id>_<YYYYMMDD>.txt`
