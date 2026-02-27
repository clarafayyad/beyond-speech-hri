# 🤖 Beyond Speech HRI

A research project exploring how a social robot can leverage non-verbal cues to dynamically adapt its behavior during collaborative human–robot interaction (HRI).

---

## Overview

This project investigates how a social robot can move beyond spoken language by using non-verbal signals (timing, prosody, turn-taking, contextual awareness) to adapt its behavior during collaborative tasks with humans.

The system integrates conversational AI, speech synthesis, and real-time interaction services to support adaptive, socially aware robot behavior.

---
## Setup Guide

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

### 5. Robot Configuration and Execution

1. Open main.py and configure the robot IP address and any robot-specific parameters.

2. Run the system:
```bash
python main.py
```

---
