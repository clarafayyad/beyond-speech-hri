# 🤖 Beyond Speech HRI

A research project exploring how a social robot can leverage non-verbal cues to dynamically adapt its behavior during collaborative human–robot interaction (HRI).

---

## Overview

This project investigates how a social robot can move beyond spoken language by using non-verbal signals (timing, prosody, turn-taking, contextual awareness) to adapt its behavior during collaborative tasks with humans.

The system integrates conversational AI, speech synthesis, and real-time interaction services to support adaptive, socially aware robot behavior.

---

## Technology Stack

This project relies on the following external services:

- Dialogflow — intent recognition and dialogue management
- ElevenLabs Text-to-Speech — speech synthesis
- OpenAI — language modeling and reasoning
- Redis — message brokering and state management

---
## Setup Guide

Follow the steps below in order to ensure a correct setup.

---

## 1. Google Cloud Setup (Dialogflow)

### 1.1 Dialogflow

1. Create a Google Cloud project and Dialogflow account by following:
   https://socialrobotics.atlassian.net/wiki/spaces/CBSR/pages/2205155343/Getting+a+google+dialogflow+key

2. Create a Dialogflow service account key and save it as:
   conf/dialogflow/google-key.json

IMPORTANT:
Never share, upload, or commit this key file to version control.

---

## 2. Dialogflow Agent Configuration

3. Create a new, empty Dialogflow agent.

4. Remove all default intents.

5. Navigate to:
   Settings → Import and Export

6. Import the following file:
   resources/droomrobot_dialogflow_agent.zip

This will add all required intents, entities, and example dialogues used in this project.

---

## 3. API Key Setup

### 3.1 OpenAI API Key

7. Generate a personal OpenAI API key:
   https://platform.openai.com/api-keys

### 3.2 ElevenLabs API Key

8. Generate an ElevenLabs API key from your ElevenLabs account.

---

### 3.3 Storing API Keys

Store the API keys using one of the following methods.

#### Option A — Environment Variables (Recommended)

Set the variables in your system environment:
- OPENAI_API_KEY="your-openai-key-here" 
- ELEVENLABS_API_KEY="your-elevenlabs-key-here"

#### Option B — .env File

1. Create the file:
   config/.env

2. Add the following lines:
   OPENAI_API_KEY="your-openai-key-here"
   ELEVENLABS_API_KEY="your-elevenlabs-key-here"

---

## 4. Start Required Services

All services below must be running at the same time.

### 4.1 Install Dependencies
```bash
pip install --upgrade social_interaction_cloud[dialogflow,openai-gpt]
```

### 4.2 Start Redis
```bash
conf/redis/redis-server.exe conf/redis/redis.conf
```

### 4.3 Start Interaction Services
(Each command in a new terminal)
```bash
run-dialogflow
```
```bash
run-gpt
```

---

## 5. Robot Configuration and Execution

15. Open main.py and configure the robot IP address and any robot-specific parameters.

16. Run the system:
```bash
python main.py
```

---
