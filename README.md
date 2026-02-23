# beyond-speech-hri

A research project exploring how a social robot can use **non-verbal cues** to adapt its behavior during a **collaborative human–robot interaction**.

---

## Setup

This project relies on **Dialogflow**, **Google Text-to-Speech**, **OpenAI**, and **Redis**.  
Please follow the steps below carefully.

---

## 1. Google Cloud setup (Dialogflow + TTS)

### 1.1 Dialogflow

1. Set up a Google Cloud project and Dialogflow account by following:  
   https://socialrobotics.atlassian.net/wiki/spaces/CBSR/pages/2205155343/Getting+a+google+dialogflow+key

2. Create a Dialogflow service account key file and save it as: conf/dialogflow/google-key.json


> ⚠️ **Important:** Never share or commit this key file.

---

### 1.2 Google Text-to-Speech (TTS)

3. Enable the Google Text-to-Speech API:  
https://console.cloud.google.com/apis/api/texttospeech.googleapis.com/

4. You will need to enable billing and add a credit card.
- Google provides **$300 in free credits**, which is more than enough for development and testing.
- In practice, this setup should not cost anything.

---

## 2. Dialogflow agent configuration

5. Create a **new, empty Dialogflow agent**.

6. Remove **all default intents**.

7. Go to: Settings → Import and Export\

8. Import the following file into your Dialogflow agent: resources/droomrobot_dialogflow_agent.zip


This will add all required intents and entities used in this project (and additional examples).

---

## 3. OpenAI API key

9. Generate a personal OpenAI API key:  
https://platform.openai.com/api-keys

10. Store your key using **one** of the following methods:

### Option A — Environment variable (recommended)

Set the variable in your system environment:
```bash
OPENAI_API_KEY="your-key-here"
```

### Option B — `.env` file
Create the following file: confing/openai/.env

Add OPENAI_API_KEY="your-key-here"

---

## 4. Start required services
The following services must be running at the same time:

11. Install dependencies:
```bash
pip install --upgrade social_interaction_cloud[dialogflow,google-tts,openai-gpt]
```
12. Start Redis:
```bash
conf/redis/redis-server.exe conf/redis/redis.conf
```
13. In a **new terminal**, start Dialogflow:
```bash
run-dialogflow
```
14. In a **new terminal**, start Google TTS:
```bash
run-google-tts
```
15. In a **new terminal**, start the OpenAI GPT service:
```bash
run-gpt
```

---

## 5. Robot Configuration & Execution
16. In `main.py`, configure the robot IP address
17. Run the system:
```bash
python main.py
```