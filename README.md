# Langchain_Reel_Scripter

## Installation
1. Install python dependencies:  
  ```
  pip install requirements.txt
  ```
2. To setup Ollama
  Refer [Ollama GitHub](https://github.com/ollama/ollama)

## Usage
1. Agent Setup - This is a chatbot which will try to understand about your
audience, platform, reel type focus, etc.
To execute run:
```
python3 Scripts/agent_setup.py
```

## Plan:
- [x] Setup Ollama with Langchain
- [ ] Create an agent to understand detailed requirement of user's content and account
  - [x] This shall incude details like niche, audience, type of videos, format of content, etc
  - [x] This information should be stored for a customised system prompt
- [ ] After creating a proper a system test with various topics and analyse the results
- [ ] Depending on results finetuning agent should be introduced
