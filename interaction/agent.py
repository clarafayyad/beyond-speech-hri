import time


class Agent:
    def say(self, text: str):
        # TODO: implement
        print(f"Agent says ${text}")

    def listen(self) -> str:
        # TODO: implement
        print(f"Agent is listening...")
        time.sleep(2)

    def talk_llm(self, system_prompt: str, user_prompt: str) -> dict:
        # TODO: implement
        print("talking llm?")