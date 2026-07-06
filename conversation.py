import json

import ollama

from prompts import SYSTEM_PROMPT

MODEL_NAME = "llama3.1:8b"


class HealthConversation:

    def __init__(self, patient_context=None):

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        if patient_context:
            self.messages.append(
                {
                    "role": "system",
                    "content": (
                        "Already collected patient context from Python: "
                        f"{json.dumps(patient_context, ensure_ascii=False)}. "
                        "Do not ask about these fields again. Focus only on symptoms."
                    )
                }
            )

    def chat(self, user_input):

        if user_input:
            self.messages.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

        response = ollama.chat(
            model=MODEL_NAME,
            messages=self.messages
        )

        assistant_reply = response["message"]["content"]

        self.messages.append(
            {
                "role": "assistant",
                "content": assistant_reply
            }
        )

        return assistant_reply