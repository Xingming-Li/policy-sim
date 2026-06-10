import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:

    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv(
                "OPENAI_API_KEY"
            )
        )

    def generate(
        self,
        prompt
    ):

        response = (
            self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={
                    "type": "json_object"
                },
                temperature=0.7
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
        )