import os

import dspy
from dotenv import load_dotenv

import datasets
import openai
from openai.types.chat import ChatCompletion

root_path = "."
load_dotenv()
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(root_path, "cache")
openai_key = os.getenv("OPENAI_API_KEY")

colbert_server = "http://index.contextual.ai:8893/api/search"


def openai_native(question: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=openai_key)

    response:ChatCompletion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tell me a story."},

                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content



def main():
    lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=openai_key)
    rm = dspy.ColBERTv2(url=colbert_server)
    dspy.settings.configure(lm=lm, rm=rm)

    # print(openai_native("Which award did Gary Zukav's first book receive?"))

    print(lm("Which award did Gary Zukav's first book receive?"))


if __name__ == "__main__":
    main()

