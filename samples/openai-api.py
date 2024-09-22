from openai import OpenAI
import time
from databricks.sdk import WorkspaceClient
from rich import print

w = WorkspaceClient()
ENDPOINT_NAME = "sg-models"


client = OpenAI(
    api_key=w.tokens.create(
        comment=f"sdk-{time.time_ns()}", lifetime_seconds=120
    ).token_value,
    base_url=f"{w.config.host}/serving-endpoints",
)

prompt = "I need help with our quantum cryptography project, SuperSecretProject."

try:
    response = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
    )
    print(response)
except Exception as e:
    if "invalid_keywords': True" in str(e):
        print(
            "Error: Invalid keywords detected in the prompt. Please revise your input."
        )
    else:
        print(f"An error occurred: {e}")