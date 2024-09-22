import mlflow
from typing import Literal
from langchain_core.tools import tool
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from rich import print
import warnings
import uuid
from databricks.sdk import WorkspaceClient
import time
mlflow.login()


warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

experiment_name = "/Users/sathish.gangichetty@databricks.com/langgraph-demo"
experiment = mlflow.set_experiment(experiment_name)

w = WorkspaceClient()
ENDPOINT_NAME = "sg-gpt4o"

mlflow.langchain.autolog()
@tool
def get_weather(city: Literal["atlanta", "sf"]):
    """Use this to get weather information."""
    if city == "atlanta":
        return "It's pretty hot and there could be thunderstorms. Hotlanta for a reason!"
    elif city == "sf":
        return "It's ok in sf. Not too hot, not too cold."


chat_model_external = ChatOpenAI(
    model=ENDPOINT_NAME,
    temperature=0.7,
    max_tokens=512,
    api_key=w.tokens.create(
        comment=f"sdk-{time.time_ns()}", lifetime_seconds=120
    ).token_value,
    base_url=f"{w.config.host}/serving-endpoints"
    # See https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.databricks.ChatDatabricks.html for other supported parameters
)
tools = [get_weather]
graph = create_react_agent(chat_model_external, tools)

query = {
    "messages": [
        {
            "role": "user",
            "content": "Should I bring an umbrella today when I go to work in Atlanta?",
        }
    ]
}

# # @mlflow.trace(name=f"langgraph-demo+{uuid.uuid4()}")
# def run_langgraph(query):
#     response = graph.invoke(query)
#     return response

# response = run_langgraph(query)
# print(response['messages'])
# print("**** The final AI response is ****")
# print(response['messages'][-1].content)

# specify the Agent as the model interface to be loaded when executing the script
mlflow.models.set_model(graph)