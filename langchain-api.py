from langchain_databricks import ChatDatabricks
from rich import print

chat_model_external = ChatDatabricks(
    endpoint="sg-models",
    temperature=0.7,
    max_tokens=512,
    # See https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.databricks.ChatDatabricks.html for other supported parameters
)
response = chat_model_external.invoke(
    "What are the top 10 reasons to build AI applications on Databricks?"
)
print(response.content)
print(response.response_metadata)
