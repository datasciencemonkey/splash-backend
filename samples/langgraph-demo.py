#%%
import mlflow
from rich import print
from databricks import agents
import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
mlflow.login()


experiment_name = "/Users/sathish.gangichetty@databricks.com/langgraph-demo"
experiment = mlflow.set_experiment(experiment_name)

input_example = {
    "messages": [{"role": "user", "content": "what is the weather in seattle today?"}]
}

with mlflow.start_run():
    model_info = mlflow.langchain.log_model(
        lc_model="./mlflow-app.py",  # specify the path to the LangGraph agent script definition
        artifact_path="langgraph",
        input_example=input_example,
    )

agent = mlflow.langchain.load_model(model_info.model_uri)
agent.invoke(input_example)

# %%
# Use Unity Catalog to log the chain
mlflow.set_registry_uri('databricks-uc')

# Register the chain to UC
UC_MODEL_NAME = "sg-langgraph-demo"
uc_registered_model_info = mlflow.register_model(model_uri=model_info.model_uri, name=UC_MODEL_NAME)

# Create a new endpoint
endpoint_name = f"langgraph-demo-{time.time_ns()}"
endpoint_config = EndpointStateConfigUpdate(
    name=endpoint_name,
    model_name=uc_registered_model_info.name,
    model_version=uc_registered_model_info.version,
)
# %%
