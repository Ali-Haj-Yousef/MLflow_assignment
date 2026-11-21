
import os
import shutil

import dagshub
import mlflow
import mlflow.sklearn

# Authenticate using the GitHub Secret
dagshub.auth.add_app_token(os.environ["DAGSHUB_TOKEN"])

# Connect to the DagsHub repository and configure MLflow
dagshub.init(
    repo_owner="Ali-Haj-Yousef",
    repo_name="my-first-repo",
    mlflow=True
)

# Select the experiment containing the models we want to deploy
EXPERIMENT_NAME = "RandomForest_V1"
ARTIFACT_PATH = "model"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError(
        f"Experiment '{EXPERIMENT_NAME}' was not found."
    )

# Retrieve completed runs, newest first
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["start_time DESC"],
    max_results=1000
)

if runs.empty:
    raise RuntimeError("No completed training runs found.")

# Try the newest completed runs until we find one with a model
model = None
selected_run_id = None

for _, run in runs.iterrows():
    run_id = run["run_id"]
    model_uri = f"runs:/{run_id}/{ARTIFACT_PATH}"

    try:
        print(f"Trying model from run: {run_id}")
        model = mlflow.sklearn.load_model(model_uri)
        selected_run_id = run_id
        break
    except Exception as error:
        print(f"No usable model in run {run_id}: {error}")

if model is None:
    raise RuntimeError(
        "No usable model was found in the experiment."
    )

# Remove any previous model directory
if os.path.exists("model"):
    shutil.rmtree("model")

# Save the selected model in the GitHub Actions workspace
mlflow.sklearn.save_model(model, path="model")

print(f"Successfully retrieved model from run: {selected_run_id}")
print("Model saved in ./model")
