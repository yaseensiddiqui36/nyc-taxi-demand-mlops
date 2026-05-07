import os, sys
sys.path.insert(0, ".")
os.environ["MLFLOW_TRACKING_URI"]      = "https://dagshub.com/yaseensiddiqui36/nyc-taxi-demand-mlops.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "yaseensiddiqui36"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "42980097cf8a1a8a00d853c961352e8592d24089"

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
filter_str = "name='taxi_demand_predictor'"
versions = client.search_model_versions(filter_str)
if not versions:
    print("No registered model versions found.")
    sys.exit(1)

for v in versions:
    arts = client.list_artifacts(v.run_id, "model")
    has_mlmodel = any("MLmodel" in a.path for a in arts)
    print(f"v{v.version}  stage={v.current_stage}  run={v.run_id[:8]}  "
          f"artifacts={[a.path for a in arts]}  has_MLmodel={has_mlmodel}")
