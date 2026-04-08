# Cosmos-WAM Policy for RoboTwin Evaluation
import sys
print(f"[COSMOS_WAM] Loading cosmos_wam_policy module from: {__file__}", file=sys.stderr)
print(f"[COSMOS_WAM] Python version: {sys.version}", file=sys.stderr)

from .deploy_policy import CosmosWAMRobotWinPolicy, get_model, eval, reset_model, encode_obs

__all__ = ["CosmosWAMRobotWinPolicy", "get_model", "eval", "reset_model", "encode_obs"]
print(f"[COSMOS_WAM] Module loaded successfully", file=sys.stderr)
