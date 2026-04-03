import torch


class ConcatLeftAlign:
    def __init__(self):
        pass

    def set_shape_meta(self, shape_meta):
        self.shape_meta = shape_meta

    def forward(self, data):
        if "action" in data:
            action_parts = []
            for meta in self.shape_meta.get("action", []):
                key = meta["key"]
                action_parts.append(data["action"][key])
            data["action"] = torch.cat(action_parts, dim=-1)

        if "state" in data:
            state_parts = []
            for meta in self.shape_meta.get("state", []):
                key = meta["key"]
                state_parts.append(data["state"][key])
            data["state"] = torch.cat(state_parts, dim=-1)

        return data
