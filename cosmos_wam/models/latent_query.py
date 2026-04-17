import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_wam.utils.projection import project_world_to_pixel, generate_gaussian_heatmap


class LatentQueryEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=2048,
        num_queries=32,
        sigma=8.0,
        camera_params_path=None,
        image_height=224,
        image_width=224,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.sigma = sigma
        self.image_height = image_height
        self.image_width = image_width

        # Patch embed: [1, 224, 224] -> [2048, 16, 16]
        self.patch_embed = nn.Conv2d(1, hidden_dim, kernel_size=14, stride=14)
        self.patch_pos_embed = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)

        # Perceiver Resampler: 256 -> 32
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Temporal position encoding
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        # Load camera params
        if camera_params_path is not None:
            with open(camera_params_path, "r") as f:
                params = json.load(f)
            self.register_buffer(
                "_intrinsic",
                torch.tensor(params["intrinsic"], dtype=torch.float32),
            )
            self.register_buffer(
                "_extrinsic",
                torch.tensor(params["extrinsic"], dtype=torch.float32),
            )
            self._render_h = params.get("image_height", 256)
            self._render_w = params.get("image_width", 256)
        else:
            self.register_buffer("_intrinsic", torch.zeros(3, 3))
            self.register_buffer("_extrinsic", torch.zeros(4, 4))
            self._render_h = 256
            self._render_w = 256

    def forward(self, eef_pos_3d):
        """
        eef_pos_3d: [B, 3]
        Returns: [B, num_queries, hidden_dim]
        """
        B = eef_pos_3d.shape[0]
        device = eef_pos_3d.device

        # Project to 2D
        pixel_uv = project_world_to_pixel(
            eef_pos_3d, self._intrinsic, self._extrinsic, self._render_h, self._render_w
        )

        # Scale to heatmap resolution
        scale_x = self.image_width / self._render_w
        scale_y = self.image_height / self._render_h
        heatmap_u = pixel_uv[:, 0] * scale_x
        heatmap_v = pixel_uv[:, 1] * scale_y

        # Generate Gaussian heatmap
        heatmap = generate_gaussian_heatmap(
            heatmap_u, heatmap_v, self.image_height, self.image_width, self.sigma
        )
        heatmap = heatmap.to(device=device, dtype=next(self.parameters()).dtype)

        # Patch embed
        patch_features = self.patch_embed(heatmap)
        patch_features = patch_features.flatten(2).permute(0, 2, 1)  # [B, 256, D]
        patch_features = patch_features + self.patch_pos_embed

        # Perceiver: 256 -> 32
        queries = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, patch_features, patch_features)
        latent_query_init = self.norm(queries + attn_out)

        # Temporal position encoding
        latent_query_init = latent_query_init + self.temporal_pos_embed

        return latent_query_init


class TrajectoryHead(nn.Module):
    # 固定 4 个点：原点 + X/Y/Z 轴端点
    NUM_POINTS = 4

    def __init__(self, hidden_dim=2048, heatmap_size=16):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.NUM_POINTS * heatmap_size * heatmap_size),
        )

    @staticmethod
    def soft_argmax(heatmap):
        """heatmap: [..., H, W] -> coords: [..., 2] (x, y)"""
        *leading, H, W = heatmap.shape
        device = heatmap.device
        dtype = heatmap.dtype

        y_grid = torch.arange(H, device=device, dtype=dtype).view(*([1] * len(leading)), H, 1)
        x_grid = torch.arange(W, device=device, dtype=dtype).view(*([1] * len(leading)), 1, W)

        flat = heatmap.view(*leading, -1)
        prob = F.softmax(flat, dim=-1).view(*leading, H, W)
        pred_y = (prob * y_grid).sum(dim=(-2, -1))
        pred_x = (prob * x_grid).sum(dim=(-2, -1))
        return torch.stack([pred_x, pred_y], dim=-1)

    def forward(self, latent_query_hidden, target_heatmap):
        """
        latent_query_hidden: [B, 32, hidden_dim]
        target_heatmap: [B, 32, 4, 16, 16]
        Returns: loss scalar
        """
        B, T, D = latent_query_hidden.shape
        pred = self.head(latent_query_hidden)  # [B, 32, 4 * H * W]
        pred = pred.view(B, T, self.NUM_POINTS, self.heatmap_size, self.heatmap_size)
        pred = torch.sigmoid(pred)

        # Soft-argmax to coordinates: only supervise keypoint locations, not Gaussian shape.
        pred_coords = self.soft_argmax(pred)
        target_coords = self.soft_argmax(target_heatmap)
        loss = F.mse_loss(pred_coords, target_coords)
        return loss
