import torch
import torch.nn as nn
from encoder import ImageEncoder

class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=1024, num_points=2048):
        super().__init__()
        self.num_points = num_points

        # global feature
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, num_points * 3)
        )

        # per-point head
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 3)
        )

    def forward(self, z):
        B = z.shape[0]
        global_feat = self.fc(z)                       # (B, hidden_dim)
        pts = global_feat.view(B, self.num_points, 3) # (B, N, 3)
        return pts

class Img2PointCloudModel(nn.Module):
    def __init__(self, latent_dim=512, num_points=2048, pretrained_encoder=False):
        super().__init__()
        self.encoder =  self.encoder = ImageEncoder(
            pretrained=pretrained_encoder,
            latent_dim=latent_dim
        )
        self.decoder = PointCloudDecoder(latent_dim=latent_dim, num_points=num_points)

    def forward(self, images):
        """
        images: tensor of shape [B, 3, H, W] or [B, V, 3, H, W]
        returns: tensor of shape [B, num_points, 3]
        """
        z = self.encoder(images)        # shape [B, latent_dim]
        point_cloud = self.decoder(z)   # shape [B, num_points, 3]
        return point_cloud

import torch
import torch.nn as nn

class ChamferDistance(nn.Module):
    """Compute Chamfer Distance between two point clouds.
       Input shapes:
         pred: [B, Np, 3]
         gt:   [B, Ng, 3]
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        # pred: [B, Np, 3]
        # gt:   [B, Ng, 3]
        B, Np, _ = pred.shape
        Ng = gt.shape[1]

        # Expand to compute pairwise distances:
        # pred_expand: [B, Np, 1, 3]
        # gt_expand:   [B, 1, Ng, 3]
        pred_expand = pred.unsqueeze(2)     # [B, Np, 1, 3]
        gt_expand = gt.unsqueeze(1)         # [B, 1, Ng, 3]

        # Pairwise squared distances: [B, Np, Ng]
        dist = torch.sum((pred_expand - gt_expand)**2, dim=3)

        # Pred → GT
        min_pred_to_gt, _ = torch.min(dist, dim=2)  # [B, Np]

        # GT → Pred
        min_gt_to_pred, _ = torch.min(dist, dim=1)  # [B, Ng]

        chamfer = min_pred_to_gt.mean(dim=1) + min_gt_to_pred.mean(dim=1)

        return chamfer.mean()
