import torch
import torch.nn as nn
from encoder import ImageEncoder


# class PointCloudDecoder(nn.Module):
#     def __init__(self, latent_dim=256, num_points=2048):
#         super().__init__()
#         self.num_points = num_points

#         self.fc = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 1024),
#             nn.ReLU(inplace=True),
#         )

#         self.point_gen = nn.Sequential(
#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, num_points * 3),
#         )

#     def forward(self, z):
#         """
#         z: tensor of shape [B, latent_dim]
#         returns: tensor of shape [B, num_points, 3]
#         """
#         x = self.fc(z)                  # shape [B, 1024]
#         pts = self.point_gen(x)         # shape [B, num_points * 3]
#         pts = pts.view(-1, self.num_points, 3)  # shape [B, N, 3]
#         return pts

class PointCloudDecoderShared(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=1024, num_points=2048):
        super().__init__()
        self.num_points = num_points

        # global feature
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True)
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
        global_feat = global_feat.unsqueeze(1)         # (B, 1, hidden_dim)
        global_feat = global_feat.expand(-1, self.num_points, -1)

        pts = self.point_head(global_feat)             # (B, N, 3)
        return pts

class Img2PointCloudModel(nn.Module):
    def __init__(self, latent_dim=256, num_points=2048, pretrained_encoder=False):
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