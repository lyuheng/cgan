import torch
import torch.nn as nn
import torch.nn.functional as F

class discriminator(nn.Module):
    def __init__(self, channel = 3):
        # Avoid Sparse Gradients: ReLU, MaxPool
        super().__init__()
        #self.use_WGAN = use_WGAN
        self.channel = channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channel, 64, kernel_size=5, stride=2, padding=2), # (64-5+4)/2 + 1 = 32
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # (32-5+4)/2 + 1 = 16
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # (16-5+4)/2 + 1 = 8
            nn.LeakyReLU(),
            nn.Conv2d(256, 384, kernel_size=5, stride=2, padding=2),  # (8-5+4)/2 + 1 = 4
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(384+23, 384, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )
        self.linear = nn.Linear(384*4*4, 1)
    def forward(self, x, y):
        # x: picture(64, 64, 3)  y:label(conditional vector) (b, 23)
        batch_size = x.shape[0]
        out = self.conv1(x) # (b,384,4,4)
        embed_y = torch.unsqueeze(y, 2) # embed_y:(b,23,1)
        embed_y = torch.unsqueeze(embed_y, 3) # (b,23,1,1)
        repeated_embedings = embed_y.repeat(1, 1, 4, 4) # (b,23, 4,4)
        h3_concat = torch.cat([out, repeated_embedings], dim=1) # (b,384+23,4,4)
        out = self.conv2(h3_concat)
        out = out.reshape(batch_size, -1) # (b, 384*4*4)
        out = self.linear(out) # (b,)
        return torch.sigmoid(out)

class generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim  # noise dimension
        self.net1 = nn.Sequential(
            nn.Linear(23+z_dim, 4*4*384),
            nn.BatchNorm1d(4*4*384, eps=1e-05, momentum=0.9)
        )
        self.conv_block = nn.Sequential(
            nn.Upsample(scale_factor=2),  # (8,8)
            nn.Conv2d(384, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.9),

            nn.Upsample(scale_factor=2), # (16,16)
            nn.Conv2d(256,128, kernel_size=5, stride=1, padding=2), # (16-5+4) +1 = 16
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.9),

            nn.Upsample(scale_factor=2),  # (32,32)
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),  # (32-5+4) +1 = 32
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.9),

            nn.Upsample(scale_factor=2),  # (64,64)
            nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2),  # (64-5+4)+1=64
        )

    def forward(self, z, y):
        # z : noise vector(b, z_dim) y:label(conditional vector) (b, 23)
        x = torch.cat([z, y], dim=-1)  # (b, z_dim+23)
        batch_size = x.shape[0]
        x_dim = x.shape[1] # z_dim + 23
        out = self.net1(x) # (b, 4*4*384)
        out = out.reshape(batch_size, 384,4,4)
        out = self.conv_block(out)  # (b, 3,64,64)
        out = torch.tanh(out)  # (b, 3,64,64)
        return out

