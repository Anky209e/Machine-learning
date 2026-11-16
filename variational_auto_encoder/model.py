
import torch
import torch.nn.functional as F
from torch import nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self,hidden_dim, latent_dim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Flatten(),
            nn.Linear(7*7*32, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        self.latent_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 7*7*32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x) # Note: self.encoder already ends with nn.ReLU()
        mean = self.mu(h)
        log_var = self.log_var(h)
        std_dev = torch.exp(0.5 * log_var)
        return mean, std_dev
    
    def decode(self, z):
        h = F.relu(self.latent_hidden(z))
        h = self.decoder(h)
        return h
    
    def forward(self, x):
        mean, std_dev = self.encode(x)
        epsilon = torch.randn_like(std_dev) #rand_like to randn_like
        z_reparametrize = mean + epsilon * std_dev
        x_reconstructed = self.decode(z_reparametrize)
        return x_reconstructed, mean, std_dev

if __name__ == "__main__":
    x = torch.randn(4, 1, 28, 28) # Fixed 4D shape
    vae = VariationalAutoEncoder(input_dim=28*28, hidden_dim=16, latent_dim=16)
    x_recon, mean, std = vae(x)
    print(x_recon.shape)
    print(mean.shape)
    print(std.shape)
