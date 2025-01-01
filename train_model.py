import os
import torch
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 16
image_size = 128
batch_size = 32
learning_rate = 1e-3
num_epochs = 50

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * (image_size // 4) * (image_size // 4), latent_dim)
        self.fc_logvar = nn.Linear(64 * (image_size // 4) * (image_size // 4), latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 64 * (image_size // 4) * (image_size // 4))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64 -> 128
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.fc_decode(z)
        x = x.view(-1, 64, image_size // 4, image_size // 4)
        x = self.decoder(x)
        return x, mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = nn.MSELoss()(recon_x, x)
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss / x.size(0)

# Data preparation
data_dir = "data/bracket_white/preprocessed_train/good"
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, optimizer, and training loop
model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training the VAE...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, _ in dataloader:  # Images only; ignore labels
        images = images.to(device)
        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        loss = vae_loss(recon_images, images, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save sample reconstructions
    if (epoch + 1) % 10 == 0:
        save_image(recon_images[:16], f"recon_epoch_{epoch + 1}.png")

# Save the model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/vae.pth")
print("Model training complete and saved!")
