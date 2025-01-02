import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 15
image_size = 128
batch_size = 64
learning_rate = 1e-4
num_epochs = 300

# VAE Definition
class VAE(nn.Module):
    def __init__(self, latent_dim, image_size=128):
        super(VAE, self).__init__()
        self.image_size = image_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Flatten()
        )
        flattened_size = (image_size // 4) * (image_size // 4) * 128
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64 -> 128
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        x = self.fc_decode(z)
        x = x.view(batch_size, 128, self.image_size // 4, self.image_size // 4)
        x = self.decoder(x)
        return x, mu, logvar

# Loss Function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)  # Reconstruction Loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  # KL Divergence
    return recon_loss + kl_loss

# Data Preparation
data_dir = "data/bracket_white/preprocessed_train/good"
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model and Optimizer
model = VAE(latent_dim, image_size=image_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
print("Training the VAE...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, _ in dataloader:
        images = images.to(device)
        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        loss = vae_loss(recon_images, images, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save Sample Reconstructions
    if (epoch + 1) % 10 == 0:
        os.makedirs("reconstructions", exist_ok=True)
        save_image(recon_images[:16], f"reconstructions/recon_epoch_{epoch + 1}.png")

# Save the Model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/vae.pth")
print("Model training complete and saved!")
