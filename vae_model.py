import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = self.bn2(self.conv2(x))
        return F.leaky_relu(x + residual, 0.2)

class DetailVAE(nn.Module):
    def __init__(self, latent_dim, image_size=128):
        super(DetailVAE, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim

        # Encoder with residual blocks and skip connections
        self.encoder_layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16 -> 8
        ])
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64),
            ResidualBlock(128),
            ResidualBlock(256),
            ResidualBlock(512)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(512)
        ])

        flattened_size = (image_size // 16) * (image_size // 16) * 512

        # Multi-scale latent space
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)
        
        # Additional detail-oriented latent space
        self.detail_fc_mu = nn.Linear(flattened_size, latent_dim // 2)
        self.detail_fc_logvar = nn.Linear(flattened_size, latent_dim // 2)

        # Decoder with upsampling and skip connections
        self.fc_decode = nn.Linear(latent_dim + latent_dim // 2, flattened_size)

        self.decoder_layers = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),  # 64 -> 128
        ])

        self.decoder_bns = nn.ModuleList([
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(64)
        ])

        # Detail enhancement layers
        self.detail_enhancement = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        ])

    def encode(self, x):
        # Store skip connections
        skip_connections = []
        
        # Encode with residual blocks
        for layer, residual, bn in zip(self.encoder_layers, self.residual_blocks, self.bns):
            x = F.leaky_relu(bn(layer(x)), 0.2)
            x = residual(x)
            skip_connections.append(x)

        x = x.view(x.size(0), -1)
        
        # Generate two sets of latent variables
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        detail_mu = self.detail_fc_mu(x)
        detail_logvar = self.detail_fc_logvar(x)
        
        return mu, logvar, detail_mu, detail_logvar, skip_connections

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skip_connections):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 512, self.image_size // 16, self.image_size // 16)
        
        # First deconvolution
        x = F.relu(self.decoder_bns[0](self.decoder_layers[0](x)))  # 512 -> 256
        x = self.detail_enhancement[0](x) + x
        x = torch.cat([x, skip_connections[2]], dim=1)  # 256 -> 512
        
        # Second deconvolution
        x = F.relu(self.decoder_bns[1](self.decoder_layers[1](x)))  # 512 -> 128
        x = self.detail_enhancement[1](x) + x
        x = torch.cat([x, skip_connections[1]], dim=1)  # 128 -> 256
        
        # Third deconvolution
        x = F.relu(self.decoder_bns[2](self.decoder_layers[2](x)))  # 256 -> 64
        x = self.detail_enhancement[2](x) + x
        x = torch.cat([x, skip_connections[0]], dim=1)  # 64 -> 128
        
        # Final layer
        x = torch.tanh(self.decoder_layers[-1](x))  # 128 -> 3
        return x

    def forward(self, x):
        # Encode
        mu, logvar, detail_mu, detail_logvar, skip_connections = self.encode(x)
        
        # Sample from both latent spaces
        z = self.reparameterize(mu, logvar)
        detail_z = self.reparameterize(detail_mu, detail_logvar)
        
        # Combine latent vectors
        combined_z = torch.cat([z, detail_z], dim=1)
        
        # Decode
        x_recon = self.decode(combined_z, skip_connections)
        
        return x_recon, mu, logvar, detail_mu, detail_logvar

# Enhanced loss function with perceptual loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:23]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        
    def forward(self, x, y):
        vgg_x = self.vgg(x)
        vgg_y = self.vgg(y)
        return F.mse_loss(vgg_x, vgg_y)

def vae_detail_loss(recon_x, x, mu, logvar, detail_mu, detail_logvar, perceptual_loss, kld_weight=0.005):
    # Reconstruction loss (MSE + L1 + Perceptual)
    mse_loss = F.mse_loss(recon_x, x, reduction='sum')
    l1_loss = F.l1_loss(recon_x, x, reduction='sum')
    perc_loss = perceptual_loss(recon_x, x)
    
    recon_loss = 0.5 * (mse_loss + l1_loss) + 0.1 * perc_loss
    
    # KL Divergence for both latent spaces
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    detail_kld_loss = -0.5 * torch.sum(1 + detail_logvar - detail_mu.pow(2) - detail_logvar.exp())
    
    return (recon_loss / x.size(0) + 
            kld_weight * (kld_loss + detail_kld_loss) / x.size(0))

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
image_size = 256
batch_size = 32
learning_rate = 3e-4
num_epochs = 150

# Data preparation with enhanced augmentation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def train_detail_vae(data_dir):
    # Create model and optimizer
    model = DetailVAE(latent_dim, image_size=image_size).to(device)
    perceptual_loss = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load dataset
    dataset = ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Create directories for saving results
    os.makedirs("reconstructions", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Training loop
    print("Training the DetailVAE...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_images, mu, logvar, detail_mu, detail_logvar = model(images)
            
            # Calculate loss
            loss = vae_detail_loss(recon_images, images, mu, logvar, 
                                 detail_mu, detail_logvar, perceptual_loss)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Save sample reconstructions
            if batch_idx == 0 and epoch % 10 == 0:
                comparison = torch.cat([images[:8], recon_images[:8]])
                save_image(comparison.cpu(),
                          f'reconstructions/reconstruction_epoch_{epoch}.png',
                          nrow=8, normalize=True)
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")
        
        # Save model periodically
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'models/detail_vae_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'models/final_detail_vae.pth')
    print("Training completed!")
    
    # Generate random samples
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, latent_dim + latent_dim // 2).to(device)
        samples = model.decode(z, [torch.randn_like(sc) for sc in model.encode(images[:1])[4]])
        save_image(samples.cpu(),
                  'reconstructions/random_samples.png',
                  nrow=8, normalize=True)

if __name__ == "__main__":
    data_dir = "data/bracket_white/preprocessed_train/good"
    train_detail_vae(data_dir)