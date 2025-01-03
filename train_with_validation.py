import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2


# Import your existing model architecture
from vae_model import DetailVAE, PerceptualLoss, vae_detail_loss

class VAETrainer:
    def __init__(self, base_dir, image_size=128, batch_size=32, latent_dim=128, 
                 learning_rate=3e-4, num_epochs=150):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = Path(base_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Initialize model and loss
        self.model = DetailVAE(latent_dim, image_size).to(self.device)
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training transform with augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Validation/Test transform without augmentation
        self.eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Setup data loaders
        self.setup_data_loaders()

        # Create output directories
        self.setup_directories()

        # Initialize tracking
        self.best_val_score = float('inf')
        self.train_losses = []
        self.val_metrics = []

    def setup_directories(self):
        dirs = ['reconstructions/train', 'reconstructions/validation', 
                'reconstructions/test', 'models', 'metrics']
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def setup_data_loaders(self):
        # Training data (good images)
        train_data = ImageFolder(
            self.base_dir / "preprocessed_train/good/class1",
            transform=self.train_transform
        )
        self.train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        # Validation data (defective images and ground truth)
        self.val_data = {
            'defective_painting': {
                'images': self.load_images(self.base_dir / "preprocessed_test/defective_painting"),
                'ground_truth': self.load_images(self.base_dir / "ground_truth/defective_painting")
            },
            'scratches': {
                'images': self.load_images(self.base_dir / "preprocessed_test/scratches"),
                'ground_truth': self.load_images(self.base_dir / "ground_truth/scratches")
            }
        }

    def load_images(self, path):
        dataset = ImageFolder(path, transform=self.eval_transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, (images, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            recon_images, mu, logvar, detail_mu, detail_logvar = self.model(images)

            # Calculate loss
            loss = vae_detail_loss(recon_images, images, mu, logvar,
                                   detail_mu, detail_logvar, self.perceptual_loss)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Save reconstructions
            if batch_idx == 0:
                comparison = torch.cat([images[:8], recon_images[:8]])
                save_image(comparison.cpu(),
                          f'reconstructions/train/epoch_{epoch}.png',
                          nrow=8, normalize=True)

        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        val_metrics = {}

        with torch.no_grad():
            for defect_type in ['defective_painting', 'scratches']:
                metrics = self.validate_defect_type(
                    self.val_data[defect_type]['images'],
                    self.val_data[defect_type]['ground_truth'],
                    defect_type,
                    epoch
                )
                val_metrics[defect_type] = metrics

        # Calculate average reconstruction error as validation score
        avg_score = np.mean([
            m['reconstruction_error'] 
            for defect_type in val_metrics.values() 
            for m in defect_type['metrics']
        ])

        return val_metrics, avg_score

    def validate_defect_type(self, defect_loader, ground_truth_loader, defect_type, epoch):
        metrics_list = []

        for batch_idx, (defect_batch, gt_batch) in enumerate(zip(defect_loader, ground_truth_loader)):
            # Unpack images
            defect_imgs, _ = defect_batch
            gt_imgs, _ = gt_batch

            # Ensure consistent batch sizes
            min_batch_size = min(len(defect_imgs), len(gt_imgs))
            defect_imgs = defect_imgs[:min_batch_size]
            gt_imgs = gt_imgs[:min_batch_size]

            defect_imgs = defect_imgs.to(self.device)
            gt_imgs = gt_imgs.to(self.device)

            # Get reconstruction
            recon_imgs, _, _, _, _ = self.model(defect_imgs)

            # Calculate raw defect map
            defect_map = torch.abs(recon_imgs - defect_imgs).cpu().detach().numpy()

            # Post-process the defect map
            refined_maps = []
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            for dmap in defect_map:
                dmap = cv2.morphologyEx(dmap, cv2.MORPH_CLOSE, kernel)
                refined_maps.append(dmap)
            refined_defect_map = torch.tensor(np.stack(refined_maps), device=self.device)

            # Apply thresholding
            threshold = 0.1
            binary_defect_map = (refined_defect_map > threshold).float()

            # Create binary masks
            predicted_mask = binary_defect_map
            ground_truth_mask = (gt_imgs > 0.5).float()

            # Calculate metrics
            intersection = (predicted_mask * ground_truth_mask).sum().item()
            union = ((predicted_mask + ground_truth_mask) > 0).sum().item()
            iou = intersection / (union + 1e-8)

            precision = intersection / (predicted_mask.sum().item() + 1e-8)
            recall = intersection / (ground_truth_mask.sum().item() + 1e-8)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

            # Append metrics
            metrics_list.append({
                'reconstruction_error': torch.mean(refined_defect_map).item(),
                'IoU': iou,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score
            })

            # Save defect maps for visualization
            if batch_idx == 0:
                comparison = torch.cat([
                    defect_imgs[:4],         # Original defective images
                    recon_imgs[:4],          # Reconstructions
                    refined_defect_map[:4]   # Refined defect maps
                ])
                save_image(comparison.cpu(),
                        f'reconstructions/validation/{defect_type}_epoch_{epoch}.png',
                        nrow=4, normalize=True)

        return {
            'metrics': metrics_list
        }

    def train(self):
        print(f"Starting training on {self.device}")

        for epoch in range(self.num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validation
            val_metrics, val_score = self.validate(epoch)
            self.val_metrics.append(val_score)

            # Print progress
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Score: {val_score:.4f}")

            # Save best model
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                torch.save(self.model.state_dict(), 'models/best_model.pth')
                print("Saved new best model!")

            # Regular checkpoints
            if (epoch + 1) % 50 == 0:
                torch.save(self.model.state_dict(), f'models/model_epoch_{epoch+1}.pth')
                self.plot_metrics()

        # Save final model
        torch.save(self.model.state_dict(), 'models/final_model.pth')
        self.plot_metrics()
        print("\nTraining completed!")

    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_metrics, label='Validation Score')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('metrics/training_curves.png')
        plt.close()

if __name__ == "__main__":
    trainer = VAETrainer(
        base_dir="data/bracket_white",
        image_size=256,
        batch_size=32,
        latent_dim=128,
        learning_rate=3e-4,
        num_epochs=15
    )
    trainer.train()
