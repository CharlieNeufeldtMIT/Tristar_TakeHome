import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, roc_curve, auc
import seaborn as sns

# Import your model architecture
from vae_model import DetailVAE, PerceptualLoss

class VAETester:
    def __init__(self, base_dir, model_path, image_size=256, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = Path(base_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Load model
        self.model = DetailVAE(latent_dim=128, image_size=image_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Transform for test images
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Create test results directory
        self.setup_directories()

    def setup_directories(self):
        dirs = [
            'test_results',
            'test_results/good',
            'test_results/defective_painting',
            'test_results/scratches',
            'test_results/metrics'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def load_test_data(self):
        # Load test data for each category
        self.test_loaders = {
            'good': self.get_dataloader("preprocessed_test/good"),
            'defective_painting': self.get_dataloader("preprocessed_test/defective_painting"),
            'scratches': self.get_dataloader("preprocessed_test/scratches")
        }
        
        # Load ground truth data
        self.ground_truth_loaders = {
            'defective_painting': self.get_dataloader("ground_truth/defective_painting"),
            'scratches': self.get_dataloader("ground_truth/scratches")
        }

    def get_dataloader(self, path):
        dataset = ImageFolder(self.base_dir / path, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def calculate_reconstruction_metrics(self, original, reconstruction):
        # Convert to numpy
        original_np = original.cpu().detach().numpy()
        recon_np = reconstruction.cpu().detach().numpy()
        
        # Calculate MSE
        mse = mean_squared_error(original_np.flatten(), recon_np.flatten())
        
        # Calculate PSNR
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        # Calculate SSIM-like metric
        mu1, mu2 = original_np.mean(), recon_np.mean()
        sigma1, sigma2 = original_np.std(), recon_np.std()
        sigma12 = ((original_np - mu1) * (recon_np - mu2)).mean()
        C1, C2 = (0.01 * 1) ** 2, (0.03 * 1) ** 2
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))
        
        return {
            'MSE': mse,
            'PSNR': psnr,
            'SSIM': ssim
        }

    def calculate_defect_metrics(self, reconstruction, ground_truth):
        # Ensure consistent batch sizes
        min_batch_size = min(reconstruction.size(0), ground_truth.size(0))
        reconstruction = reconstruction[:min_batch_size]
        ground_truth = ground_truth[:min_batch_size]
        
        # Calculate difference between reconstruction and ground truth
        diff = torch.abs(reconstruction - ground_truth)
        
        # Convert to binary mask using threshold
        threshold = 0.1
        pred_mask = (diff.mean(dim=1) > threshold).float()
        gt_mask = (ground_truth.mean(dim=1) > 0.5).float()
        
        # Calculate pixel-wise metrics
        true_pos = (pred_mask * gt_mask).sum().item()
        false_pos = (pred_mask * (1 - gt_mask)).sum().item()
        false_neg = ((1 - pred_mask) * gt_mask).sum().item()
        true_neg = ((1 - pred_mask) * (1 - gt_mask)).sum().item()
        
        # Calculate metrics
        precision = true_pos / (true_pos + false_pos + 1e-8)
        recall = true_pos / (true_pos + false_neg + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }

    def test_good_images(self):
        metrics_list = []
        print("\nTesting good images...")
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(self.test_loaders['good']):
                images = images.to(self.device)
                recon_images, _, _, _, _ = self.model(images)
                
                # Calculate reconstruction metrics
                metrics = self.calculate_reconstruction_metrics(images, recon_images)
                metrics_list.append(metrics)
                
                # Save sample reconstructions
                if batch_idx == 0:
                    comparison = torch.cat([images[:8], recon_images[:8]])
                    save_image(comparison.cpu(),
                             'test_results/good/reconstruction_comparison.png',
                             nrow=8, normalize=True)
        
        # Average metrics
        avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
        return avg_metrics

    def test_defective_images(self, category):
        metrics_list = []
        defect_metrics_list = []
        print(f"\nTesting {category} images...")
        
        test_loader = self.test_loaders[category]
        gt_loader = self.ground_truth_loaders[category]
        
        with torch.no_grad():
            for batch_idx, ((images, _), (gt_images, _)) in enumerate(zip(test_loader, gt_loader)):
                images = images.to(self.device)
                gt_images = gt_images.to(self.device)
                
                # Get reconstruction
                recon_images, _, _, _, _ = self.model(images)
                
                # Calculate metrics
                recon_metrics = self.calculate_reconstruction_metrics(images, recon_images)
                defect_metrics = self.calculate_defect_metrics(recon_images, gt_images)
                
                metrics_list.append(recon_metrics)
                defect_metrics_list.append(defect_metrics)
                
                # Save sample images
                if batch_idx == 0:
                    comparison = torch.cat([
                        images[:4],      # Original defective
                        recon_images[:4], # Reconstruction
                        gt_images[:4]     # Ground truth
                    ])
                    save_image(comparison.cpu(),
                             f'test_results/{category}/reconstruction_comparison.png',
                             nrow=4, normalize=True)
        
        # Average metrics
        avg_recon_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
        avg_defect_metrics = {k: np.mean([m[k] for m in defect_metrics_list]) for k in defect_metrics_list[0].keys()}
        
        return {**avg_recon_metrics, **avg_defect_metrics}

    def plot_metrics(self, metrics):
        # Create heatmap of metrics
        plt.figure(figsize=(12, 8))
        categories = list(metrics.keys())
        metric_types = list(metrics[categories[0]].keys())
        
        data = np.array([[metrics[cat][metric] for metric in metric_types] for cat in categories])
        
        sns.heatmap(data, annot=True, fmt='.3f', 
                    xticklabels=metric_types, 
                    yticklabels=categories)
        plt.title('Test Metrics Heatmap')
        plt.tight_layout()
        plt.savefig('test_results/metrics/metrics_heatmap.png')
        plt.close()

    def run_test(self):
        print(f"Starting testing on {self.device}")
        self.load_test_data()
        
        # Test results dictionary
        results = {
            'good': self.test_good_images(),
            'defective_painting': self.test_defective_images('defective_painting'),
            'scratches': self.test_defective_images('scratches')
        }
        
        # Print results
        print("\nTest Results:")
        for category, metrics in results.items():
            print(f"\n{category.upper()}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        # Plot metrics
        self.plot_metrics(results)
        
        return results

if __name__ == "__main__":
    tester = VAETester(
        base_dir="data/bracket_white",
        model_path="models/best_model.pth",  # or "models/final_model.pth"
        image_size=256,
        batch_size=32
    )
    results = tester.run_test()