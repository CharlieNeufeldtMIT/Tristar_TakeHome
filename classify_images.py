import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from pathlib import Path
import numpy as np
import cv2

# Import the trained model
from vae_model import DetailVAE

class ImageClassifier:
    def __init__(self, model_path, base_dir, image_size=256, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.base_dir = Path(base_dir)
        self.image_size = image_size
        self.batch_size = batch_size

        # Load the trained model
        self.model = DetailVAE(latent_dim=128, image_size=image_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Create output directories
        self.setup_directories()

    def setup_directories(self):
        os.makedirs('classification_results', exist_ok=True)

    def load_images(self, path):
        dataset = ImageFolder(self.base_dir / path, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def classify_images(self, data_path, output_path, threshold=0.05):
        dataloader = self.load_images(data_path)
        results = []

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device)

                # Get reconstruction
                recon_images, _, _, _, _ = self.model(images)

                # Calculate defect maps
                defect_map = torch.abs(recon_images - images).cpu().numpy()
                reconstruction_error = np.mean(defect_map, axis=(1, 2, 3))  # Per image

                # Classify based on threshold
                classifications = ["good" if err < threshold else "defective" for err in reconstruction_error]

                # Save defect maps and classification results
                for i, (image, classification, error) in enumerate(zip(images, classifications, reconstruction_error)):
                    image_idx = batch_idx * self.batch_size + i
                    save_image(image.cpu(), f"{output_path}/{image_idx}_{classification}.png")
                    results.append({
                        "image_idx": image_idx,
                        "classification": classification,
                        "reconstruction_error": error
                    })

        return results

if __name__ == "__main__":
    classifier = ImageClassifier(
        model_path="models/best_model.pth",
        base_dir="data/bracket_white",
        image_size=256,
        batch_size=32
    )

    # Classify images
    results = classifier.classify_images(
        data_path="preprocessed_test/classify",
        output_path="classification_results"
    )

    # Print classification summary
    good_count = sum(1 for r in results if r['classification'] == 'good')
    defective_count = sum(1 for r in results if r['classification'] == 'defective')

    print(f"Good images: {good_count}")
    print(f"Defective images: {defective_count}")
