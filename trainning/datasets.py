import os
import re
import logging
from pathlib import Path
from difflib import get_close_matches
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

logger = logging.getLogger(__name__)


class ImageNotFoundException(Exception):
    """Custom exception for when an image cannot be found"""
    pass


class BaseMultimodalDataset(Dataset):
    def __init__(self, tabular_data, text_data, image_folder, targets):
        self.tabular_data = torch.FloatTensor(
            tabular_data.toarray() if hasattr(tabular_data, 'toarray')
            else tabular_data
        )

        self.text_data = text_data.reset_index(drop=True)
        self.image_folder = image_folder
        self.targets = torch.FloatTensor(targets.values)

        # Create image mapping at initialization
        self.image_mapping = self.create_image_mapping()

        # Validate all images exist
        self.validate_images()

    def normalize_name(self, name):
        """Normalize a name by removing special characters and converting to lowercase."""
        return re.sub(r'[^a-z0-9]', '', name.lower())

    def create_image_mapping(self):
        """Create a mapping of normalized names to actual image paths."""
        image_mapping = {}
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

        for file_path in Path(self.image_folder).glob('*'):
            if file_path.suffix.lower() in valid_extensions:
                normalized_name = self.normalize_name(file_path.stem)
                image_mapping[normalized_name] = file_path.name

        return image_mapping

    def validate_images(self):
        """Validate that all text entries have corresponding images"""
        missing_images = []
        for idx in range(len(self.text_data)):
            text = self.text_data.iloc[idx]
            normalized_search = self.normalize_name(text)

            if normalized_search not in self.image_mapping:
                close_matches = get_close_matches(
                    normalized_search,
                    self.image_mapping.keys(),
                    n=1,
                    cutoff=0.8
                )
                if not close_matches:
                    missing_images.append((idx, text))

        if missing_images:
            error_msg = "\nMissing images for the following entries:\n"
            for idx, text in missing_images:
                error_msg += f"Index {idx}: {text}\n"
            raise ImageNotFoundException(error_msg)

    def load_image(self, image_path: str) -> Image.Image:
        """Helper function to load image with appropriate format handling"""
        try:
            img = Image.open(image_path)

            # Handle transparency for PNG images
            if img.format == 'PNG' and img.mode in ('P', 'RGBA'):
                img = img.convert('RGBA')
                background = Image.new('RGB', img.size, (255, 255, 255))
                if 'A' in img.getbands():
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img)
                return background
            else:
                return img.convert('RGB')
        except Exception as e:
            raise ImageNotFoundException(f"Error loading image {
                                         image_path}: {str(e)}")

    def get_image(self, text):
        """Enhanced image loading logic with fuzzy matching"""
        normalized_search = self.normalize_name(text)

        # Try exact match first
        if normalized_search in self.image_mapping:
            image_path = os.path.join(
                self.image_folder,
                self.image_mapping[normalized_search]
            )
            return self.load_image(image_path)

        # Try fuzzy matching if no exact match found
        close_matches = get_close_matches(
            normalized_search,
            self.image_mapping.keys(),
            n=1,
            cutoff=0.8
        )
        if close_matches:
            image_path = os.path.join(
                self.image_folder,
                self.image_mapping[close_matches[0]]
            )
            logger.info(f"Using closest match '{close_matches[0]}' for '{text}'")
            return self.load_image(image_path)

        raise ImageNotFoundException(
            f"No matching image found for '{
                text}' (normalized: '{normalized_search}')"
        )

    def __len__(self):
        return len(self.tabular_data)

    def process_text(self, text):
        """To be implemented by child classes"""
        raise NotImplementedError

    def process_image(self, image):
        """To be implemented by child classes"""
        raise NotImplementedError

    def __getitem__(self, idx):
        tabular = self.tabular_data[idx]
        text = self.text_data.iloc[idx]
        image = self.get_image(text)

        processed_text = self.process_text(text)
        processed_image = self.process_image(image)

        return tabular, processed_text, processed_image, self.targets[idx]


class MultimodalBERTDataset(BaseMultimodalDataset):
    def __init__(self, tabular_data, text_data, image_folder, targets, tokenizer, transform=None, max_length=128):
        super().__init__(tabular_data, text_data, image_folder, targets)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def process_text(self, text):
        text_encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in text_encoding.items()}

    def process_image(self, image):
        return self.transform(image) if self.transform else image


class MultimodalCLIPDataset(BaseMultimodalDataset):
    def __init__(self, tabular_data, text_data, image_folder, targets, max_length=128, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__(tabular_data, text_data, image_folder, targets)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.max_length = max_length

    def process_text(self, text):
        inputs = self.processor(
            text=text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in inputs.items() if k != 'pixel_values'}

    def process_image(self, image):
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        return inputs['pixel_values'].squeeze(0)
