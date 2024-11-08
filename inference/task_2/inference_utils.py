import torch
import os
import joblib
import redis
import torch
import logging
import re
import numpy as np
from pathlib import Path
from difflib import get_close_matches
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torchvision import transforms
from torchvision import models
from transformers import DistilBertTokenizer, DistilBertModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Redis configuration with fallback
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_ENABLED = os.getenv('REDIS_ENABLED', 'true').lower() == 'true'


class ImageNotFoundException(Exception):
    """Custom exception for when an image cannot be found"""
    pass

def safe_log(x):
    return np.log1p(np.clip(x, 0, None))

def inverse_safe_log(x):
    return np.expm1(x)

def normalize_name(name):
    """Normalize a name by removing special characters and converting to lowercase."""
    return re.sub(r'[^a-z0-9]', '', name.lower())

def create_image_mapping(image_directory):
    """Create a mapping of normalized names to actual image paths."""
    image_mapping = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

    for file_path in Path(image_directory).glob('*'):
        if file_path.suffix.lower() in valid_extensions:
            normalized_name = normalize_name(file_path.stem)
            image_mapping[normalized_name] = file_path.name

    return image_mapping

def load_image(image_path: str) -> Image.Image:
    """Helper function to load image with appropriate format handling"""
    try:
        img = Image.open(image_path)

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
        raise ImageNotFoundException(f"Error loading image {image_path}: {str(e)}")

def find_image(description: str, image_directory: str) -> Image.Image:
    """Enhanced image loading logic with fuzzy matching"""
    try:
        normalized_search = normalize_name(description)
        
        image_mapping = create_image_mapping(image_directory)

        if normalized_search in image_mapping:
            image_path = os.path.join(
                image_directory,
                image_mapping[normalized_search]
            )
            return load_image(image_path)

        close_matches = get_close_matches(
            normalized_search,
            image_mapping.keys(),
            n=1,
            cutoff=0.8
        )
        if close_matches:
            image_path = os.path.join(
                image_directory,
                image_mapping[close_matches[0]]
            )
            logger.info(f"Using closest match '{close_matches[0]}' for '{description}'")
            return load_image(image_path)

        raise ImageNotFoundException(
            f"No matching image found for '{description}' (normalized: '{normalized_search}')"
        )

    except Exception as e:
        logger.error(f"Error finding/loading image: {str(e)}")
        raise ImageNotFoundException(str(e))

class XGBoostFeatureExtractor:
    def __init__(self, xgb_model):
        self.model = xgb_model

    def extract_features(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        y_pred = self.model.predict(X)
        tensor = torch.tensor(y_pred, dtype=torch.float32)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor.unsqueeze(1)

    def get_num_leaves(self):
        trees = self.model.get_dump()
        total_leaves = 0
        for tree in trees:
            leaf_count = tree.count('leaf=')
            total_leaves += leaf_count
        return total_leaves
    

class BaseMultimodalModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.validation_step_outputs = []
        self.grad_clip_val = 1.0

class MultimodalAttentionModel(BaseMultimodalModel):
    def __init__(self, xgb_model, text_model, text_dim=768, learning_rate=0.001):
        super().__init__(learning_rate)
        self.xgb_feature_extractor = XGBoostFeatureExtractor(xgb_model)
        self.text_model = text_model
        self.text_fc = nn.Linear(text_dim, 32)

        self.img_model = models.efficientnet_b0(pretrained=True)
        for param in list(self.img_model.parameters())[:-4]:
            param.requires_grad = False
        self.image_fc = nn.Linear(1000, 32)

        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=2)
        self.combined_fc = nn.Sequential(
            nn.Linear(65, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, tabular, text, img):
        tabular = tabular.float()
        xgb_pred = self.xgb_feature_extractor.extract_features(tabular)
        text_out = self.text_fc(self.text_model(
            **text).last_hidden_state[:, 0, :])
        img_out = self.image_fc(self.img_model(img))

        text_img = torch.stack([text_out, img_out], dim=0)
        attn_out, _ = self.attention(text_img, text_img, text_img)
        text_out, img_out = attn_out[0], attn_out[1]

        combined = torch.cat((xgb_pred, img_out, text_out), dim=1)
        return self.combined_fc(combined).squeeze()
    

class ModelManager:
    _instance = None
    
    # Singleton class for model management
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def initialize(self):
        if not self.initialized:
            logger.info("Initializing ModelManager...")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.tokenizer = None
            self.test_transform = None
            self.preprocessor = None
            self.initialized = True
            
    async def load_models(self):
        try:
            checkpoint_path = "./checkpoints/multimodal-attentionepoch=03-val_loss=0.08.ckpt"
            
            # Load XGBoost model
            xgb_model = joblib.load('./checkpoints/optuna_best_model.joblib')
            logger.info("XGBoost model loaded successfully")
            
            # Initialize text components
            text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            logger.info("Text model and tokenizer loaded successfully")
            
            # Initialize multimodal model
            self.model = MultimodalAttentionModel(
                xgb_model=xgb_model,
                text_model=text_model,
                text_dim=768,
                learning_rate=0.001
            )
            
            # Load checkpoint with weights_only=True to address the warning
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
            
            # Set models to eval mode
            self.model.eval()
            self.model.text_model.eval()
            self.model.img_model.eval()
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.model.text_model = self.model.text_model.cuda()
            
            # Initialize transforms and preprocessor
            self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self.preprocessor = joblib.load('./checkpoints/preprocessing_pipeline.joblib')
            logger.info("All models and components loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

class RedisWrapper:
    def __init__(self):
        self.client = None
        self.connected = False
        if REDIS_ENABLED:
            self.connect()

    def connect(self):
        try:
            self.client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=0,
                decode_responses=True,
                socket_connect_timeout=2
            )
            self.client.ping()
            self.connected = True
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.warning(f"Could not connect to Redis: {e}. Caching will be disabled.")
            self.connected = False

    def get(self, key):
        if not self.connected:
            return None
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def setex(self, key, timeout, value):
        if not self.connected:
            return
        try:
            self.client.setex(key, timeout, value)
        except Exception as e:
            logger.error(f"Redis setex error: {e}")