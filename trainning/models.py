from transformers import CLIPModel
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import mean_squared_error, r2_score
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import CLIPModel


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

    def training_step(self, batch, batch_idx):
        outputs = self(*batch[:-1])  # All inputs except target
        loss = self.criterion(outputs, batch[-1])  # Last element is target
        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(*batch[:-1])
        loss = self.criterion(outputs, batch[-1])
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        self.validation_step_outputs.append({
            'val_loss': loss,
            'predictions': outputs,
            'targets': batch[-1]
        })
        return loss

    def on_validation_epoch_end(self):
        predictions = torch.cat(
            [x['predictions'] for x in self.validation_step_outputs]).cpu().numpy()
        targets = torch.cat([x['targets']
                            for x in self.validation_step_outputs]).cpu().numpy()

        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        self.log('mse', mse)
        self.log('r2', r2)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        scheduler = {
            'scheduler': CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=1e-6
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]


class MultimodalBasicModel(BaseMultimodalModel):
    def __init__(self, xgb_model, text_model, text_dim=768, learning_rate=0.001):
        super().__init__(learning_rate)
        self.xgb_feature_extractor = XGBoostFeatureExtractor(xgb_model)
        self.text_model = text_model
        self.text_fc = nn.Linear(text_dim, 32)

        self.img_model = models.efficientnet_b0(pretrained=True)
        for param in list(self.img_model.parameters())[:-4]:
            param.requires_grad = False

        self.combined_fc = nn.Sequential(
            nn.Linear(1033, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, tabular, text, img):
        tabular = tabular.float()
        xgb_pred = self.xgb_feature_extractor.extract_features(tabular)
        text_out = self.text_model(**text).last_hidden_state[:, 0, :]
        text_out = self.text_fc(text_out)
        img_out = self.img_model(img)
        combined = torch.cat((xgb_pred, img_out, text_out), dim=1)
        return self.combined_fc(combined).squeeze()


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


class MultimodalClipModel(BaseMultimodalModel):
    def __init__(self, xgb_model, clip_model_name="openai/clip-vit-base-patch32", learning_rate=0.001):
        super().__init__(learning_rate)
        self.xgb_feature_extractor = XGBoostFeatureExtractor(xgb_model)
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.clip_dim = self.clip.config.projection_dim

        self.combined_fc = nn.Sequential(
            nn.Linear(1 + self.clip_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, tabular, text_inputs, img):
        xgb_pred = self.xgb_feature_extractor.extract_features(tabular)
        clip_outputs = self.clip(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            pixel_values=img,
            return_dict=True
        )
        combined = torch.cat([
            xgb_pred,
            clip_outputs.image_embeds,
            clip_outputs.text_embeds
        ], dim=1)
        return self.combined_fc(combined).squeeze()

class MultimodalMLPAttentionModel(BaseMultimodalModel):
    def __init__(self, input_len, text_model, text_dim=768, learning_rate=0.001):
        super().__init__(learning_rate)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(input_len, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
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
        tabular_out = self.tabular_mlp(tabular.float())
        text_out = self.text_fc(self.text_model(
            **text).last_hidden_state[:, 0, :])
        img_out = self.image_fc(self.img_model(img))

        text_img = torch.stack([text_out, img_out], dim=0)
        attn_out, _ = self.attention(text_img, text_img, text_img)
        text_out, img_out = attn_out[0], attn_out[1]

        combined = torch.cat((tabular_out, img_out, text_out), dim=1)
        return self.combined_fc(combined).squeeze()
