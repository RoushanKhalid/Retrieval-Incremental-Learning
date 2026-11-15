"""
CLIP Model Wrapper: Visual encoder, zero-shot classifier, and linear probe classifier.
Provides semantic representation and feature extraction via openai/clip-vit-base-patch32.
"""


import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
try:
    from transformers import CLIPModel, CLIPProcessor
except (ImportError, AttributeError):
    from transformers.models.clip.modeling_clip import CLIPModel
    from transformers.models.clip.processing_clip import CLIPProcessor

from .base_model import BaseClassifier


class CLIPWrapper(BaseClassifier):
    """CLIP visual backbone with linear probe head and zero-shot capability."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        num_classes: int = 2
    ):
        super().__init__(name="CLIP (ViT-B/32)", device=device)
        self.model_name = model_name
        self.num_classes = num_classes
        self.embedding_dim = 512
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
            self.clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True, local_files_only=True)
        except Exception:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        self.build_model()
        if self.model is not None:
            self.model.to(self.device)

    def build_model(self) -> nn.Module:
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Linear probe head on top of frozen 512-dim CLIP visual embedding
        self.probe = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.embedding_dim, self.num_classes)
        )

        class FullModel(nn.Module):
            def __init__(self, clip_model, probe):
                super().__init__()
                self.clip_model = clip_model
                self.probe = probe

            def forward(self, pixel_values):
                features = self.clip_model.get_image_features(pixel_values=pixel_values)
                if hasattr(features, "image_embeds"):
                    features = features.image_embeds
                elif hasattr(features, "pooler_output"):
                    features = features.pooler_output
                features = nn.functional.normalize(features, dim=-1)
                return self.probe(features)

        self.model = FullModel(self.clip_model, self.probe)
        return self.model

    def unfreeze_backbone(self, unfreeze: bool = True):
        """Unfreeze visual projection and top transformer block for fine-tuning."""
        if hasattr(self.clip_model, "visual_projection"):
            for p in self.clip_model.visual_projection.parameters():
                p.requires_grad = unfreeze
        if hasattr(self.clip_model, "vision_model"):
            for layer in self.clip_model.vision_model.encoder.layers[-2:]:
                for p in layer.parameters():
                    p.requires_grad = unfreeze

    def extract_features(self, loader: DataLoader) -> np.ndarray:
        """Extract 512-dimensional normalized CLIP visual embeddings."""
        self.clip_model.to(self.device)
        self.clip_model.eval()
        features = []
        with torch.no_grad():
            for images, _, _ in loader:
                images = images.to(self.device)
                emb = self.clip_model.get_image_features(pixel_values=images)
                if hasattr(emb, "image_embeds"):
                    emb = emb.image_embeds
                elif hasattr(emb, "pooler_output"):
                    emb = emb.pooler_output
                emb = nn.functional.normalize(emb, dim=-1)
                features.append(emb.cpu().numpy())
        return np.concatenate(features, axis=0)

    def extract_features_from_paths(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        """Extract features directly from image file paths."""
        self.clip_model.to(self.device)
        self.clip_model.eval()
        all_embeddings = []

        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            pil_images = []
            for p in batch_paths:
                with Image.open(p) as img:
                    pil_images.append(img.convert("RGB"))

            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
                if hasattr(features, "image_embeds"):
                    features = features.image_embeds
                elif hasattr(features, "pooler_output"):
                    features = features.pooler_output
                features = nn.functional.normalize(features, dim=-1)
                all_embeddings.append(features.cpu().numpy())

        return np.vstack(all_embeddings)

    def zero_shot_predict(self, loader: DataLoader, class_descriptions: list[str]) -> np.ndarray:
        """Evaluate zero-shot text-image classification."""
        self.clip_model.to(self.device)
        self.clip_model.eval()

        text_inputs = self.processor(
            text=class_descriptions,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = nn.functional.normalize(text_features, dim=-1)

        all_probs = []
        with torch.no_grad():
            for images, _, _ in loader:
                images = images.to(self.device)
                img_features = self.clip_model.get_image_features(pixel_values=images)
                if hasattr(img_features, "image_embeds"):
                    img_features = img_features.image_embeds
                elif hasattr(img_features, "pooler_output"):
                    img_features = img_features.pooler_output
                img_features = nn.functional.normalize(img_features, dim=-1)

                similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
                all_probs.append(similarity.cpu().numpy())

        return np.concatenate(all_probs, axis=0)
