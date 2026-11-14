"""
Base model interface for Thesis multi-model benchmarking.
Standardizes training, inference, feature extraction, and profiling across all architectures.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseClassifier(ABC):
    """Abstract base class for all benchmarked visual models."""

    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = torch.device(device)
        self.model: nn.Module | None = None

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct the underlying neural architecture."""
        pass

    def get_param_count(self) -> Dict[str, int]:
        """Return total, trainable, and frozen parameter counts."""
        if self.model is None:
            return {"total": 0, "trainable": 0, "frozen": 0}
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    def get_model_size_mb(self, temp_path: Path | None = None) -> float:
        """Return serialized model checkpoint size in megabytes."""
        if self.model is None:
            return 0.0
        if temp_path is None:
            temp_path = Path("temp_model.pt")
        try:
            torch.save(self.model.state_dict(), temp_path)
            size_mb = temp_path.stat().st_size / (1024 ** 2)
            if temp_path.exists():
                temp_path.unlink()
            return float(size_mb)
        except Exception:
            return 0.0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 0.0003,
        weight_decay: float = 0.0001,
        patience: int = 3,
        verbose: bool = False
    ) -> Dict[str, list]:
        """Train classification head with early stopping."""
        if self.model is None:
            self.build_model()
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

        history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for images, labels, _ in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

            epoch_loss = total_loss / total
            epoch_acc = correct / total

            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += images.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            scheduler.step(epoch_val_loss)

            history["loss"].append(epoch_loss)
            history["acc"].append(epoch_acc)
            history["val_loss"].append(epoch_val_loss)
            history["val_acc"].append(epoch_val_acc)

            if verbose:
                print(f"[{self.name}] Epoch {epoch}/{epochs}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, val_loss={epoch_val_loss:.4f}, val_acc={epoch_val_acc:.4f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"[{self.name}] Early stopping at epoch {epoch}")
                    break

        if best_weights:
            self.model.load_state_dict(best_weights)

        return history

    def unfreeze_backbone(self, unfreeze: bool = True):
        """Unfreeze backbone parameters for end-to-end fine-tuning."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = unfreeze

    def fit_with_fine_tuning(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        warmup_epochs: int = 5,
        fine_tune_epochs: int = 10,
        lr_head: float = 0.0003,
        lr_backbone: float = 0.00001,
        weight_decay: float = 0.0001,
        patience: int = 4,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Two-stage fine-tuning:
        Phase 1: Linear probe warmup while backbone remains frozen.
        Phase 2: Discriminative end-to-end fine-tuning of backbone and head.
        """
        if verbose:
            print(f"[{self.name}] Phase 1: Linear Probe Warmup ({warmup_epochs} epochs)...")
        history = self.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=warmup_epochs,
            lr=lr_head,
            weight_decay=weight_decay,
            patience=patience,
            verbose=verbose
        )

        if fine_tune_epochs > 0:
            if verbose:
                print(f"[{self.name}] Phase 2: End-to-End Fine-Tuning ({fine_tune_epochs} epochs, lr_backbone={lr_backbone})...")
            self.unfreeze_backbone(True)

            head_params = [
                param for name, param in self.model.named_parameters()
                if any(h in name.lower() for h in ["classifier", "fc", "probe", "head", "heads"]) and param.requires_grad
            ]
            backbone_params = [
                param for name, param in self.model.named_parameters()
                if not any(h in name.lower() for h in ["classifier", "fc", "probe", "head", "heads"]) and param.requires_grad
            ]

            param_groups = []
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
            if head_params:
                param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": 0.0})

            optimizer = torch.optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-7)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float("inf")
            best_weights = None
            patience_counter = 0

            for epoch in range(1, fine_tune_epochs + 1):
                self.model.train()
                total_loss, correct, total = 0.0, 0, 0
                for images, labels, _ in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += images.size(0)

                epoch_loss = total_loss / total
                epoch_acc = correct / total

                # Validation
                self.model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for images, labels, _ in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * images.size(0)
                        preds = outputs.argmax(dim=-1)
                        val_correct += (preds == labels).sum().item()
                        val_total += images.size(0)

                epoch_val_loss = val_loss / val_total
                epoch_val_acc = val_correct / val_total
                scheduler.step()

                history["loss"].append(epoch_loss)
                history["acc"].append(epoch_acc)
                history["val_loss"].append(epoch_val_loss)
                history["val_acc"].append(epoch_val_acc)

                if verbose:
                    print(f"[{self.name} FT] Epoch {epoch}/{fine_tune_epochs}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, val_loss={epoch_val_loss:.4f}, val_acc={epoch_val_acc:.4f}")

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"[{self.name} FT] Early stopping triggered at fine-tuning epoch {epoch}")
                        break

            if best_weights:
                self.model.load_state_dict(best_weights)

        return history

    def predict_proba(self, loader: DataLoader) -> np.ndarray:
        """Generate class probabilities for input loader."""
        if self.model is None:
            raise RuntimeError(f"Model {self.name} has not been built.")
        self.model.to(self.device)
        self.model.eval()
        probabilities = []
        with torch.no_grad():
            for images, _, _ in loader:
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.softmax(logits, dim=-1)
                probabilities.append(probs.cpu().numpy())
        return np.concatenate(probabilities, axis=0)

    def predict(self, loader: DataLoader) -> np.ndarray:
        """Generate discrete class predictions."""
        probs = self.predict_proba(loader)
        return np.argmax(probs, axis=-1)

    @abstractmethod
    def extract_features(self, loader: DataLoader) -> np.ndarray:
        """Extract high-dimensional feature embeddings."""
        pass
