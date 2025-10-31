import os
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
from dotenv import load_dotenv
load_dotenv()

from models.distribution_estimation import estimate_distribution, UnifiedConfig
from models.ot_calibration import SemiDiscreteOT1D


@dataclass
class MultiAttrTrainingConfig:
    """Training configuration parameters."""
    hidden_dim: int = 512
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    num_epochs: int = 100
    batch_size: int = 4096
    random_seed: int = 12
    log_interval: int = 10
    eval_qwk: bool = True
    ot_calibration: bool = False


@dataclass
class SingleAttrTrainingConfig:
    """Training configuration parameters."""
    hidden_dim: int = 256
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    num_epochs: int = 100
    batch_size: int = 4096
    random_seed: int = 12
    log_interval: int = 10
    eval_qwk: bool = True
    ot_calibration: bool = False


class PairwiseRankingDataset(Dataset):
    def __init__(
        self,
        embeddings_essay1: np.ndarray,
        embeddings_essay2: np.ndarray,
        labels: np.ndarray
    ):
        self.embeddings_essay1 = embeddings_essay1
        self.embeddings_essay2 = embeddings_essay2
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> dict[torch.Tensor, torch.Tensor, torch.Tensor]:
        return {
            'embedding_essay1': torch.from_numpy(self.embeddings_essay1[idx]).float(),
            'embedding_essay2': torch.from_numpy(self.embeddings_essay2[idx]).float(),
            'labels': torch.tensor(self.labels[idx]).float()
        }


class RankNet(nn.Module):
    """Neural network for Bradley-Terry scoring with regularization."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding)
    
    def compare(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Calculate Bradley-Terry probability: P(item1 > item2)."""
        score1 = self.forward(emb1)
        score2 = self.forward(emb2)
        return torch.sigmoid(score1 - score2)


class RankNetTrainer:
    """Handles training for multi-attribute Bradley-Terry model."""
    
    def __init__(
        self,
        model: nn.Module,
        config: MultiAttrTrainingConfig,
        sample_config: Optional[UnifiedConfig] = None,
        device: torch.device = None,
        eval_data: Optional[Dict] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config
        self.eval_data = eval_data
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # OT Calibration setup
        if self.config.ot_calibration:
            cfg = sample_config or UnifiedConfig()
            self.res = estimate_distribution(self.eval_data['true_scores'], cfg)
            print("OT Calibration Results:")
            print(self.res.stopped_n)
            print(self.res.overall_reason)
            print(f"{len(self.eval_data['true_scores']) - len(self.res.sample_indices)} samples used for evaluation.")

    def train(self, dataloader: DataLoader) -> Tuple[float, List[float], List[Dict]]:
        """Train model and return loss history and QWK history."""
        loss_history = []
        metrics_history = []
        
        for epoch in range(self.config.num_epochs):
            avg_loss = self._train_epoch(dataloader)
            loss_history.append(avg_loss)
            
            if (epoch + 1) % self.config.log_interval == 0:
                log_msg = f"Epoch [{epoch + 1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}"
                
                # QWK評価
                if self.config.eval_qwk and self.eval_data is not None:
                    metrics = self._evaluate_metrics()
                    metrics_history.append({'epoch': epoch + 1, **metrics})
                    
                    log_msg += f" | QWK: {metrics['qwk']:.3f}| Spearman: {metrics['spearman_corr']:.3f}"
                
                print(log_msg)
        
        return avg_loss, loss_history, metrics_history

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Execute single training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            emb1 = batch['embedding_essay1'].to(self.device)
            emb2 = batch['embedding_essay2'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            probs = self.model.compare(emb1, emb2)              # (B, 1)
            labels_u = labels.unsqueeze(1)                       # (B, 1)
            mask = (~torch.isnan(labels_u)).float()              # (B, 1)
            labels_filled = torch.nan_to_num(labels_u, nan=0.0)  # NaN -> 0（重みで無視）

            loss = F.binary_cross_entropy(
                probs, labels_filled, weight=mask, reduction='sum'
            ) / mask.sum().clamp_min(1.0)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)

    @torch.no_grad()
    def predict_scores(self, essay_embeddings: np.ndarray) -> np.ndarray:
        """
        Generate scores for all essays for all attributes.
        
        Args:
            num_essays: Total number of essays
        
        Returns:
            scores: (num_essays, num_attributes)
        """
        self.model.eval()
        
        essay_embeddings = torch.from_numpy(essay_embeddings).float().to(self.device)
        scores = self.model(essay_embeddings).cpu().numpy()
        
        return scores
    
    def _evaluate_metrics(self) -> Dict[str, float]:
        """Evaluate Metrics: QWK and Spearman's rank correlation."""
        if self.eval_data is None:
            return {}
        
        pred_latent_scores = self.predict_scores(self.eval_data['essay_embeddings'])
        y_true = self.eval_data['true_scores']
        y_min, y_max = self.eval_data['score_range']
        
        # Linear transformation
        if not self.config.ot_calibration:
            s_min, s_max = pred_latent_scores.min(), pred_latent_scores.max()
            y_pred = ((pred_latent_scores - s_min) / (s_max - s_min) * (y_max - y_min) + y_min)
            y_pred = np.round(y_pred).astype(int)
            
        # OT-based calibration
        else:
            x = pred_latent_scores[self.res.sample_indices]
            y = y_true[self.res.sample_indices]
            T = SemiDiscreteOT1D().fit(x, y)
            pred_latent_scores = np.delete(pred_latent_scores, self.res.sample_indices)
            y_true = np.delete(y_true, self.res.sample_indices)
            y_pred = T(pred_latent_scores)
        
        # Calculate QWK
        qwk = cohen_kappa_score(
            y_true,
            y_pred,
            weights='quadratic',
            labels=list(range(y_min, y_max + 1))
        )
        # Calculate Spearman's rank correlation
        spearman_corr, _ = spearmanr(pred_latent_scores, y_true)
        
        return {'qwk': qwk, 'spearman_corr': spearman_corr}


class MultiAttrRankNet(nn.Module):
    """Neural network for multi-attribute Bradley-Terry scoring with embedding lookup."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_attributes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_attributes = num_attributes
        
        # Architecture candidate 1
        # # Shared embedding encoder
        # self.shared_encoder = nn.Sequential(
        #     nn.Linear(embedding_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout)
        # )
        
        # # Attribute-specific heads
        # self.attribute_heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.LayerNorm(hidden_dim),
        #         nn.Dropout(p=dropout),
        #         nn.Linear(hidden_dim, hidden_dim // 2),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim // 2, 1)
        #     )
        #     for _ in range(num_attributes)
        # ])

        # Architecture candidate 2
        # Shared embedding encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        # Attribute-specific heads
        self.attribute_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim // 2),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_attributes)
        ])


    def forward(self, essay_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for all attributes simultaneously.
        Args:
            essay_indices: (batch_size, emb_dim)
        Returns:
            scores: (batch_size, num_attributes)
        """
        
        shared_features = self.shared_encoder(essay_embedding)  # (batch_size, hidden_dim)
        
        # Parallel processing for all attributes
        scores = []
        for head in self.attribute_heads:
            score = head(shared_features)  # (batch_size, 1)
            scores.append(score)
        
        return torch.cat(scores, dim=1)  # (batch_size, num_attributes)
    
    def compare(self, essay1_embedding: torch.Tensor, essay2_embedding: torch.Tensor) -> torch.Tensor:
        """
        Calculate Bradley-Terry probabilities for all attributes.
        Args:
            essay1_embedding: Essay embbeddings for first essays (batch_size, emb_dim)
            essay2_embedding: Essay embbeddings for second essays (batch_size, emb_dim)
        Returns:
            probs: (batch_size, num_attributes)
        """
        score1 = self.forward(essay1_embedding)  # (batch_size, num_attributes)
        score2 = self.forward(essay2_embedding)  # (batch_size, num_attributes)
        return torch.sigmoid(score1 - score2)  # (batch_size, num_attributes)


class MultiAttrRankNetTrainer:
    """Handles training for multi-attribute Bradley-Terry model."""
    
    def __init__(
        self,
        model: nn.Module,
        config: MultiAttrTrainingConfig,
        sample_config: Optional[UnifiedConfig] = None,
        device: torch.device = None,
        eval_data: Optional[Dict] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config
        self.eval_data = eval_data
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # OT Calibration setup
        if self.config.ot_calibration:
            cfg = sample_config or UnifiedConfig()
            self.res = estimate_distribution(self.eval_data['true_scores'], cfg)
            print("OT Calibration Results:")
            print(self.res.stopped_n, self.res.overall_reason)
            print(self.res.reason_per_metric)
            print(f"{len(self.eval_data['true_scores']) - len(self.res.sample_indices)} samples used for evaluation.")

    def train(self, dataloader: DataLoader) -> Tuple[float, List[float], List[Dict]]:
        """Train model and return loss history and QWK history."""
        loss_history = []
        qwk_history = []
        
        for epoch in range(self.config.num_epochs):
            avg_loss = self._train_epoch(dataloader)
            loss_history.append(avg_loss)
            
            if (epoch + 1) % self.config.log_interval == 0:
                log_msg = f"Epoch [{epoch + 1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}"
                
                # QWK評価
                if self.config.eval_qwk and self.eval_data is not None:
                    qwk_scores = self._evaluate_qwk()
                    qwk_history.append({'epoch': epoch + 1, **qwk_scores})
                    
                    avg_qwk = np.mean(list(qwk_scores.values()))
                    qwk_str = ', '.join([f"{k}: {v:.3f}" for k, v in qwk_scores.items()])
                    log_msg += f" | QWK: {qwk_str} | Avg: {avg_qwk:.3f}"
                
                print(log_msg)
        
        return avg_loss, loss_history, qwk_history

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Execute single training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            emb1 = batch['embedding_essay1'].to(self.device)
            emb2 = batch['embedding_essay2'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            probs = self.model.compare(emb1, emb2)              # (B, num_attributes)
            mask = (~torch.isnan(labels)).float()               # (B, num_attributes)
            labels_filled = torch.nan_to_num(labels, nan=0.0)   # NaN -> 0（重みで無視）

            loss = F.binary_cross_entropy(
                probs, labels_filled, weight=mask, reduction='sum'
            ) / mask.sum().clamp_min(1.0)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)

    @torch.no_grad()
    def predict_scores(self, essay_embeddings: np.ndarray) -> np.ndarray:
        """
        Generate scores for all essays for all attributes.
        
        Args:
            num_essays: Total number of essays
        
        Returns:
            scores: (num_essays, num_attributes)
        """
        self.model.eval()
        
        essay_embeddings = torch.from_numpy(essay_embeddings).float().to(self.device)
        scores = self.model(essay_embeddings).cpu().numpy()
        
        return scores
    
    def _evaluate_qwk(self) -> Dict[str, float]:
        """Evaluate QWK for each attribute."""
        if self.eval_data is None:
            return {}
        
        scores = self.predict_scores(self.eval_data['essay_embeddings'])
        qwk_scores = {}
        for i, attr in enumerate(self.eval_data['attributes']):
            latent_scores = scores[:, i]
            true_scores = self.eval_data['true_scores'][:, i]
            y_min, y_max = self.eval_data['score_ranges'][attr]
            
            # Linear transformation
            if not self.config.ot_calibration:
                s_min, s_max = latent_scores.min(), latent_scores.max()
                pred_scores = ((latent_scores - s_min) / (s_max - s_min) * (y_max - y_min) + y_min)
                pred_scores = np.round(pred_scores).astype(int)
                
            # OT-based calibration
            else:
                x = latent_scores[self.res.sample_indices]
                y = true_scores[self.res.sample_indices]
                T = SemiDiscreteOT1D().fit(x, y)
                latent_scores_for_eval = np.delete(latent_scores, self.res.sample_indices)
                true_scores = np.delete(true_scores, self.res.sample_indices)
                pred_scores = T(latent_scores_for_eval)
            
            # Calculate QWK
            qwk = cohen_kappa_score(
                true_scores,
                pred_scores,
                weights='quadratic',
                labels=list(range(y_min, y_max + 1))
            )
            
            qwk_scores[attr] = qwk
        
        return qwk_scores


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE)

    - experts: K個の専門家ネットワーク（同一アーキテクチャ）
    - gates:   タスクごとにK次元のゲーティング（softmax）を出力
    入力:  x              (B, in_dim)
    出力:  task_features  List[Tensor] (タスク数個の (B, out_dim))
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_experts: int,
        num_tasks: int,
        dropout: float = 0.3,
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Expert: 最小限で shared encoder 相当（Linear -> Act -> Dropout）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                activation,
                nn.Dropout(p=dropout),
                nn.Linear(out_dim, out_dim // 2),
                activation,
                nn.Dropout(p=dropout)
            )
            for _ in range(num_experts)
        ])

        # Task-wise gates: 各タスクごとに K（expert数）の重みを出す
        self.gates = nn.ModuleList([
            nn.Linear(in_dim, num_experts, bias=True)
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns:
            task_features: length=num_tasks の list。各要素は (B, out_dim)
        """
        # Expertの出力をまとめる: (B, K, out_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, K, D)

        task_features = []
        for gate in self.gates:
            # (B, K)
            gate_logits = gate(x)
            gate_weights = F.softmax(gate_logits, dim=-1)
            # 重み付き和: (B, 1, K) @ (B, K, D) -> (B, 1, D) -> (B, D)
            mixed = torch.bmm(gate_weights.unsqueeze(1), expert_outputs).squeeze(1)
            task_features.append(mixed)

        return task_features


class MultiAttrMMoERankNet(nn.Module):
    """
    MMoEベースの multi-attribute Bradley-Terry スコアリングモデル。
    既存の MultiAttrRankNet と同一インターフェース・同一入出力形状。

    Args:
        embedding_dim: 入力埋め込み次元
        num_attributes: タスク数（属性数）
        hidden_dim: MMoEの expert 出力次元（そのまま各タワー/ヘッドの入力次元）
        dropout: Dropout率（experts と heads に適用）
        num_experts: Expert の個数
        activation: Expert に使う活性化関数（デフォルト ReLU）
    """
    def __init__(
        self,
        embedding_dim: int,
        num_attributes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_experts: int = 3,
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.num_attributes = num_attributes

        # === MMoE (shared encoder の置き換え) ===
        self.mmoe = MMoE(
            in_dim=embedding_dim,
            out_dim=hidden_dim,
            num_experts=num_experts,
            num_tasks=num_attributes,
            dropout=dropout,
            activation=activation
        )

        # === Attribute-specific heads（既存と互換）===
        self.attribute_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim // 2),
                # nn.Dropout(p=dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_attributes)
        ])

    def forward(self, essay_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            essay_embedding: (B, embedding_dim)
        Returns:
            scores: (B, num_attributes)
        """
        # タスクごとの特徴 (num_attributes 個の (B, hidden_dim))
        task_features = self.mmoe(essay_embedding)

        # 各タスクを独立ヘッドでスコア化
        scores = []
        for feat, head in zip(task_features, self.attribute_heads):
            score = head(feat)  # (B, 1)
            scores.append(score)

        return torch.cat(scores, dim=1)  # (B, num_attributes)

    def compare(self, essay1_embedding: torch.Tensor, essay2_embedding: torch.Tensor) -> torch.Tensor:
        """
        Bradley-Terry 確率 P(item1 > item2) を属性ごとに出力。
        Args:
            essay1_embedding: (B, embedding_dim)
            essay2_embedding: (B, embedding_dim)
        Returns:
            probs: (B, num_attributes)
        """
        score1 = self.forward(essay1_embedding)  # (B, num_attributes)
        score2 = self.forward(essay2_embedding)  # (B, num_attributes)
        return torch.sigmoid(score1 - score2)    # (B, num_attributes)