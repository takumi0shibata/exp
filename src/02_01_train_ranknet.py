import argparse
import polars as pl
import numpy as np
from pathlib import Path
import pickle
import torch
from torch.utils.data import DataLoader
from typing import Optional
import os

from models.ranknet_utils import (
    SingleAttrTrainingConfig,
    PairwiseRankingDataset,
    RankNet,
    RankNetTrainer
)
from models.distribution_estimation import UnifiedConfig
from utils.helper import load_asap, calculate_agreement_rates, get_min_max_scores, set_seed


def main(args, training_config: SingleAttrTrainingConfig, distribution_estimation_config: Optional[UnifiedConfig] = None):
    # -----------------------------
    # General Setup
    # -----------------------------
    set_seed(args.seed)

    # -----------------------------
    # Data Preparation
    # -----------------------------
    battles = pl.read_csv(f"./sample/train_{args.model}_{args.prompt}_{args.seed}_{args.attribute}_{args.num_pairs}.csv")
    table, overall_agreement_rate, non_tie_agreement_rate = calculate_agreement_rates(battles)
    print(f"Overall agreement rate (including ties): {overall_agreement_rate:.4f}")
    print(f"Non-tie agreement rate: {non_tie_agreement_rate:.4f}")
    print(table)

    # Load embeddings
    emb_dir = Path("./embeddings")
    embedding_path = emb_dir / 'ASAP' / args.embedding_model / 'embeddings.pkl'
    with open(embedding_path, 'rb') as f:
        cached_embeddings = pickle.load(f)
    # Collect labels for all attributes
    labels = battles['predicted_preference'].to_numpy() # (num_battles,)
    # Collect embeddings for all essays in battles
    essay1_ids = battles['essay1_id'].to_numpy()
    essay2_ids = battles['essay2_id'].to_numpy()
    # Look up embeddings from cache
    embeddings_essay1 = np.array([cached_embeddings[eid] for eid in essay1_ids]) # (num_battles, emb_dim)
    embeddings_essay2 = np.array([cached_embeddings[eid] for eid in essay2_ids]) # (num_battles, emb_dim)
    # Create dataset
    dataset = PairwiseRankingDataset(embeddings_essay1, embeddings_essay2, labels)
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    if training_config.eval_qwk:
        # Create eval data
        df = load_asap(args.prompt)[['essay_id', args.attribute]]
        df = df.drop_nulls()
        true_scores = df[args.attribute].to_numpy()  # (num_essays,)
        essay_embeddings = np.array([cached_embeddings[eid] for eid in df['essay_id']])  # (num_essays, emb_dim)
        score_range = get_min_max_scores(args.prompt, args.attribute)
        eval_data = {
            'essay_embeddings': essay_embeddings,
            'true_scores': true_scores,
            'score_range': score_range
        }
    else:
        eval_data = None

    # -----------------------------
    # Model Training
    # -----------------------------
    model = RankNet(
        embedding_dim=embeddings_essay1.shape[1],
        hidden_dim=training_config.hidden_dim,
        dropout=training_config.dropout_rate,
    )
    trainer = RankNetTrainer(model, training_config, sample_config=distribution_estimation_config, eval_data=eval_data)
    _, loss_history, metrics_history = trainer.train(dataloader)

    met_df = pl.DataFrame(metrics_history, orient='row')
    os.makedirs(f'./results/main/ranknet/', exist_ok=True)
    met_df.write_csv(f'./results/main/ranknet/{args.prompt}_{args.attribute}_{args.seed}_{args.num_pairs}_{args.model}_{args.ot_calibration}.csv')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', type=int, default=1)
    p.add_argument('--attribute', type=str, default='overall')
    p.add_argument('--seed', type=int, default=12)
    p.add_argument('--num_pairs', type=int, default=5000)
    p.add_argument('--embedding_model', type=str, default='text-embedding-3-large')
    p.add_argument('--model', type=str, default='gpt-5-mini-2025-08-07')
    p.add_argument('--ot_calibration', action='store_true')
    args = p.parse_args()

    training_config = SingleAttrTrainingConfig(
        hidden_dim=256,
        num_epochs=100,
        ot_calibration=args.ot_calibration,
    )
    if args.ot_calibration:
        distribution_estimation_config = UnifiedConfig(
            max_samples=100,   # Max budget
            batch_size=10,     # add 10 samples per iteration
            ks_tol=0.03,       # KS distance tolerance
            band_supwidth_tol=0.15,  # 95%同時帯の最大幅しきい値
            B_boot=400,        # ブートストラップ回数
            seed=args.seed,
        )
    else:
        distribution_estimation_config = None

    print(vars(args))
    print(training_config)
    print(distribution_estimation_config) if distribution_estimation_config else None
    main(args, training_config, distribution_estimation_config)
