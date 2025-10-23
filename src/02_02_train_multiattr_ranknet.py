import os
import argparse
import pickle
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from typing import Optional

from utils.helper import load_asap, target_attribute, set_seed, get_min_max_scores
from models.ranknet_utils import (
    MultiAttrTrainingConfig,
    PairwiseRankingDataset,
    MultiAttrRankNet,
    MultiAttrRankNetTrainer
)
from models.distribution_estimation import UnifiedConfig


def pairs_equal(battles_dict: dict[str, pl.DataFrame]) -> bool:
    keys = list(battles_dict)
    ref = battles_dict[keys[0]].select(["essay1_id", "essay2_id"])
    return all(
        ref.equals(battles_dict[k].select(["essay1_id", "essay2_id"]))
        for k in keys[1:]
    )

def main(args, training_config: MultiAttrTrainingConfig, distribution_estimation_config: Optional[UnifiedConfig] = None):
    # -----------------------------
    # General Setup
    # -----------------------------
    TARGET_ATTRIBUTES = target_attribute(args.prompt)
    set_seed(args.seed)

    # -----------------------------
    # Data Preparation
    # -----------------------------
    # Load embeddings
    emb_dir = Path(os.getenv("EMB_DIR"))
    embedding_path = emb_dir / 'ASAP' / args.embedding_model / 'embeddings.pkl'
    with open(embedding_path, 'rb') as f:
        cached_embeddings = pickle.load(f)
    # Load battles for all attributes
    battles_dict: dict[str, pl.DataFrame] = {}
    for attr in TARGET_ATTRIBUTES:
        csv_path = f'./sample/train_{args.prompt}_{attr}_narrowed.csv'
        battles = pl.read_csv(csv_path)
        battles_dict[attr] = battles

    assert pairs_equal(battles_dict) , "Essay pairs do not match across attributes."

    # Collect labels for all attributes
    labels = np.zeros((len(battles_dict[TARGET_ATTRIBUTES[0]]), len(TARGET_ATTRIBUTES))) # (num_battles, num_attributes)
    for i, attr in enumerate(TARGET_ATTRIBUTES):
        labels[:, i] = battles_dict[attr]['predicted_preference'].to_numpy()
    # Collect embeddings for all essays in battles
    essay1_ids = battles_dict[TARGET_ATTRIBUTES[0]]['essay1_id'].to_numpy()
    essay2_ids = battles_dict[TARGET_ATTRIBUTES[0]]['essay2_id'].to_numpy()
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
        df = load_asap(args.prompt)
        df = df.select(['essay_id'] + TARGET_ATTRIBUTES)
        print('null count:', len(df.filter(pl.col('content').is_null())))
        df = df.drop_nulls()
        # Create eval data
        true_scores = df[TARGET_ATTRIBUTES].to_numpy()  # (num_essays, num_attributes)
        essay_embeddings = np.array([cached_embeddings[eid] for eid in df['essay_id']])  # (num_essays, emb_dim)

        # Get score ranges
        score_ranges = {attr: get_min_max_scores(args.prompt, attr) for attr in TARGET_ATTRIBUTES}

        eval_data = {
            'essay_embeddings': essay_embeddings,
            'attributes': TARGET_ATTRIBUTES,
            'true_scores': true_scores,
            'score_ranges': score_ranges
        }
    else:
        eval_data = None

    # -----------------------------
    # Model Training
    # -----------------------------
    model = MultiAttrRankNet(
        embedding_dim=embeddings_essay1.shape[1],
        num_attributes=len(TARGET_ATTRIBUTES),
        hidden_dim=training_config.hidden_dim,
        dropout=training_config.dropout_rate
    )
    trainer = MultiAttrRankNetTrainer(model, training_config, sample_config=distribution_estimation_config, eval_data=eval_data)
    _, loss_history, qwk_history = trainer.train(dataloader)

    met_df = pl.DataFrame(qwk_history, orient='row')
    met_df.write_csv(f'./results/main/multiattr_ranknet/{args.prompt}_{args.seed}_{args.num_pairs}_{args.model}_{args.ot_calibration}.csv')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', type=int, default=1)
    p.add_argument('--seed', type=int, default=12)
    p.add_argument('--num_pairs', type=int, default=5000)
    p.add_argument('--embedding_model', type=str, default='text-embedding-3-large')
    p.add_argument('--model', type=str, default='gpt-5-mini-2025-08-07')
    p.add_argument('--ot_calibration', action='store_true')
    args = p.parse_args()

    training_config = MultiAttrTrainingConfig(
        hidden_dim=512,
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
            stop_aggregator='all'
        )
    else:
        distribution_estimation_config = None

    print(vars(args))
    print(training_config)
    print(distribution_estimation_config) if distribution_estimation_config else None
    main(args, training_config, distribution_estimation_config)