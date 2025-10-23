import polars as pl
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    np.random.seed(seed)
    os.environ['PYTHONSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_asap(
    prompt: int,
    data_path: str = './data/asap_with_traits.csv',
    stratify: bool = False,
) -> pl.DataFrame:
    df = pl.read_csv(data_path, infer_schema_length=15000)
    df = df.filter(pl.col('essay_set') == prompt)
    df = df.drop_nulls(subset=['overall'])

    if not stratify:
        return df
    else:
        score_count = df.group_by('overall').len()
        test_df = df.filter(pl.col('overall').is_in(score_count.filter(pl.col('len')==1)['overall']))
        df_remaining = df.filter(~pl.col('overall').is_in(score_count.filter(pl.col('len')==1)['overall']))

        _, tmp_test_df = train_test_split(df_remaining, test_size=0.1, stratify=df_remaining['overall'], random_state=123)
        test_df = pl.concat([test_df, tmp_test_df], how='vertical')
        return test_df


def calculate_agreement_rates(df: pl.DataFrame):
    # Calculate the score difference
    df = df.with_columns(
        (abs(pl.col('essay1_score') - pl.col('essay2_score')).alias('score_diff'))
    )
    # Calculate agreement rate for each score difference, including 'tie' preferences
    agreement_rates = (
        df.group_by('score_diff')
        .agg([
            (pl.col('preference') == pl.col('predicted_preference')).mean().alias('agreement_rate'),
            pl.len().alias('count')
        ])
        .sort('score_diff')
    )
    # Calculate overall agreement rate, including 'tie' preferences
    overall_agreement_rate = (
        df.select((pl.col('preference') == pl.col('predicted_preference')).mean())
        .item()
    )
    # Calculate agreement rate for non-tie preferences
    non_tie_agreement_rate = (
        df.filter(pl.col('preference') != 0.5)
        .select((pl.col('preference') == pl.col('predicted_preference')).mean())
        .item()
    )
    return agreement_rates, overall_agreement_rate, non_tie_agreement_rate


def get_min_max_scores(prompt: int, attribute: str) -> tuple:
    attribute_ranges = {
        1: {'overall': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6), 'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        2: {'overall': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6), 'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        3: {'overall': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        4: {'overall': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        5: {'overall': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        6: {'overall': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        7: {'overall': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6), 'conventions': (0, 6), 'style': (0, 6)},
        8: {'overall': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word_choice': (2, 12), 'sentence_fluency': (2, 12), 'conventions': (2, 12), 'voice': (2, 12)},
    }
    
    if prompt in attribute_ranges and attribute in attribute_ranges[prompt]:
        return attribute_ranges[prompt][attribute]
    else:
        raise ValueError(f"Invalid prompt {prompt} or attribute {attribute}.")


def target_attribute(prompt: int) -> list[str]:
    attribute_map = {
        1: ['overall', 'content', 'organization', 'word_choice', 'sentence_fluency', 'conventions'],
        2: ['overall', 'content', 'organization', 'word_choice', 'sentence_fluency', 'conventions'],
        3: ['overall', 'content', 'prompt_adherence', 'language', 'narrativity'],
        4: ['overall', 'content', 'prompt_adherence', 'language', 'narrativity'],
        5: ['overall', 'content', 'prompt_adherence', 'language', 'narrativity'],
        6: ['overall', 'content', 'prompt_adherence', 'language', 'narrativity'],
        7: ['overall', 'content', 'organization', 'conventions', 'style'],
        8: ['overall', 'content', 'organization', 'word_choice', 'sentence_fluency', 'conventions', 'voice'],
    }
    return attribute_map.get(prompt, [])