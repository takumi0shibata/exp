from __future__ import annotations

"""Annotation pipeline utilities.

This script orchestrates the following steps for pairwise LLM-based essay annotation:

1. Load the target ASAP dataset split for a given prompt ID.
2. Build all (or a sampled subset of) *unordered* essay ID pairs.
3. Construct LLM messages using a system prompt, a user prompt template, an essay topic,
   and a rubric specific to the target attribute.
4. Execute batched LLM calls via :class:`utils.gpt_api.LLMRunner` and record JSONL outputs.
5. Post-process the JSONL into a CSV with gold preference labels and model predictions.

Functionality is preserved exactly; this pass adds comprehensive docstrings, type hints,
light refactoring of comments, logging, and minor typing correctness (e.g., accurate return types).
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
import random
from itertools import combinations
import logging

import polars as pl
import tiktoken
from dotenv import load_dotenv

from utils.helper import load_asap, set_seed
from utils.gpt_api import LLMRunner
from utils.gemma_api import GemmaRunner


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def configure_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure root logging.

    Parameters
    ----------
    log_level
        One of {"CRITICAL","ERROR","WARNING","INFO","DEBUG"}. Case-insensitive.
    log_file
        Optional path to a file to write logs; logs will also stream to stdout.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Build handlers: always stream to stdout; optionally also to file
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
        handlers=handlers,
        force=True,  # ensure reconfiguration if basicConfig was called elsewhere
    )

    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
        logging.getLogger(noisy).propagate = False


# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------
from typing import TypedDict


class ChatMessage(TypedDict):
    """Minimal message payload expected by the LLM runner."""
    role: str
    content: str


# Mapping from "{essay1_id}_{essay2_id}" -> list of chat messages
Queries = Dict[str, List[ChatMessage]]


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def draw_from_all_unique_pairs(
    xs: Iterable[Any],
    k: Optional[int] = None,
    seed: int = 12,
) -> List[Tuple[Any, Any]]:
    """Return a shuffled list of unique unordered pairs (combinations) from *xs*."""
    xs = list(xs)
    all_pairs = list(combinations(xs, 2))
    rnd = random.Random(seed)
    rnd.shuffle(all_pairs)
    logger.debug("Generated %d unique pairs (k=%s)", len(all_pairs), str(k))
    return all_pairs if k is None else all_pairs[:k]


def find_rubric_path(rubric_dir: Path, target_prompt: int) -> Path:
    """Find the rubric Markdown file in *rubric_dir* matching *target_prompt*."""
    logger.debug("Searching for rubric in %s for prompt %s", rubric_dir, target_prompt)
    try:
        rubric_filename = next(
            f for f in os.listdir(rubric_dir)
            if f.endswith(".md") and str(target_prompt) in re.findall(r"\d+", f)
        )
        path = rubric_dir / rubric_filename
        logger.info("Using rubric file: %s", path)
        return path
    except StopIteration:
        msg = f"Rubric file for prompt {target_prompt} not found in {rubric_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)


def load_text(path: Path) -> str:
    """Read a UTF-8 text file into a single string."""
    logger.debug("Loading text from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def count_total_tokens(queries: Queries, model: str = "gpt-4o-mini") -> int:
    """Count total tokens across all message contents for a given model.

    Uses :mod:`tiktoken` to retrieve the encoding for *model* and sums the
    token counts of each message's ``content`` field in *queries*.
    """
    enc = tiktoken.encoding_for_model(model)
    total_tokens = 0
    for qlist in queries.values():
        for msg in qlist:
            total_tokens += len(enc.encode(msg["content"]))
    logger.info("Estimated total tokens for %s: %d", model, total_tokens)
    return total_tokens


def create_messages(
    system_prompt: str,
    user_prompt: str,
    essay_topic: str,
    rubric: str,
    essay1: str,
    essay2: str,
    llm_name: str
) -> List[ChatMessage]:
    """Construct chat messages for a single essay pair."""
    user_content = (
        user_prompt
        .replace("{prompt}", essay_topic)
        .replace("{rubric}", rubric)
        .replace("{essay1}", essay1)
        .replace("{essay2}", essay2)
    )
    if 'gemma-3n' in llm_name.lower():
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_content}]},
        ]
    elif 'gpt-5' in llm_name.lower():
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    else:
        raise ValueError(f"Unsupported LLM name for message creation: {llm_name}")


def postprocess_to_csv(
    jsonl_path: Path,
    df: pl.DataFrame,
    target_prompt: int,
    target_att: str,
    queries: Queries,
    num_pairs: int,
    sample_dir: Path,
    model: str,
    seed: int,
) -> Path:
    """Convert LLM JSONL outputs into a CSV with gold vs. predicted preferences.

    The JSONL is expected to have lines containing an object with keys:
    - ``id``: a string like "{essay1_id}_{essay2_id}"
    - ``text``: a JSON-encoded object that includes ``{"preference": "essay1"|"essay2"}``

    The function parses these rows, aligns them with ground-truth scores from *df*,
    computes a *gold* preference (1 if essay1_score > essay2_score, 0 if less,
    else 0.5 for ties), maps model predictions to {1, 0, 0.5}, filters to only the
    sampled query pairs, and writes a CSV file to *sample_dir*.

    Returns
    -------
    Path
        The path to the written CSV file.
    """

    logger.info("Post-processing JSONL: %s", jsonl_path)

    # LLM output -> in-memory table
    results: Dict[str, List[Any]] = {
        "essay1_id": [],
        "essay2_id": [],
        "predicted_preference": [],
    }
    errors = 0
    total_lines = 0

    if not jsonl_path.exists():
        logger.warning("JSONL file not found: %s (skipping)", jsonl_path)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON line %d", total_lines)
                continue

            entry_id = entry.get("id")
            text = entry.get("text", "")

            try:
                output = json.loads(text)
                preference = output.get("preference")
            except json.JSONDecodeError:
                # Model did not return JSON; mark as undecided (0.5)
                preference = None
                errors += 1

            # Expect IDs like "123_456"
            results["essay1_id"].append(int(entry_id.split("_")[0]))
            results["essay2_id"].append(int(entry_id.split("_")[1]))
            results["predicted_preference"].append(preference)

    if errors:
        logger.warning("Malformed model outputs: %d/%d lines (treated as undecided)", errors, total_lines)
    else:
        logger.info("Parsed %d lines with no decode errors", total_lines)

    ids = df["essay_id"].to_list()
    final = (
        pl.DataFrame(results)
        .filter(
            pl.col("essay1_id").is_in(ids),
            pl.col("essay2_id").is_in(ids),
        )
        # Join ground-truth scores for each essay in the pair
        .join(df[["essay_id", target_att]], left_on="essay1_id", right_on="essay_id", how="left")
        .rename({target_att: "essay1_score"})
        .join(df[["essay_id", target_att]], left_on="essay2_id", right_on="essay_id", how="left")
        .rename({target_att: "essay2_score"})
        .with_columns(
            # Gold preference based on true attribute scores
            pl.when(pl.col("essay1_score") > pl.col("essay2_score")).then(1)
            .when(pl.col("essay1_score") < pl.col("essay2_score")).then(0)
            .otherwise(0.5)
            .alias("preference"),
            # Map model output string to numeric preference
            pl.when(pl.col("predicted_preference") == "essay1").then(1)
            .when(pl.col("predicted_preference") == "essay2").then(0)
            .otherwise(0.5)
            .alias("predicted_preference"),
        )
        # Keep only rows corresponding to sampled pairs
        .filter(
            pl.struct(["essay1_id", "essay2_id"]).is_in([
                {"essay1_id": int(key.split("_")[0]), "essay2_id": int(key.split("_")[1])}
                for key in queries.keys()
            ])
        )
    )

    sample_dir.mkdir(parents=True, exist_ok=True)
    out_csv = sample_dir / f"train_{model}_{target_prompt}_{seed}_{target_att}_{num_pairs}.csv"
    final.write_csv(out_csv.as_posix())

    logger.info("Exported CSV: %s (rows=%d)", out_csv, final.height)
    return out_csv


# ---------------------------------------------------------------------------
# Program entry point
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    """Run the end-to-end annotation workflow.

    Steps:
      1. Seed PRNGs for reproducibility.
      2. Load ASAP data for the target prompt.
      3. Resolve rubric path for the target attribute and prompt.
      4. Build LLM queries for *NUM_PAIRS* sampled essay pairs.
      5. Execute parallel LLM calls and collect outputs.
      6. Post-process outputs into a CSV summary.
    """
    # Configure logging first to capture subsequent messages
    configure_logging(args.log_level, args.log_file)

    load_dotenv()
    set_seed(args.seed)

    TARGET_PROMPT: int = args.target_prompt
    TARGET_ATT: str = args.target_att
    NUM_PAIRS: int = args.num_pairs

    logger.info(
        "Starting annotation run | prompt=%s | att=%s | pairs=%s | model=%s",
        TARGET_PROMPT,
        TARGET_ATT,
        NUM_PAIRS,
        args.model,
    )

    # 1) Data loading
    df = load_asap(TARGET_PROMPT)
    logger.info("Loaded ASAP split for prompt %s: %d essays", TARGET_PROMPT, df.height)

    # 2) Load rubric, prompts, and topic
    rubric_dir = args.llm_prompts_dir / TARGET_ATT
    try:
        rubric_path = find_rubric_path(rubric_dir, TARGET_PROMPT)
    except FileNotFoundError:
        # Already logged inside find_rubric_path
        raise

    system_prompt = load_text(args.llm_prompts_dir / "system.md")
    user_prompt = load_text(args.llm_prompts_dir / "user.md")
    essay_topic = load_text(args.llm_prompts_dir / "essay_prompts" / f"prompt_{TARGET_PROMPT}.md")
    rubric = load_text(rubric_path)

    # 3) Build queries: sample unique unordered essay ID pairs
    queries: Queries = {}
    for id_1, id_2 in draw_from_all_unique_pairs(df["essay_id"], k=NUM_PAIRS, seed=args.seed):
        queries[f"{id_1}_{id_2}"] = create_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            essay_topic=essay_topic,
            rubric=rubric,
            essay1=df.filter(pl.col("essay_id") == id_1)["essay"].item(),
            essay2=df.filter(pl.col("essay_id") == id_2)["essay"].item(),
            llm_name=args.model,
        )
    logger.info("Built %d query pairs", len(queries))

    # 4) Token accounting (informational)
    if 'gpt-5' in args.model:
        total = count_total_tokens(queries, model="gpt-4o-mini")
        logger.info("Total Tokens: %s", f"{total:,}")

    # 5) Output paths
    OUTDIR: Path = args.out_base / args.model.split('/')[-1] / str(TARGET_PROMPT) / TARGET_ATT
    os.makedirs(OUTDIR, exist_ok=True)
    JSONL_PATH: Path = OUTDIR / "results.jsonl"
    logger.debug("Output directory: %s", OUTDIR)

    # 6) Execute LLM calls via runner
    if 'gpt-5' in args.model:
        runner = LLMRunner(
            model=args.model,
            outdir=OUTDIR,
            jsonl_path=JSONL_PATH,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            base_sleep=args.base_sleep,
        )
    elif 'gemma-3n' in args.model:
        runner = GemmaRunner(jsonl_path=JSONL_PATH, model_id=args.model, max_new_tokens=512)
    logger.info("Dispatching %d LLM calls (batched)", len(queries))
    runner.run_all(queries)
    logger.info("LLM calls completed; results at %s", JSONL_PATH)

    # 7) Post-process: JSONL -> CSV
    postprocess_to_csv(
        jsonl_path=JSONL_PATH,
        df=df,
        target_prompt=TARGET_PROMPT,
        target_att=TARGET_ATT,
        queries=queries,
        num_pairs=NUM_PAIRS,
        sample_dir=args.sample_dir,
        model=args.model,
        seed=args.seed,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Annotation via LLMs")

    # General settings
    p.add_argument("--target-prompt", type=int, default=1, help="Prompt ID for annotation (e.g., 1)")
    p.add_argument("--target-att", type=str, default="overall", help="Target attribute for annotation (e.g., overall)")
    p.add_argument("--num-pairs", type=int, default=5000, help="Number of essay pairs to sample")
    p.add_argument("--seed", type=int, default=12, help="Random seed for pair selection")

    # Model/API settings
    p.add_argument("--model", type=str,
        default="gpt-5-mini-2025-08-07",
        choices=[
            "gpt-5-mini-2025-08-07",
            "gpt-5-2025-08-07",
            "google/gemma-3n-e2b-it",
            "google/gemma-3n-e4b-it"
        ],
        help="LLM model identifier for annotation"
    )
    p.add_argument("--max-workers", type=int, default=20, help="Maximum parallel LLM calls")
    p.add_argument("--max-retries", type=int, default=6, help="Maximum retries per LLM call")
    p.add_argument("--base-sleep", type=float, default=1.0, help="Base sleep time (seconds) between retries")

    # Directories
    p.add_argument("--llm-prompts-dir", type=Path, default=Path("./llm_prompts"), help="Directory containing prompt templates and rubrics")
    p.add_argument("--out-base", type=Path, default=Path("./out"), help="Base directory to save JSONL outputs")
    p.add_argument("--sample-dir", type=Path, default=Path("./sample"), help="Directory to save CSV samples")

    # Logging
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to write logs (in addition to stdout)",
    )

    args = p.parse_args()
    main(args)
