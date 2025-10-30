from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch
from pathlib import Path
import json
from typing import Any, Dict, Optional
import time
from tqdm import tqdm

class GemmaRunner:
    """
    A lightweight runner that handles GPT calls, retries, saving responses,
    duplicate skipping, and progress display.
    """

    def __init__(
        self,
        jsonl_path: Path,
        model_id: str = "google/gemma-3n-e2b-it",
        max_new_tokens: int = 100,
    ) -> None:
        self.model_id = model_id
        self.jsonl_path = jsonl_path
        self.max_new_tokens = max_new_tokens
        self.jsonl_path.touch(exist_ok=True)


    # ---------- I/O helpers ----------
    def _load_done_ids(self) -> set[str]:
        """
        JSONL に記録されたレコードの "id" を既処理として扱う。
        outdir/*.json は参照しない（_save_per_id 廃止対応）。
        """
        done: set[str] = set()
        if not self.jsonl_path.exists():
            return done

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # 書きかけ行や壊れた行は読み飛ばす
                    continue

                if "id" in obj and obj["id"] is not None:
                    done.add(str(obj["id"]))
        return done

    def _dump_jsonl(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


    def run_all(self, queries: Dict[str, Any]) -> None:
        """
        Run all unprocessed ``(id, messages)`` in parallel and show a progress bar.
        """
        done_ids = self._load_done_ids()
        todo = [(str(i), m) for i, m in queries.items() if str(i) not in done_ids]

        if not todo:
            print("All queries were already processed.")
            return

        model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_id,
            device="cuda",
            torch_dtype=torch.bfloat16,
        ).eval()
        processor = AutoProcessor.from_pretrained(self.model_id)

        for i, m in tqdm(todo, desc="processing"):
            inputs = processor.apply_chat_template(
                m,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                generation = generation[0][input_len:]

            decoded = processor.decode(generation, skip_special_tokens=True)

            record = {
                "id": str(i),
                "text": decoded,
                "saved_at": int(time.time()),
                "model": self.model,
            }
            self._dump_jsonl(record)

        print(f"Finished: {len(todo)} items. Saved to: {self.jsonl_path}")


# Example usage:
if __name__ == "__main__":
    runner = GemmaRunner(jsonl_path=Path("output.jsonl"))

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                {"type": "text", "text": "What animal is on the candy?"}
            ]
        }
    ]
    queries = {
        "1": messages,
    }
    runner.run_all(queries)