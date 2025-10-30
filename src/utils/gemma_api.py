from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch
from pathlib import Path
import json
from typing import Any, Dict, List, Tuple
import time
from tqdm import tqdm

def _chunk(seq: List[Tuple[str, Any]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

class GemmaRunner:

    def __init__(
        self,
        jsonl_path: Path,
        model_id: str = "google/gemma-3n-e2b-it",
        max_new_tokens: int = 100,
        batch_size: int = 10,
    ) -> None:
        self.model_id = model_id
        self.jsonl_path = jsonl_path
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.jsonl_path.touch(exist_ok=True)

    # ---------- I/O helpers ----------
    def _load_done_ids(self) -> set[str]:
        done: set[str] = set()
        if not self.jsonl_path.exists():
            return done
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
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
        未処理の (id -> messages) をミニバッチでまとめて生成
        """
        done_ids = self._load_done_ids()
        todo: List[Tuple[str, Any]] = [
            (str(i), m) for i, m in queries.items() if str(i) not in done_ids
        ]
        if not todo:
            print("All queries were already processed.")
            return

        # モデルとプロセッサをロード
        model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        processor = AutoProcessor.from_pretrained(self.model_id)

        # pad_token_id が無いと警告が出るモデルがあるので補完
        if getattr(model.generation_config, "pad_token_id", None) is None:
            try:
                model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
            except Exception:
                pass

        total = len(todo)
        pbar = tqdm(total=total, desc="processing")

        for batch in _chunk(todo, self.batch_size):
            ids = [i for i, _ in batch]
            conversations = [m for _, m in batch]

            try:
                # まとめてテンプレート適用＆トークナイズ
                # ※HFのChatTemplateは複数会話のリストを受け取れます
                inputs = processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,   # ← ミニバッチ整形
                )

                # デバイスとdtypeを正しく移動:
                #  - 整数テンソルは整数のまま
                #  - 浮動小数だけモデルのdtypeに
                model_dtype = next(model.parameters()).dtype
                moved = {}
                for k, v in inputs.items():
                    if torch.is_floating_point(v):
                        moved[k] = v.to(model.device, dtype=model_dtype, non_blocking=True)
                    else:
                        moved[k] = v.to(model.device, non_blocking=True)
                inputs = moved

                pad_side = getattr(processor.tokenizer, "padding_side", "right")
                max_in_len = inputs["input_ids"].shape[1]
                input_lens = inputs["attention_mask"].sum(dim=1) if "attention_mask" in inputs else None

                with torch.inference_mode():
                    sequences = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=0.1,
                        eos_token_id=getattr(model.generation_config, "eos_token_id", None) or processor.tokenizer.eos_token_id,
                    )

                records = []
                for row_idx, qid in enumerate(ids):
                    if pad_side == "left":
                        start = max_in_len
                    else:
                        start = int(input_lens[row_idx].item()) if input_lens is not None else max_in_len
                    gen_part = sequences[row_idx, start:]
                    text = processor.tokenizer.decode(gen_part, skip_special_tokens=True)
                    records.append({"id": qid, "text": text, "saved_at": int(time.time()), "model": self.model_id})

                # 書き出し
                for r in records:
                    self._dump_jsonl(r)

            except Exception as e:
                # バッチ失敗時は各サンプルを単発でリトライ（保守的に）
                for qid, conv in batch:
                    try:
                        single_inputs = processor.apply_chat_template(
                            conv,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_tensors="pt",
                            return_dict=True,
                        )
                        model_dtype = next(model.parameters()).dtype
                        moved = {}
                        for k, v in single_inputs.items():
                            if torch.is_floating_point(v):
                                moved[k] = v.to(model.device, dtype=model_dtype, non_blocking=True)
                            else:
                                moved[k] = v.to(model.device, non_blocking=True)
                        single_inputs = moved

                        input_len = (
                            single_inputs["attention_mask"].sum().item()
                            if "attention_mask" in single_inputs
                            else single_inputs["input_ids"].shape[-1]
                        )

                        with torch.inference_mode():
                            out = model.generate(
                                **single_inputs,
                                max_new_tokens=self.max_new_tokens,
                                do_sample=False,
                            )[0]

                        decoded = processor.decode(out[input_len:], skip_special_tokens=True)
                        self._dump_jsonl({
                            "id": qid,
                            "text": decoded,
                            "saved_at": int(time.time()),
                            "model": self.model_id,
                        })
                    except Exception as ee:
                        # このサンプルは諦め、ログだけ残す
                        self._dump_jsonl({
                            "id": qid,
                            "error": f"{type(ee).__name__}: {ee}",
                            "saved_at": int(time.time()),
                            "model": self.model_id,
                        })

            finally:
                pbar.update(len(batch))

        pbar.close()
        print(f"Finished: {total} items. Saved to: {self.jsonl_path}")


# Example usage:
if __name__ == "__main__":
    runner = GemmaRunner(jsonl_path=Path("output.jsonl"), batch_size=8, max_new_tokens=100)

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
        # "2": other_messages, ...
    }
    runner.run_all(queries)