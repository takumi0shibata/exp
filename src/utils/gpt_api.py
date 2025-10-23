from __future__ import annotations

import json
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple, Optional

from tqdm import tqdm
from openai import OpenAI
from openai import APIError, RateLimitError, InternalServerError, APITimeoutError


class LLMRunner:
    """
    A lightweight runner that handles GPT calls, retries, saving responses,
    duplicate skipping, and progress display.
    """

    def __init__(
        self,
        model: str,
        outdir: Path,
        jsonl_path: Path,
        max_workers: int = 20,
        max_retries: int = 6,
        base_sleep: float = 1.0,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.model = model
        self.outdir = outdir
        self.jsonl_path = jsonl_path
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        self.client = client or OpenAI()

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.touch(exist_ok=True)

    # ---------- I/O helpers ----------
    def _load_done_ids(self) -> set:
        """
        Treat IDs found in ``outdir/*.json`` and in the JSONL as already processed.
        """
        done = {p.stem for p in self.outdir.glob("*.json")}
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        done.add(str(obj["id"]))
                except json.JSONDecodeError:
                    pass
        return done

    def _dump_jsonl(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def _save_per_id(self, id_: str, resp_obj: Dict[str, Any]) -> None:
        out_path = self.outdir / f"{id_}.json"
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(resp_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)  # almost atomic

    @staticmethod
    def _to_dict_safe(response) -> Dict[str, Any]:
        """
        Be resilient to pydantic v2 and compatible client objects.
        Returns a plain ``dict`` representation of the response when possible.
        """
        try:
            return response.model_dump()
        except AttributeError:
            try:
                return json.loads(response.model_dump_json())
            except Exception:
                try:
                    return json.loads(response.json())
                except Exception:
                    return {"_raw_repr": repr(response)}

    # ---------- Call OpenAI ----------
    def _call_api_once(self, id_: str, messages) -> Any:
        """
        Call the Responses API. For models whose name contains 'gpt-5',
        use minimal reasoning settings; otherwise use standard settings.
        """
        if "gpt-5" in self.model:
            return self.client.responses.create(
                model=self.model,
                input=messages,
                reasoning={"effort": "minimal"},
                text={"verbosity": "low"},
            )
        else:
            return self.client.responses.create(
                model=self.model,
                max_output_tokens=1024,
                temperature=0.0,
                input=messages,
            )

    def _call_with_retry(self, id_: str, messages):
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_api_once(id_, messages)
            except (RateLimitError, APITimeoutError, InternalServerError, APIError):
                if attempt == self.max_retries:
                    raise
                sleep = self.base_sleep * (2 ** (attempt - 1))
                sleep = sleep * (0.5 + random.random())  # jitter
                time.sleep(sleep)
            except Exception:
                # Re-raise unexpected exceptions (original behavior)
                raise

    def _worker(self, id_: str, messages):
        resp = self._call_with_retry(id_, messages)

        try:
            text = resp.output_text
        except Exception:
            text = None

        resp_dict = self._to_dict_safe(resp)

        # 1) Save per-ID JSON
        self._save_per_id(id_, resp_dict)

        # 2) Append to JSONL
        record = {
            "id": str(id_),
            "text": text,
            "saved_at": int(time.time()),
            "model": self.model,
        }
        self._dump_jsonl(record)

        return id_, text

    # ---------- Public ----------
    def run_all(self, queries: Dict[str, Any]) -> None:
        """
        Run all unprocessed ``(id, messages)`` in parallel and show a progress bar.
        """
        done_ids = self._load_done_ids()
        todo = [(str(i), m) for i, m in queries.items() if str(i) not in done_ids]

        if not todo:
            print("All queries were already processed.")
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._worker, i, m) for i, m in todo]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="processing"):
                pass

        print(f"Finished: {len(todo)} items. Saved to: {self.jsonl_path} and {self.outdir}/")