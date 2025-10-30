```bash
uv venv
```

```bash
source .venv/bin/activate
```

```bash
uv sync
```

```bash
TQDM_DISABLE=1 nohup bash src/run_annotation.sh > log/annotation.log 2>&1 &
```