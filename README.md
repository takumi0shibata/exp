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
TQDM_DISABLE=1 nohup bash src/01_annotation.sh > logs/annotation.log 2>&1 &
```