# vidmakr-music pins

| What | Pin | Why |
|---|---|---|
| YuE2 runtime | `multimodal-art-projection/YuE` @ `9c6c4b349be978b06a9d0d958471a07a6cdeff4d` (tag `yue2-v0.1.6`) | `ARG YUE2_REF` in the Dockerfile; bump deliberately, then re-run `imagine test music` and a live song |
| torch | 2.10.0 (PyPI wheel, bundles CUDA 12.8 libs → sm_120 kernels for the RTX PRO 6000) | pinned by yue2-infer's pyproject |
| transformers | 4.57.6 | pinned by yue2-infer (`modeling_yue2.py` remote code targets it) |
| Model | `m-a-p/YuE2-3B` (model.safetensors 7.26 GB) | weights CC BY-NC 4.0 — non-commercial, fine for the private box |
| Decoder | `m-a-p/YuE2-Vae` (0.53 GB) | listening decoder; `YuE2-Vae-legacy` is the benchmark decoder only |
| Python | 3.12 (`python:3.12-slim-bookworm`) | what the YuE2 docs target |

Weights live in the named volume `vidmakr_music-hf-cache` (HF hub layout under
`/data/hf-cache`). They were pre-seeded 2026-09-10 with `snapshot_download` of both
repos; the service runs with `MUSIC_LOCAL_FILES_ONLY=1` so it never needs internet.
To re-seed: `docker run --rm --name vidmakr-music-weights -v vidmakr_music-hf-cache:/data/hf-cache -e HF_HOME=/data/hf-cache python:3.12-slim sh -c "pip install -q huggingface_hub==0.36.2 && python -c \"from huggingface_hub import snapshot_download as d; d('m-a-p/YuE2-3B'); d('m-a-p/YuE2-Vae')\""`.

Runtime knobs (compose env): `MUSIC_BUDGET_GIB` (24; sets the per-process VRAM
fraction and the VAE tile size), `MUSIC_BACKEND` (torch | torch-eager | vllm — vllm
needs the `fast` extra, not installed), `MUSIC_QUANTIZATION` (none | fp8, experimental),
`MUSIC_IDLE_UNLOAD_MIN` (10), `MUSIC_JOB_RETAIN_MIN` (30), `MUSIC_VERIFY_HASHES`
(first | always | never — sha256 of 7.3 GB on every load is slow).

Known limits of the model: one song at a time; the semantic stage caps at 9000
tokens (~200 s of audio) and sets `truncated.semantic` when hit; length follows the
lyrics; mixed stereo only (no stems); English and Mandarin.
