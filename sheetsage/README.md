# vidmakr-sheetsage — SheetSage2 transcription sidecar

Song audio → melody-only ABC score, for YuE2 cover mode (see docs/music-plan.md §5).
Port 3016 (`SHEETSAGE_PORT`), GPU `MUSIC_GPU_UUID` (shared with vidmakr-music, sequential use).

API: `POST /transcribe {audio_b64, name, melody_only=true, max_seconds?}` → `{job_id}`;
`GET /jobs/{id}` → `{state, stage, progress, window/windows, abc, abc_error, warnings, seconds, key, bpm, bars,
sections, timing}`; `GET /jobs/{id}/midi`; `DELETE /jobs/{id}`; `POST /unload`; `GET /healthz`.

Weights: named volume `vidmakr_sheetsage-hf-cache` (`m-a-p/SheetSage2` 229 MB adapter + its parent
`m-a-p/MERT-v2-FullSong` 2.5 GB; both CC BY-NC 4.0). Seed: `docker run --rm --name vidmakr-sheetsage-weights
-v vidmakr_sheetsage-hf-cache:/data/hf-cache -e HF_HOME=/data/hf-cache python:3.11-slim sh -c "pip install -q
huggingface_hub==0.36.0 && python -c \"from huggingface_hub import snapshot_download as d; d('m-a-p/SheetSage2');
d('m-a-p/MERT-v2-FullSong')\""`. Runs with `SHEETSAGE_LOCAL_FILES_ONLY=1`.

Tests: `imagine test sheetsage` (pytest in the built image, fake model, no GPU).
