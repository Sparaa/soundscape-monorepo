# Instructions for AI assistants helping install or run Soundscape

You are helping a person set up **Soundscape**, a self-hosted AI radio, most likely on a **Windows PC** with Docker
Desktop. Read this whole file before running anything. The human guides are `docs/WINDOWS.md` and `docs/LINUX.md`;
this file is the same knowledge organised for you: facts, a procedure with checks, a diagnosis playbook, and rules.

## 1. What this is (facts, not to be re-derived)

- **Everything runs in Docker.** Six services from one `docker-compose.yml` at the repo root:
  `web` (Next.js UI, :3020), `api` (FastAPI agent, :3021), `clipgrab` (yt-dlp, no GPU), `yue2` (song renderer, GPU),
  `sheetsage` (transcriber + CLAP, GPU), `ollama` (writer LLM, GPU). Ports are published on `127.0.0.1` only.
- **Profiles:** `yue2` and `sheetsage` are in profile `gpu`; `ollama` is in profile `llm`. The canonical start
  command is `docker compose --profile gpu --profile llm up -d --build`. Without the profiles those services are
  simply not started (compose does not error), and the api reports them offline.
- **Nothing runs natively on the host.** No Python, Node, pip, npm, CUDA toolkit or model download is ever needed on
  Windows or on the Linux host. If you find yourself about to install one of those, stop — you are off the path.
- **GPU:** NVIDIA only, ≥ 16 GB VRAM. In Docker Desktop on Windows the GPU is exposed through WSL 2 by the Windows
  driver; Docker Desktop already contains the NVIDIA container runtime. The one proof of a working GPU path is:
  `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` printing the GPU table.
- **Sizes and timings:** images ≈ 20 GB (first build 20–40 min), model weights ≈ 12 GB (downloaded into named
  volumes on first use, or ahead of time with `make weights` / `soundscape.ps1 weights`), writer LLM 4.7 GB
  (`pull-llm`). First song on a fresh install: 5–15 min. Steady state on a 4090: 70–100 s per 3-minute song.
- **Configuration is only `.env`** (template `.env.example`, every key documented there). `LIBRARY_DIR`
  (default `./library`) is the only persistent host data besides the three named volumes
  `yue2-hf-cache`, `sheetsage-hf-cache`, `ollama-models`.
- **Version pins are deliberate** (`sidecars/*/Dockerfile`, `requirements.txt`, `sidecars/yue2-sidecar/PINNED.md`):
  YuE2 needs torch 2.10 / transformers 4.57.6, SheetSage2 needs torch 2.8 / transformers 4.45.2 / python 3.11. Do not
  "upgrade" or "align" them to fix a build; a build failure has another cause (network, disk, CRLF).
- **The web build hard-codes `http://localhost:3021` for the browser.** `API_PORT` must stay 3021. `WEB_PORT` is free.
- **The api's LLM call** is a plain `POST {LLM_BASE_URL}/chat/completions` with `model`, `messages`, `temperature`,
  `max_tokens` and a `Bearer {LLM_API_KEY}` header. Any OpenAI-compatible server works; it must be reachable *from
  inside the api container* (so `127.0.0.1` on the host is wrong; `host.docker.internal` or a compose service name
  is right).
- **Windows shell facts:** use PowerShell. `curl` there is an alias for `Invoke-WebRequest` — use `curl.exe`. The
  command is `docker compose` (space), not `docker-compose`. `copy`, not `cp`, for files (though `cp` exists as an
  alias). Run all compose commands from the repo root. There is no `make` on Windows; `scripts/windows/soundscape.ps1`
  wraps the same commands, and the plain equivalents are at the end of `docs/WINDOWS.md`.

## 2. Procedure (do it in this order, one step at a time, run each check)

| # | Do | Check (run it, read the output) |
|---|---|---|
| 1 | Confirm hardware: `nvidia-smi` in PowerShell | GPU name and VRAM ≥ 16 GB. Otherwise stop: unsupported hardware. |
| 2 | WSL 2: `wsl --install` (admin), reboot, `wsl --update` | `wsl --status` → Default Version 2 |
| 3 | Docker Desktop installed, WSL 2 engine on | `docker version` shows a linux Server; `docker compose version` v2 |
| 4 | GPU through Docker | `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` prints the GPU |
| 5 | `.wslconfig` with `memory=` ≥ 12GB (24GB on 32 GB machines), `wsl --shutdown`, restart Docker Desktop | `docker run --rm alpine free -g` shows the new total |
| 6 | `git clone` the repo, `cd` into it | `git status` clean; `Get-Content sidecars/clipgrab-sidecar/entrypoint.sh -Raw` contains no `\r` (`.gitattributes` enforces LF) |
| 7 | `copy .env.example .env`; set `MUSIC_BUDGET_GIB=12` on 16 GB cards | `Get-Content .env` |
| 8 | `docker compose --profile gpu --profile llm up -d --build` (20–40 min) | `docker compose --profile gpu --profile llm ps` → six containers running |
| 9 | `docker compose --profile gpu --profile llm exec ollama ollama pull qwen2.5:7b-instruct` | `... exec ollama ollama list` shows it |
| 10 | Optional: pre-download weights (`make weights` or `soundscape.ps1 weights`) | volumes populated; no downloads in `logs yue2` at first play |
| 11 | `curl.exe http://localhost:3021/healthz` | `"ok": true`, every sidecar `"ok": true` |
| 12 | Open http://localhost:3020, make a station, add a seed, Play | a song arrives within 15 min; `docker compose logs -f api` shows plan → write → render → gate → cue |

Do not skip a check because a step "looked fine". Do not run step 8 before step 4 passes; the build alone takes
half an hour and cannot fix a GPU problem.

## 3. Diagnosis playbook

Always: (1) `docker compose --profile gpu --profile llm ps`, (2) `docker compose logs --tail 100 <service>` for
anything not `running`, (3) `curl.exe http://localhost:3021/healthz`. Read the actual output before naming a cause.

| Evidence | Cause | Fix |
|---|---|---|
| `could not select device driver "nvidia"` | Docker not on WSL 2 engine, or old Windows driver | Docker Desktop → Settings → General → WSL 2 engine; update the NVIDIA driver; re-run the step-4 test |
| `exec ./entrypoint.sh: no such file or directory` | CRLF line endings on checkout | `git config core.autocrlf false; git rm -r --cached .; git reset --hard`, then `up --build` |
| `torch.OutOfMemoryError` / `CUDA out of memory` in yue2 | VRAM budget too high, or another model resident | `MUSIC_BUDGET_GIB=12`, `OLLAMA_KEEP_ALIVE=0`, close other GPU apps; recreate: `up -d` |
| container exits 137 / `Killed` while loading | WSL VM out of RAM | raise `memory=` in `.wslconfig`; `wsl --shutdown`; restart Docker Desktop |
| `no space left on device` | Docker disk full | `docker system prune`; free the drive; never `down -v` (deletes weights) |
| `service "ollama" is not running` / `no such service` | `llm` profile not passed | add `--profile llm` |
| api log `llm 404 … model not found` | model not pulled or `LLM_MODEL` mismatch | `exec ollama ollama list`, fix `.env` or pull; `up -d` to apply `.env` |
| api log `ConnectError` / connection refused to LLM | endpoint bound to host `127.0.0.1` | use `host.docker.internal` and bind the server on `0.0.0.0` (`OLLAMA_HOST=0.0.0.0`) |
| healthz: a sidecar `ok: false` right after `up` | still starting (yue2/sheetsage import torch ≈ 30–60 s) | wait 1–2 min, re-check; then `logs <sidecar>` |
| `logs yue2` shows huggingface downloads | first-use weight download (12 GB) | wait; or `weights` command next time |
| song rejected by the gate repeatedly | too short / too quiet / unlike the station | normal occasionally; add seeds; check `logs api` for the gate reason |
| web shows *api offline* but healthz is fine | browser hitting the wrong port | `API_PORT` must be 3021; `WEB_PORT` matches the URL opened |
| `docker compose` complains about `deploy.resources.reservations.devices` | very old compose (v1) | use `docker compose` v2 that ships with Docker Desktop |

After any `.env` change: `docker compose --profile gpu --profile llm up -d` (recreates only the changed services).
After a code change in `api/` or `web/`: `up -d --build --no-deps api` (or `web`) — rebuilding web with deps
recreates api and drops the in-memory radio state.

## 4. Rules

**Do**
- Verify before diagnosing: run the check, paste the output, then name one cause and one fix. One change at a time.
- Prefer the documented commands. If the person prefers the script, `scripts/windows/soundscape.ps1 <cmd>` where
  cmd ∈ `check up down status logs pull-llm weights update test`.
- Tell the person how long a step takes before starting it (build 20–40 min, weights 12 GB, first song up to 15 min).
- Keep `.env` out of git and out of chat logs if it contains a hosted API key.

**Don't**
- Don't install Python, Node, CUDA, PyTorch, Ollama or model weights on the host to "help". The containers own that.
- Don't install NVIDIA drivers or `nvidia-container-toolkit` inside WSL on Windows. Docker Desktop provides it.
- Don't edit Dockerfiles, `requirements.txt` or version pins to get past a build error. Find the real error first.
- Don't remove the GPU `deploy:` blocks, switch to CPU, or set `count: all` without being asked.
- Don't change `API_PORT`. Don't rename containers or volumes.
- Don't run `docker compose down -v` or `docker volume rm` — that deletes 17 GB of downloaded weights.
- Don't `git pull` upstream component repos into this monorepo except via `git subtree pull` (see README "Upstream").
- Don't run `docker system prune -a --volumes`.

## 5. Where the code is, if you must read it

- `api/app/config.py` — every env var the api reads. `api/app/llm.py` — the LLM client. `api/app/main.py` — routes
  (`/healthz`, `/stations…`). `api/app/sidecars.py` — health probes of the three sidecars.
- `web/lib/api.ts` — the browser's API base URL. `web/app/` — pages; `web/lib/scenes.ts` + `web/lib/visual.ts` — visualizer.
- `sidecars/yue2-sidecar/app.py` (`/generate`, `/jobs/{id}`, `/healthz`), `sidecars/sheetsage-sidecar/app.py`
  (`/transcribe`, `/healthz`), `sidecars/clipgrab-sidecar/app.py` (`/clip`, `/info`, `/healthz`).
- Design: `docs/plan.md`. Visual feed contract: `docs/visual-feed.md`.

Tests (Linux or WSL shell, not needed for an install): `make test` (api pytest + web vitest/tsc in throwaway
containers), `make test-sidecars` (in the built images, fake models, no GPU).
