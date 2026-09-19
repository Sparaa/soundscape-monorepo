# Soundscape (monorepo)

**A radio that never runs out of songs.** Seed a station with your own music — a file, or a link `yt-dlp` can fetch —
and a music agent learns the seeds' sound and keeps composing: new songs in that style, reinterpretations, hooks with
new verses, the occasional straight cover with new words. Two songs stay cued while you listen. A beat-locked
visualizer fills the screen. Save what you like, build playlists, export them.

Self-hosted, single-user, personal use. Not a service. Runs on **Windows** (Docker Desktop + WSL2) and **Linux**.

This repository contains **everything** needed to run it — the app and the three model sidecars — so one
`git clone` and one `docker compose up` is the whole install.

| Path | What | Upstream |
|---|---|---|
| `api/` `web/` `docs/` `scripts/` | Soundscape itself: FastAPI agent + Next.js radio UI | [Sparaa/soundscape](https://github.com/Sparaa/soundscape) |
| `sidecars/yue2-sidecar/` | YuE2-3B song renderer (style + lyrics → stereo FLAC), port 3015 | [Sparaa/yue2-sidecar](https://github.com/Sparaa/yue2-sidecar) |
| `sidecars/sheetsage-sidecar/` | SheetSage2 melody/chord transcription + CLAP tagging, port 3016 | [Sparaa/sheetsage-sidecar](https://github.com/Sparaa/sheetsage-sidecar) |
| `sidecars/clipgrab-sidecar/` | yt-dlp fetcher for seed links (no GPU), port 3014 | [Sparaa/clipgrab-sidecar](https://github.com/Sparaa/clipgrab-sidecar) |
| `docker-compose.yml` | builds and runs all of it, plus an optional bundled Ollama for the writer LLM | — |
| `docs/WINDOWS.md` | step-by-step Windows install | — |
| `docs/LINUX.md` | Linux install | — |
| `AGENTS.md` / `CLAUDE.md` | instructions for an AI assistant (Claude Code, Copilot, Cursor, ChatGPT…) helping you set it up | — |

## What you need

- **An NVIDIA GPU with 16 GB VRAM or more** (24 GB recommended). YuE2 peaks at 11–14 GB; SheetSage2/CLAP and the
  writer LLM load in turn and unload when idle. A 4090 renders a 3-minute song in roughly 70–100 s; a 16 GB card is
  slower but works. AMD / Intel / Apple GPUs are **not** supported.
- **32 GB system RAM** recommended (model loading is RAM-hungry; 16 GB works with a tuned `.wslconfig` on Windows).
- **About 60 GB free disk** where Docker keeps its data: ~20 GB of images, ~12 GB of model weights, ~5 GB for the LLM.
- **Docker**: Docker Desktop with the WSL 2 engine on Windows 10 21H2+/11; Docker Engine + NVIDIA Container Toolkit
  on Linux. Nothing else is installed on the host — no Python, no Node, no CUDA toolkit.
- **A writer LLM** (OpenAI-compatible endpoint). The bundled Ollama (`--profile llm`) is the default; any local
  server (LM Studio, vLLM, llama.cpp) or hosted API works instead — see `.env.example`.

## Quick start

**Windows** — full walkthrough with checks in [`docs/WINDOWS.md`](docs/WINDOWS.md). The short version, in PowerShell:

```powershell
git clone https://github.com/Sparaa/soundscape-monorepo
cd soundscape-monorepo
.\scripts\windows\soundscape.ps1 check      # Docker, WSL2 engine, GPU visible from a container
copy .env.example .env                      # edit MUSIC_BUDGET_GIB=12 if your card has 16 GB
.\scripts\windows\soundscape.ps1 up         # builds ~20 GB of images: 20-40 min the first time
.\scripts\windows\soundscape.ps1 pull-llm   # downloads the 4.7 GB writer model into the Ollama volume
.\scripts\windows\soundscape.ps1 weights    # optional: fetch the 12 GB of music weights now instead of at first play
.\scripts\windows\soundscape.ps1 status     # every line should say ok
start http://localhost:3020
```

**Linux** — details in [`docs/LINUX.md`](docs/LINUX.md):

```bash
git clone https://github.com/Sparaa/soundscape-monorepo && cd soundscape-monorepo
cp .env.example .env
docker compose --profile gpu --profile llm up -d --build      # or: make up
docker compose --profile gpu --profile llm exec ollama ollama pull qwen2.5:7b-instruct   # or: make pull-llm
curl -s localhost:3021/healthz
xdg-open http://localhost:3020
```

Day to day: the containers restart with Docker. `docker compose --profile gpu --profile llm down` stops everything,
`… up -d` (without `--build`) starts it again. To update: `git pull`, then `up -d --build`.

## Using it

1. **Make a station** on the home page and give it a seed: upload a song, or paste a link. SheetSage2 transcribes the
   melody and chords into a score, CLAP describes the sound. One or more seeds become the *station profile*.
2. **Press Play.** The agent plans each track (*inspired* 50 % · *faithful cover* 20 % · *reinterpret* 15 % · *hook*
   15 %), the LLM writes a title, style line and lyrics fitted to the melody's phrasing, YuE2 renders, a gate rejects
   tracks that are too short, too quiet or don't sound like the station, and the buffer keeps two songs cued. The
   **first song of a fresh install takes several minutes** (weights download + model load); after that a new song is
   usually ready before the current one ends.
3. **Steer.** Skip, ♥ more like this, 👎 less like this, move the covers ↔ new slider, add more seeds.
4. **Keep.** Save songs (FLAC + a JSON with style, lyrics, score and plan) into the library, build playlists, export a
   zip with an `.m3u`. Unsaved radio songs are pruned after 24 h. The library lives in `./library` (see `LIBRARY_DIR`).
5. **Watch.** The visualizer is driven by the player's analyser: a beat clock locked to the planned BPM, section
   changes from the score, palettes from the station's mood. Drop your own logo at `web/public/logo.png` and rebuild
   `web`. Every frame is also published as a JSON feed (`docs/visual-feed.md`).

## Configuration

All settings are environment variables read from `.env` (documented in `.env.example`). The ones people change:

| Variable | Default | Notes |
|---|---|---|
| `MUSIC_BUDGET_GIB` | `20` | VRAM YuE2 may use. **Set `12` on a 16 GB card.** |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | bundled Ollama, `qwen2.5:7b-instruct` | any OpenAI-compatible chat endpoint reachable **from inside Docker** |
| `OLLAMA_KEEP_ALIVE` | `0` | bundled Ollama only; `0` frees the GPU for YuE2 right after each call |
| `SOUNDSCAPE_GPU_UUID` | empty (= first GPU) | pick a GPU on multi-GPU machines (`nvidia-smi -L`) |
| `LIBRARY_DIR` | `./library` | where songs, seeds and the SQLite DB persist |
| `WEB_PORT` | `3020` | browser port; keep `API_PORT` at `3021` |

Compose profiles: `gpu` runs the two GPU sidecars, `llm` runs the bundled Ollama. Already running YuE2/SheetSage or an
LLM elsewhere? Set the `*_URL` variables and start without that profile.

## How the pieces fit

```
browser ──► web  (Next.js 15 · WebAudio two decks + crossfade · three.js scenes · visual feed)   :3020
              │
              ▼
            api  (FastAPI · SQLite · library on disk)                                              :3021
              ingest → analyze → profile → agent → write → render → gate → cue
              │            │                          │        │
              ▼            ▼                          ▼        ▼
          clipgrab     sheetsage                    LLM      yue2
          yt-dlp       SheetSage2 + CLAP          Ollama     YuE2-3B
          :3014        :3016  (GPU)          :11434 (GPU)    :3015 (GPU)
```
Design, decisions and phases: `docs/plan.md`. Feed contract: `docs/visual-feed.md`. Sidecar HTTP contracts: each
sidecar's `README.md`. Tests: `make test` (api + web in throwaway containers) and `make test-sidecars`.

## Troubleshooting

`docker compose ps` should list six running containers; `curl localhost:3021/healthz` (Windows: `curl.exe`) names
any unreachable sidecar. More in the OS guides, and a diagnosis playbook for assistants in `AGENTS.md`.

- *api offline / sidecar offline* on the home page → a container is down or still building; `docker compose logs <service>`.
- *First song takes minutes* → normal on a fresh install: weights download, YuE2 loads (~1 min), the gate may reject
  an attempt. Run `make weights` / `soundscape.ps1 weights` once to move the download out of your first listen.
- *CUDA out of memory* → `MUSIC_BUDGET_GIB=12`, `OLLAMA_KEEP_ALIVE=0`, close other GPU apps (games, browser video).
- *LLM errors in the api log* → the endpoint must be reachable from inside Docker; with the bundled Ollama, did you
  `pull-llm`? (`docker compose --profile llm exec ollama ollama list`).
- *Everything sounds the same* → add more seeds, move the covers ↔ new slider, vote 👎 on the direction you dislike.

## Licenses

Soundscape's and the sidecars' code is **Apache-2.0** (see the `LICENSE` files). The default models they drive are
not: **YuE2-3B**, **SheetSage2** and **MERT-v2-FullSong** are **CC BY-NC 4.0** (non-commercial); **CLAP**
(`laion/larger_clap_music_and_speech`) is Apache-2.0; Qwen2.5 is Apache-2.0. You download those weights yourself, for
personal use, under their terms. Links are fetched with yt-dlp only for your own analysis; nothing is redistributed.

## Upstream

The four component repos stay the source of truth for their code; this monorepo vendors them with `git subtree`
(history preserved). To pull upstream changes: `git subtree pull --prefix=sidecars/yue2-sidecar https://github.com/Sparaa/yue2-sidecar main`
(same for the other two), and `git pull https://github.com/Sparaa/soundscape main` for the app.
