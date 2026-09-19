# Soundscape on Linux

Tested on Ubuntu 24.04 with Docker Engine 27+, Compose v2.40 and the NVIDIA Container Toolkit.

1. **Driver + toolkit.** `nvidia-smi` must work on the host. Install the
   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   and run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. Check:
   `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`.
2. **Clone + configure.**
   ```bash
   git clone https://github.com/Sparaa/soundscape-monorepo && cd soundscape-monorepo
   cp .env.example .env        # MUSIC_BUDGET_GIB=12 on a 16 GB card; SOUNDSCAPE_GPU_UUID on multi-GPU boxes
   ```
3. **Build + start.** `make up` (= `docker compose --profile gpu --profile llm up -d --build`). First build pulls the
   torch stacks: ~20 GB of images, 15–40 min depending on bandwidth.
4. **Writer model.** `make pull-llm` (4.7 GB into the `ollama-models` volume). Skip the `llm` profile if you already
   run an OpenAI-compatible server: set `LLM_BASE_URL` (reachable from inside Docker — bind it on `0.0.0.0` or use
   `host.docker.internal`, not `127.0.0.1`) and `LLM_MODEL` in `.env`.
5. **Weights (optional).** `make weights` pre-downloads YuE2-3B, YuE2-Vae, SheetSage2 and MERT (~12 GB) so the first
   play doesn't. Otherwise they download on first use.
6. **Verify.** `curl -s localhost:3021/healthz | python3 -m json.tool` → `"ok": true` and every sidecar `ok`. Open
   http://localhost:3020, create a station, add a seed, press Play. `scripts/e2e.sh song.mp3` does the same from
   the shell and waits until two songs are cued.

Operations: `make down` stops; `docker compose --profile gpu --profile llm up -d` starts without rebuilding;
`make logs` tails the api; `make deploy-web` / `make deploy-api` rebuild one service without recreating the other
(a plain `up --build web` also recreates api and drops the in-memory play state). Tests: `make test` and
`make test-sidecars`. Ports bind to `127.0.0.1` only; put a reverse proxy in front if you want LAN access.
