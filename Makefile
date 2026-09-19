.PHONY: up down test test-api test-web test-sidecars logs deploy-web deploy-api pull-llm weights
PROFILES = --profile gpu --profile llm
up: ; docker compose $(PROFILES) up -d --build
down: ; docker compose $(PROFILES) down
pull-llm: ; docker compose $(PROFILES) exec ollama ollama pull $${LLM_MODEL:-qwen2.5:7b-instruct}
# Pre-download the ~12 GB of model weights into the named volumes (otherwise they download on the first song).
weights:
	docker compose $(PROFILES) run --rm --no-deps yue2 python -c "from huggingface_hub import snapshot_download as d; d('m-a-p/YuE2-3B'); d('m-a-p/YuE2-Vae')"
	docker compose $(PROFILES) run --rm --no-deps sheetsage python -c "from huggingface_hub import snapshot_download as d; d('m-a-p/SheetSage2'); d('m-a-p/MERT-v2-FullSong')"
test: test-api test-web
test-api: ; docker run --rm -v $(PWD)/api:/app -w /app python:3.12-slim sh -c "apt-get install -y -qq ffmpeg >/dev/null 2>&1; pip install -q -r requirements.txt >/dev/null && python -m pytest -q tests -p no:cacheprovider"
test-web: ; docker run --rm -v $(PWD)/web:/app -w /app node:22-alpine sh -c "[ -d node_modules ] || npm install --silent --no-audit --no-fund; node node_modules/vitest/vitest.mjs run && node node_modules/typescript/bin/tsc --noEmit -p . && echo 'tsc: clean'"
# Sidecar suites run inside the built images (fake models, no GPU).
test-sidecars:
	docker run --rm -v $(PWD)/sidecars/yue2-sidecar:/app -w /app yue2-sidecar:latest sh -c "pip install -q pytest && python -m pytest -q tests -p no:cacheprovider"
	docker run --rm -v $(PWD)/sidecars/sheetsage-sidecar:/app -w /app sheetsage-sidecar:latest sh -c "python -m pytest -q tests -p no:cacheprovider"
	docker run --rm -v $(PWD)/sidecars/clipgrab-sidecar:/app -w /app --entrypoint sh clipgrab-sidecar:latest -c "pip install -q pytest && python -m pytest -q tests -p no:cacheprovider"
logs: ; docker compose logs -f --tail 100 api
# Rebuild ONE service without touching the other: a plain `up --build web` also rebuilds and RECREATES api (its
# dependency), which drops the radio's in-memory play state under a listening user.
deploy-web: ; docker compose up -d --build --no-deps web
deploy-api: ; docker compose up -d --build --no-deps api
