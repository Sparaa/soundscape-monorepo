# Soundscape on Windows

Soundscape runs on Windows entirely inside Docker Desktop's WSL 2 engine. Nothing is installed on Windows itself
except Git, Docker Desktop and the NVIDIA driver — no Python, no Node, no CUDA toolkit, and **never** an NVIDIA
driver inside WSL (the Windows driver is shared with WSL automatically).

Every step below has a check. If a check fails, fix it before moving on — later steps can't succeed without it.
If an AI assistant is helping you, point it at `AGENTS.md` in the repo root.

## 0. Hardware and Windows

| Need | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, 16 GB VRAM (e.g. RTX 4060 Ti 16 GB, 4080, 5070 Ti) | 24 GB (RTX 3090 / 4090 / 5090) |
| RAM | 16 GB (with the `.wslconfig` in step 3) | 32 GB |
| Disk | 60 GB free on the drive Docker Desktop uses (`C:` by default) | SSD |
| Windows | Windows 10 21H2 or Windows 11, 64-bit, virtualization enabled in the BIOS/UEFI | Windows 11 |
| NVIDIA driver | a current Game Ready or Studio driver (2025 or later) | latest |

Check: open PowerShell and run `nvidia-smi`. It must print your GPU and a driver version. If it doesn't, install the
driver from nvidia.com first.

## 1. WSL 2

Open **PowerShell as Administrator**:

```powershell
wsl --install
```
Reboot when asked. After the reboot an Ubuntu window may open and ask for a username/password — pick anything, you
won't use it. Then check:

```powershell
wsl --status          # "Default Version: 2"
wsl -l -v             # Ubuntu ... VERSION 2
wsl --update          # keeps the WSL kernel current (needed for GPU support)
```

If `wsl --install` says virtualization is disabled, enable "Intel VT-x" / "AMD-V" / "SVM" in your BIOS/UEFI.

## 2. Docker Desktop

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) and start it.
2. **Settings → General**: "Use the WSL 2 based engine" must be checked.
3. **Settings → Resources → WSL integration**: enable integration with your Ubuntu distro (optional but harmless).
4. Check, in a normal PowerShell:

```powershell
docker version                  # Client and Server sections, Server OS/Arch: linux/amd64
docker compose version          # v2.x
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```
The last command must print the same GPU table as `nvidia-smi` on Windows. **That line is the whole GPU setup** —
Docker Desktop ships the NVIDIA container runtime, and WSL forwards the Windows driver. If it fails with
`could not select device driver "nvidia"`, Docker Desktop is not on the WSL 2 engine or the Windows driver is too old.

## 3. Memory for WSL (`.wslconfig`)

Model loading needs several GB of RAM inside the Docker VM. WSL takes 50 % of your RAM by default; give it more if
you can. Create the file `C:\Users\<your name>\.wslconfig` (Notepad is fine; no extension) with:

```ini
[wsl2]
memory=24GB
swap=8GB
```
Use `memory=12GB` on a 16 GB machine, `memory=24GB` on 32 GB. Then apply it:

```powershell
wsl --shutdown
```
and start Docker Desktop again (it restarts the VM). Check: `docker run --rm alpine free -g` shows the new total.

## 4. Get the code

Install [Git for Windows](https://git-scm.com/download/win) (defaults are fine), then in PowerShell:

```powershell
cd $HOME
git clone https://github.com/Sparaa/soundscape-monorepo
cd soundscape-monorepo
```
The repo carries a `.gitattributes` that keeps Linux scripts in LF line endings even on Windows, so no Git settings
need changing. Check: `git config --get core.autocrlf` may say `true`; that's fine.

## 5. Configure

```powershell
copy .env.example .env
notepad .env
```
The defaults work. Change these if they apply to you:

- `MUSIC_BUDGET_GIB=12` if your GPU has **16 GB** of VRAM (leave `20` for 24 GB and up).
- `SOUNDSCAPE_GPU_UUID=GPU-xxxx` if you have more than one NVIDIA GPU (`nvidia-smi -L` lists them).
- The LLM block, only if you don't want the bundled Ollama (see "Other writer LLMs" below).

## 6. Build and start

```powershell
.\scripts\windows\soundscape.ps1 check
.\scripts\windows\soundscape.ps1 up
```
`check` repeats the tests from steps 2–5. `up` runs
`docker compose --profile gpu --profile llm up -d --build`: it builds six images (about 20 GB, the two GPU sidecars
download their torch stacks) — expect 20–40 minutes the first time, seconds afterwards.

If PowerShell refuses to run the script ("running scripts is disabled"), either run the `docker compose` commands
from the table at the end directly, or once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Check: `docker compose ps` lists `soundscape-web`, `-api`, `-yue2`, `-sheetsage`, `-clipgrab`, `-ollama`, all
`running` (the api may say `unhealthy`/restarting for a minute while the sidecars come up).

## 7. Writer model and (optionally) music weights

```powershell
.\scripts\windows\soundscape.ps1 pull-llm     # 4.7 GB, qwen2.5:7b-instruct into the ollama volume
.\scripts\windows\soundscape.ps1 weights      # optional: 12 GB of music weights now, instead of during your first play
```
Check: `.\scripts\windows\soundscape.ps1 status` prints the health JSON with `"ok": true`, every sidecar `ok`, and
the Ollama model list containing `qwen2.5:7b-instruct`.

## 8. Listen

Open **http://localhost:3020**. Create a station, upload a song (or paste a YouTube link), press **Play**.

The very first song of a fresh install takes **5–15 minutes**: the weights download if you skipped step 7, YuE2 loads
(about a minute), and the quality gate may reject the first attempt and try again. Watch it happen:

```powershell
.\scripts\windows\soundscape.ps1 logs        # api log; Ctrl+C to stop following
.\scripts\windows\soundscape.ps1 logs yue2   # the renderer's progress
```
After that, the buffer keeps two songs cued and a new one is usually ready before the current one ends.

## Day to day

| Do | Command |
|---|---|
| Stop everything | `.\scripts\windows\soundscape.ps1 down` |
| Start again (no rebuild) | `.\scripts\windows\soundscape.ps1 up` (fast once built) |
| See what's running / health | `.\scripts\windows\soundscape.ps1 status` |
| Update to the newest code | `.\scripts\windows\soundscape.ps1 update` (= `git pull` + rebuild) |
| Where are my songs? | `soundscape-monorepo\library\` — FLAC + JSON per saved song, seeds, `soundscape.db` |
| Free disk space | `docker system prune` removes old build layers (never the model volumes) |

Quitting Docker Desktop stops the containers; they come back when it starts (`restart: unless-stopped`). Only
`localhost` can reach the ports; nothing is exposed to your network.

## Other writer LLMs

The api only needs an OpenAI-compatible `/v1/chat/completions` endpoint **reachable from inside Docker**. Edit `.env`
and start **without** `--profile llm` (use `docker compose --profile gpu up -d` instead of the script's `up`):

- **Ollama for Windows** (installed on the host): `LLM_BASE_URL=http://host.docker.internal:11434/v1`,
  `LLM_MODEL=<a model you pulled>`. Ollama listens on `127.0.0.1` only by default; set the Windows environment
  variable `OLLAMA_HOST=0.0.0.0` and restart Ollama if the api log says the LLM is unreachable.
- **LM Studio**: start its server (Developer tab, "Serve on local network" on), then
  `LLM_BASE_URL=http://host.docker.internal:1234/v1`, `LLM_MODEL=<the model name it shows>`.
- **Hosted**: `LLM_BASE_URL=https://api.openai.com/v1`, `LLM_MODEL=gpt-4o-mini`, `LLM_API_KEY=sk-...` (or any
  OpenAI-compatible provider). Cheapest option for a 16 GB card, since it leaves the whole GPU to YuE2.

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | Docker Desktop not on the WSL 2 engine, or the Windows NVIDIA driver is old. Step 2's `nvidia-smi` test must pass. |
| `exec ./entrypoint.sh: no such file or directory` (clipgrab) | CRLF line endings. Run `git config core.autocrlf false; git rm -r --cached .; git reset --hard` in the repo, then `up` again. |
| `CUDA out of memory` in `logs yue2` | Set `MUSIC_BUDGET_GIB=12`, keep `OLLAMA_KEEP_ALIVE=0`, close games / GPU-accelerated apps, then `up`. |
| Container `Killed` / exit 137 while loading a model | The WSL VM ran out of RAM: raise `memory=` in `.wslconfig`, `wsl --shutdown`, restart Docker Desktop. |
| `no space left on device` during build | Free space on the Docker drive, `docker system prune`, or move Docker's disk image (Settings → Resources → Advanced). |
| `service "ollama" is not running` | It belongs to the `llm` profile: use the script (which passes both profiles) or add `--profile llm`. |
| api log: `llm 404` / `model not found` | `pull-llm` not run, or `LLM_MODEL` in `.env` differs from what was pulled (`docker compose --profile llm exec ollama ollama list`). |
| api log: connection refused to the LLM | A host LLM bound to `127.0.0.1`. Use `host.docker.internal` in `LLM_BASE_URL` and make the server listen on `0.0.0.0`. |
| Port 3020 already in use | Change `WEB_PORT` in `.env`. Keep `API_PORT=3021`. |
| Home page says *api offline* right after `up` | Wait a minute; `docker compose ps` should show api `running`. Otherwise `logs api`. |
| First song never arrives, `logs yue2` shows downloads | Weights downloading (12 GB). Wait, or run `weights` once and retry. |
| `wsl --install` fails / no virtualization | Enable VT-x / AMD-V (SVM) in the BIOS/UEFI; on Windows Home, WSL 2 still works without Hyper-V. |
| PowerShell `curl` shows an HTML/objects mess | PowerShell's `curl` is `Invoke-WebRequest`. Use `curl.exe http://localhost:3021/healthz`. |

Plain `docker compose` equivalents of the script, run from the repo folder:

```powershell
$p = "--profile","gpu","--profile","llm"
docker compose $p up -d --build
docker compose $p exec ollama ollama pull qwen2.5:7b-instruct
docker compose $p ps
docker compose $p logs -f --tail 100 api
docker compose $p down
curl.exe http://localhost:3021/healthz
```
