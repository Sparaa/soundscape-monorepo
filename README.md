# vidmakr-clipgrab

Standalone fetcher for **publicly reachable web video** (YouTube first; anything
[yt-dlp](https://github.com/yt-dlp/yt-dlp) supports) so vidmakr can use outside
footage as a **motion reference** (MiniMax H3 R2V) and as editor media.

- **Self-updating yt-dlp**: `entrypoint.sh` runs `pip install --upgrade yt-dlp`
  on every container start, then serves. YouTube breaks extractors every few
  weeks; the version baked into the image is only the offline fallback.
  `POST /update` does the same on a running container.
- **JS runtime**: Deno (yt-dlp's recommended runtime for YouTube's n-param /
  signature challenges) is copied from the official binary image.
- **No content on host disk**: each request downloads into a per-request dir
  under a tmpfs `/scratch`, removed once the bytes are streamed back.
- **Isolated**: no GPU, no model mounts, no published port. Only the `api`
  container reaches it (`http://clipgrab:3014`). Non-public hosts (loopback,
  RFC1918, link-local, dotless names) are refused before yt-dlp runs.
- **Scope**: public content only. No cookies / login / age-gate support by
  design. Live streams are refused. Whole pulls are capped at `MAX_CLIP_SEC`
  (default 600 s) — give a start/end window for anything longer.

## Routes

| Route | Purpose |
|---|---|
| `GET /healthz` | `{ok, yt_dlp, js_runtime}` |
| `GET /info?url=` | title / duration / uploader / thumbnail, no download |
| `POST /clip {url, start?, end?}` | `video/mp4` body; H.264+AAC preferred, `--download-sections` + `--force-keyframes-at-cuts` when a window is given |
| `POST /update` | upgrade yt-dlp in place |

Response headers on `/clip`: `X-Clipgrab-Title` (percent-encoded),
`X-Clipgrab-Source`, `X-Clipgrab-Duration`.

## In vidmakr

`api` → `POST /import/url` proxies here, lands the mp4 in the tmpfs media dir
and registers it like an upload. The web UI exposes it as **From URL** in the
editor's Asset Library and **+ From URL** on Imagine's R2V tab; by default the
import is also cut into 5 s reference clips (origin `refclip`) so Imagine,
the chat agent and Director see it.

## Tests

```
docker run --rm --name vidmakr-clipgrab-tests -v $PWD/clipgrab:/app -w /app vidmakr-clipgrab:latest \
  sh -c "pip install -q pytest && python -m pytest -q tests"
```
(`imagine test` runs this as the third suite.)
