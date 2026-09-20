"""YuE2 sidecar client: submit, poll with progress, fetch the FLAC and the planned score."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

import httpx

from . import config
from .abc import MusicError

log = logging.getLogger("soundscape.composer")
Progress = Callable[[dict[str, Any]], Awaitable[None] | None]


class Yue2:
    def __init__(self, base_url: str = config.YUE2_URL, client: Optional[httpx.AsyncClient] = None):
        self.base_url, self.client = base_url.rstrip("/"), client

    def _c(self) -> httpx.AsyncClient:
        return self.client or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    async def render(self, request: dict[str, Any], *, on_progress: Optional[Progress] = None, poll_s: float = 3.0,
                     timeout_s: float = 1800.0) -> tuple[bytes, dict[str, Any]]:
        """→ (flac bytes, job dict incl. abc/planned_seconds/audio_seconds)."""
        client = self._c()
        job_id: Optional[str] = None
        try:
            body = {k: v for k, v in request.items() if k not in ("cover_mode", "vocal_promoted")}
            body.setdefault("priority", "background")   # a radio filling its buffer never delays someone's click on a shared sidecar
            r = await client.post(f"{self.base_url}/generate", json=body)
            if r.status_code != 200:
                raise MusicError(f"yue2 {r.status_code}: {r.text[:300]}")
            job_id = r.json()["job_id"]
            waited = 0.0
            while True:
                s = (await client.get(f"{self.base_url}/jobs/{job_id}")).json()
                if on_progress:
                    res = on_progress(s)
                    if asyncio.iscoroutine(res):
                        await res
                if s.get("state") == "done":
                    break
                if s.get("state") in ("error", "cancelled"):
                    raise MusicError(f"render failed: {s.get('error')}")
                await asyncio.sleep(poll_s)
                waited += poll_s
                if waited > timeout_s:
                    raise MusicError("render timed out")
            audio = (await client.get(f"{self.base_url}/jobs/{job_id}/audio", timeout=300.0)).content
            score = await client.get(f"{self.base_url}/jobs/{job_id}/score")
            s["abc"] = score.text if score.status_code == 200 else s.get("abc")
            return audio, s
        except asyncio.CancelledError:       # the radio was reset (start fresh / station deleted): free the GPU too
            if job_id:
                try:
                    await client.delete(f"{self.base_url}/jobs/{job_id}", timeout=10.0)
                except Exception as e:
                    log.warning("cancel yue2 job %s: %s", job_id, e)
            raise
        finally:
            if self.client is None:
                await client.aclose()
