# vidmakr-music

YuE2 song generation as a sidecar HTTP worker. `style` + `lyrics` in, one stereo
48 kHz FLAC out (mp3/m4a on request). Only `vidmakr-ai` talks to it; the browser
never does. See `PINNED.md` for versions, weights, and knobs, and
`docs/music-plan.md` for how it plugs into Imagine and the `/music` page.

```
POST /generate  {style, lyrics, cot?, seed?, cfg_scale?, id?}  -> {job_id, state, position, seed}
GET  /jobs/{id}                                              -> state/stage/progress/tokens/audio_seconds/truncated/timing
GET  /jobs/{id}/audio?format=flac|mp3|m4a                    -> bytes
GET  /jobs/{id}/score                                        -> ABC plan (cot=full|melody)
DELETE /jobs/{id}                                            -> cancel + drop scratch
POST /unload                                                 -> free the model now
GET  /healthz                                                -> loaded/busy/queue/gpu
```

Progress: plan and semantic stages count generated tokens against their caps
(4096 / 9000); synthesis and decoding are timed against the previous run.

Tests: `imagine test music` (fake pipeline, no GPU). Debug from the host:
`curl -s localhost:3015/healthz`.
