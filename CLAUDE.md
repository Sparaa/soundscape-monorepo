# Soundscape monorepo — Claude Code instructions

@AGENTS.md

The file above is the complete brief: facts, the numbered install procedure with checks, the diagnosis playbook and
the do/don't rules. Follow it. In particular: everything runs in Docker (never install Python/Node/CUDA on the
host), verify with the listed commands before naming a cause, never `docker compose down -v`, never touch the
version pins in `sidecars/*`.
