/** Radio player core: WebAudio graph (two decks with gain crossfade) that pulls the next song from the API when one
 * ends. The AnalyserNode is exposed for the visualizer (Phase 3). Pure helpers are exported for tests. */
import type { Song } from "./api";

export const CROSSFADE_S = 3;
/** How long a crossfade waits for the incoming deck's play() before switching anyway (a promise that never settles
 * used to hang the pull forever: no more automatic advances, only Skip worked). */
export const PLAY_TIMEOUT_MS = 5000;
/** A playhead that has not moved for this long while we are meant to be playing counts as a dead song → next one. */
export const STUCK_MS = 15000;

/** Seconds into a track at which the next one should start (crossfade), never before 1 s. */
export function nextStartAt(durationS: number | null | undefined, crossfadeS = CROSSFADE_S): number {
  if (!durationS || !isFinite(durationS)) return Infinity;
  return Math.max(1, durationS - crossfadeS);
}

/** Equal-power crossfade gains at t in [0,1]. */
export function crossfadeGains(t: number): { out: number; in: number } {
  const x = Math.min(1, Math.max(0, t));
  return { out: Math.cos(x * Math.PI / 2), in: Math.sin(x * Math.PI / 2) };
}

/** `gen` counts loads: a fade-out cleanup scheduled for one load must not wipe the deck after a later load reused it. */
export interface Deck { el: HTMLAudioElement; gain: GainNode; src: MediaElementAudioSourceNode; song: Song | null; gen: number }

export class RadioPlayer {
  ctx: AudioContext;
  analyser: AnalyserNode;
  master: GainNode;
  decks: [Deck, Deck];
  active = 0;
  onSongChange: (s: Song | null) => void = () => {};
  onNeedNext: () => Promise<Song | null> = async () => null;
  /** A failed next-song fetch (API restarting, network blip): the player retries by itself; the UI may show it. */
  onNextError: (e: unknown) => void = () => {};
  private timer: number | null = null;
  private fetching = false;
  private retryAt = 0;      // performance.now() before which we don't ask again (nothing cued yet, or the last ask failed)
  private playing = false;  // between start/resume/crossfadeTo and pause/stop: the radio is expected to keep going by itself
  private lastT = -1;       // watchdog: last playhead seen and when it last moved
  private lastMove = 0;

  constructor(audioUrl: (id: string) => string) {
    const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.ctx = new Ctx();
    this.master = this.ctx.createGain();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.8;
    this.master.connect(this.analyser).connect(this.ctx.destination);
    const mk = (): Deck => {
      const el = new Audio();
      el.crossOrigin = "anonymous";
      el.preload = "auto";
      const gain = this.ctx.createGain();
      gain.gain.value = 0;
      const src = this.ctx.createMediaElementSource(el);
      src.connect(gain).connect(this.master);
      return { el, gain, src, song: null, gen: 0 };
    };
    this.decks = [mk(), mk()];
    this.audioUrl = audioUrl;
  }
  private audioUrl: (id: string) => string;

  get current(): Deck { return this.decks[this.active]; }
  get standby(): Deck { return this.decks[1 - this.active]; }

  async start(first: Song): Promise<void> {
    await this.ctx.resume();
    this.load(this.current, first);
    this.current.gain.gain.cancelScheduledValues(this.ctx.currentTime);
    this.current.gain.gain.value = 1;
    this.playing = true;
    this.ensureTicking();                 // before play(): a rejected play() must not leave the radio without its tick
    await this.current.el.play();
    this.onSongChange(first);
  }

  private load(deck: Deck, song: Song): void {
    deck.gen++;
    deck.song = song;
    deck.el.src = this.audioUrl(song.id);
    deck.el.load();
  }

  /** Exactly one tick chain, whatever sequence of start / Stop / Play / crossfade got us here. */
  private ensureTicking(): void {
    if (this.timer === null) this.tick();
  }

  /** Called every 250 ms while playing: near the end (or after it), fetch + start the next song on the standby deck and
   * crossfade. A deck that lost its song while we are meant to be playing (a stale cleanup, a media error) counts as
   * "over" too, so the radio recovers by itself instead of sitting silent with songs cued. */
  private tick = (): void => {
    try {
      const cur = this.current;
      const dur = isFinite(cur.el.duration) && cur.el.duration > 0 ? cur.el.duration : cur.song?.seconds ?? null;
      const now = performance.now();
      const t = cur.el.currentTime;
      if (t !== this.lastT || !this.playing || this.fetching) { this.lastT = t; this.lastMove = now; }
      const stuck = this.playing && !!cur.song && !cur.el.paused && now - this.lastMove > STUCK_MS;   // play() never started, a dead stream
      const over = !cur.song || cur.el.ended || stuck;
      const due = this.playing && (over || t >= nextStartAt(dur));
      if (stuck) this.onNextError(new Error(`"${cur.song?.title ?? cur.song?.id}" stopped moving — skipping ahead`));
      if (due && !this.fetching && !this.standby.song && performance.now() >= this.retryAt) void this.pull(over ? 0.2 : CROSSFADE_S);
    } catch (e) {
      this.onNextError(e);                // never let one bad read kill the chain
    }
    this.timer = window.setTimeout(this.tick, 250);
  };

  /** One attempt at the next song. `fetching` ALWAYS clears: a rejected fetch (API restarting mid-deploy, a blip) used to
   * leave it stuck and the radio silent after the song ended until the user pressed Skip. Nothing cued / failed → ask
   * again in a second (the agent is still composing), not every 250 ms. */
  private async pull(crossfadeS: number, get: () => Promise<Song | null> = this.onNeedNext): Promise<void> {
    this.fetching = true;
    try {
      const next = await get();
      if (next) await this.crossfadeTo(next, crossfadeS);
      else this.retryAt = performance.now() + 1000;
    } catch (e) {
      this.retryAt = performance.now() + 1000;
      this.onNextError(e);
    } finally {
      this.fetching = false;
    }
  }

  async crossfadeTo(next: Song, seconds = CROSSFADE_S): Promise<void> {
    const out = this.current, inn = this.standby;
    this.load(inn, next);
    const outGen = out.gen, innGen = inn.gen;
    this.playing = true;
    this.ensureTicking();
    // Wait for the incoming deck to start, but not forever: a play() that never settles must not hang the pull.
    const started = await Promise.race([inn.el.play().then(() => true, () => false),
                                        new Promise<null>((r) => window.setTimeout(() => r(null), PLAY_TIMEOUT_MS))]);
    if (inn.gen !== innGen) return;       // a later crossfade already reloaded this deck: it owns the swap and the announcement
    if (started === null) this.onNextError(new Error(`"${next.title ?? next.id}" did not start within ${PLAY_TIMEOUT_MS / 1000} s — switching anyway`));
    this.lastT = -1; this.lastMove = performance.now();
    const t0 = this.ctx.currentTime;
    out.gain.gain.cancelScheduledValues(t0); inn.gain.gain.cancelScheduledValues(t0);
    out.gain.gain.setValueAtTime(out.gain.gain.value, t0); inn.gain.gain.setValueAtTime(0, t0);
    out.gain.gain.linearRampToValueAtTime(0, t0 + seconds); inn.gain.gain.linearRampToValueAtTime(1, t0 + seconds);
    this.active = 1 - this.active;
    this.onSongChange(next);
    // Release the faded-out deck — unless a second crossfade (Skip mid-fade, two playlist clicks in a row) reused it
    // meanwhile: wiping it then killed the song that had just started and left the radio silent until the next Skip.
    window.setTimeout(() => {
      if (out.gen !== outGen || out === this.current) return;
      out.el.pause(); out.el.removeAttribute("src"); out.el.load(); out.song = null;
    }, seconds * 1000 + 100);
  }

  /** Jump to whatever the radio has next. Ignored while an automatic pull is already in flight. */
  async skip(): Promise<void> {
    if (this.fetching) return;
    await this.pull(0.5);
  }

  /** Play a specific song now (a cued one the listener picked, a saved one) through the same guarded path as the
   * automatic pull. `get` may return null (the song is gone) → nothing changes. Returns false if a pull was in flight. */
  async playNow(get: () => Promise<Song | null>, seconds = 0.5): Promise<boolean> {
    if (this.fetching) return false;
    await this.pull(seconds, get);
    return true;
  }

  /** Stop = pause: the current song stays loaded so Play picks it up where it was (the server keeps a spare behind it). */
  pause(): void {
    this.playing = false;
    if (this.timer) window.clearTimeout(this.timer);
    this.timer = null;
    for (const d of this.decks) d.el.pause();
  }

  get paused(): boolean { return !!this.current.song && this.current.el.paused; }

  async resume(): Promise<boolean> {
    if (!this.current.song) return false;
    this.playing = true;
    this.ensureTicking();
    await this.ctx.resume();
    await this.current.el.play();
    return true;
  }

  stop(): void {
    this.playing = false;
    if (this.timer) window.clearTimeout(this.timer);
    this.timer = null;
    for (const d of this.decks) { d.gen++; d.el.pause(); d.el.removeAttribute("src"); d.el.load(); d.song = null; d.gain.gain.value = 0; }
    this.onSongChange(null);
  }

  position(): { t: number; d: number } {
    const el = this.current.el;
    return { t: el.currentTime || 0, d: (isFinite(el.duration) && el.duration) || this.current.song?.seconds || 0 };
  }
}
