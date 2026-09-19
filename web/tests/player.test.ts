import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { crossfadeGains, nextStartAt, RadioPlayer, STUCK_MS } from "@/lib/player";
import { planLabel, type Song } from "@/lib/api";

describe("player helpers", () => {
  it("schedules the next track before the end", () => {
    expect(nextStartAt(200)).toBe(197);
    expect(nextStartAt(2)).toBe(1);
    expect(nextStartAt(null)).toBe(Infinity);
  });
  it("equal-power crossfade", () => {
    const mid = crossfadeGains(0.5);
    expect(mid.out).toBeCloseTo(mid.in, 5);
    expect(crossfadeGains(0)).toEqual({ out: 1, in: 0 });
    expect(crossfadeGains(1).out).toBeCloseTo(0, 5);
  });
  it("labels plans", () => {
    expect(planLabel({ mode: "faithful", explain: "cover of “X” with new words" } as never)).toBe("cover of “X” with new words");
    expect(planLabel({ mode: "hook", explain: "" } as never)).toBe("hook");
    expect(planLabel(null)).toBe("");
  });
});

// ---- RadioPlayer against stubbed <audio> + WebAudio (vitest runs in node): only the pulling logic is exercised.
class FakeAudio {
  src = ""; currentTime = 0; duration = NaN; ended = false; paused = true; crossOrigin = ""; preload = "";
  play() { this.paused = false; return Promise.resolve(); }
  pause() { this.paused = true; }
  load() { this.duration = NaN; this.currentTime = 0; this.ended = false; this.paused = true; }   // like the media load algorithm
  removeAttribute(n: string) { if (n === "src") this.src = ""; }
}
const param = () => ({ value: 0, cancelScheduledValues() {}, setValueAtTime() {}, linearRampToValueAtTime() {} });
const node = () => { const n = { gain: param(), connect: (_: unknown) => n, fftSize: 0, smoothingTimeConstant: 0 }; return n; };
class FakeCtx {
  currentTime = 0; destination = {};
  createGain() { return node(); }
  createAnalyser() { return node(); }
  createMediaElementSource() { return node(); }
  resume() { return Promise.resolve(); }
}
const song = (id: string, seconds: number): Song => ({ id, seconds } as unknown as Song);

describe("RadioPlayer pulls the next song", () => {
  let audios: FakeAudio[];
  beforeEach(() => {
    audios = [];
    vi.useFakeTimers();
    vi.stubGlobal("window", { AudioContext: FakeCtx, setTimeout, clearTimeout });
    vi.stubGlobal("Audio", class extends FakeAudio { constructor() { super(); audios.push(this); } });
    vi.stubGlobal("performance", { now: () => Date.now() });
  });
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

  async function playToTheEnd(p: RadioPlayer) {
    await p.start(song("a", 100));
    const a = audios[0];
    a.duration = 100; a.currentTime = 98;                      // inside the crossfade window
    await vi.advanceTimersByTimeAsync(300);                    // one tick → first ask
    a.currentTime = 100; a.ended = true; a.paused = true;      // the song ran out while the ask was failing
  }

  it("keeps asking after a failed fetch (API restart mid-deploy) instead of stalling until Skip", async () => {
    const p = new RadioPlayer((id) => `/audio/${id}`);
    const errors: unknown[] = [];
    p.onNextError = (e) => errors.push(e);
    const asks = vi.fn<() => Promise<Song | null>>()
      .mockRejectedValueOnce(new Error("Failed to fetch"))
      .mockResolvedValue(song("b", 120));
    p.onNeedNext = asks;
    await playToTheEnd(p);
    expect(asks).toHaveBeenCalledTimes(1);
    expect(errors).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1500);                   // retry after ~1 s, not never
    expect(asks).toHaveBeenCalledTimes(2);
    expect(p.current.song?.id).toBe("b");
    expect(audios.find((x) => x.src === "/audio/b")?.paused).toBe(false);
    p.stop();
  });

  it("waits a second between asks while nothing is cued, then crossfades when a song appears", async () => {
    const p = new RadioPlayer((id) => `/audio/${id}`);
    const asks = vi.fn<() => Promise<Song | null>>().mockResolvedValueOnce(null).mockResolvedValueOnce(null).mockResolvedValue(song("b", 120));
    p.onNeedNext = asks;
    await playToTheEnd(p);
    expect(asks).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(500);
    expect(asks).toHaveBeenCalledTimes(1);                     // not hammering the API every 250 ms
    await vi.advanceTimersByTimeAsync(2100);
    expect(asks).toHaveBeenCalledTimes(3);
    expect(p.current.song?.id).toBe("b");
    p.stop();
  });
});

describe("RadioPlayer survives a second crossfade inside the first one's cleanup window", () => {
  let audios: FakeAudio[];
  beforeEach(() => {
    audios = [];
    vi.useFakeTimers();
    vi.stubGlobal("window", { AudioContext: FakeCtx, setTimeout, clearTimeout });
    vi.stubGlobal("Audio", class extends FakeAudio { constructor() { super(); audios.push(this); } });
    vi.stubGlobal("performance", { now: () => Date.now() });
  });
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });
  const deckOf = (id: string) => audios.find((x) => x.src === `/audio/${id}`);

  it("two playlist clicks 1 s apart keep the second song playing and the radio advancing", async () => {
    const p = new RadioPlayer((id) => `/audio/${id}`);
    const asks = vi.fn<() => Promise<Song | null>>().mockResolvedValue(song("d", 100));
    p.onNeedNext = asks;
    await p.start(song("a", 100));
    await p.crossfadeTo(song("b", 100), 3);                    // click song b in the playlist
    await vi.advanceTimersByTimeAsync(1000);
    await p.crossfadeTo(song("c", 100), 0.4);                  // click song c a second later: lands on a's deck
    await vi.advanceTimersByTimeAsync(4000);                   // a's 3.1 s cleanup fires meanwhile
    expect(p.current.song?.id).toBe("c");                      // was: null — the cleanup wiped the deck now playing c
    const c = deckOf("c")!;
    expect(c.src).toBe("/audio/c");
    expect(c.paused).toBe(false);
    c.duration = 100; c.currentTime = 98;                      // c reaches its crossfade point → the radio must ask
    await vi.advanceTimersByTimeAsync(600);
    expect(asks).toHaveBeenCalledTimes(1);
    expect(p.current.song?.id).toBe("d");
    p.stop();
  });

  it("Skip during a natural crossfade keeps the skipped-to song", async () => {
    const p = new RadioPlayer((id) => `/audio/${id}`);
    const asks = vi.fn<() => Promise<Song | null>>().mockResolvedValueOnce(song("b", 100)).mockResolvedValue(song("c", 100));
    p.onNeedNext = asks;
    await p.start(song("a", 100));
    const a = audios[0];
    a.duration = 100; a.currentTime = 98;                      // natural crossfade a → b (3 s)
    await vi.advanceTimersByTimeAsync(300);
    expect(p.current.song?.id).toBe("b");
    await vi.advanceTimersByTimeAsync(1000);
    await p.skip();                                            // user skips b 1 s into the fade: c lands on a's deck
    expect(p.current.song?.id).toBe("c");
    await vi.advanceTimersByTimeAsync(4000);                   // a's cleanup fires
    expect(p.current.song?.id).toBe("c");
    expect(deckOf("c")!.src).toBe("/audio/c");
    expect(deckOf("c")!.paused).toBe(false);
    p.stop();
  });

  it("recovers by itself if the playing deck loses its song (instead of sitting silent with songs cued)", async () => {
    const p = new RadioPlayer((id) => `/audio/${id}`);
    const asks = vi.fn<() => Promise<Song | null>>().mockResolvedValue(song("b", 100));
    p.onNeedNext = asks;
    await p.start(song("a", 100));
    Object.assign(p.current, { song: null }); p.current.el.removeAttribute("src"); p.current.el.load();   // whatever wiped it
    await vi.advanceTimersByTimeAsync(600);
    expect(asks).toHaveBeenCalledTimes(1);
    expect(p.current.song?.id).toBe("b");
    p.pause();
    Object.assign(p.current, { song: null });                  // paused: nothing is expected to play, so no pull
    await vi.advanceTimersByTimeAsync(2000);
    expect(asks).toHaveBeenCalledTimes(1);
    p.stop();
  });

  it("Skip while an automatic pull is in flight does not fetch twice; Stop → Play keeps a single tick chain", async () => {
    const p = new RadioPlayer((id) => `/audio/${id}`);
    let release: (s: Song) => void = () => {};
    const asks = vi.fn<() => Promise<Song | null>>().mockImplementationOnce(() => new Promise((r) => { release = r; })).mockResolvedValue(song("c", 100));
    p.onNeedNext = asks;
    await p.start(song("a", 100));
    audios[0].duration = 100; audios[0].currentTime = 98;
    await vi.advanceTimersByTimeAsync(300);                    // tick asks; the API is slow
    await p.skip();                                            // user hammers Skip meanwhile
    expect(asks).toHaveBeenCalledTimes(1);
    release(song("b", 100));
    await vi.advanceTimersByTimeAsync(10);
    expect(p.current.song?.id).toBe("b");
    await vi.advanceTimersByTimeAsync(3500);                   // a's fade-out finishes and its deck is released
    p.pause(); await p.resume(); p.pause(); await p.resume();  // Stop / Play twice
    deckOf("b")!.duration = 100; deckOf("b")!.currentTime = 98;
    await vi.advanceTimersByTimeAsync(300);
    expect(asks).toHaveBeenCalledTimes(2);                     // one chain, one ask
    p.stop();
  });

  it("playNow swaps to a chosen cued song and keeps advancing from it", async () => {
    const p = new RadioPlayer((id) => `/audio/${id}`);
    const asks = vi.fn<() => Promise<Song | null>>().mockResolvedValue(song("z", 100));
    p.onNeedNext = asks;
    await p.start(song("a", 100));
    expect(await p.playNow(async () => song("k", 100))).toBe(true);
    expect(p.current.song?.id).toBe("k");
    expect(deckOf("k")!.paused).toBe(false);
    await vi.advanceTimersByTimeAsync(1000);
    deckOf("k")!.duration = 100; deckOf("k")!.currentTime = 99;
    await vi.advanceTimersByTimeAsync(300);
    expect(asks).toHaveBeenCalledTimes(1);
    expect(p.current.song?.id).toBe("z");
    p.stop();
  });

  it("a play() that never settles no longer hangs the pull: the crossfade proceeds after the timeout", async () => {
    const p = new RadioPlayer((id) => `/audio/${id}`);
    const errors: unknown[] = [];
    p.onNextError = (e) => errors.push(e);
    const asks = vi.fn<() => Promise<Song | null>>().mockResolvedValueOnce(song("b", 100)).mockResolvedValue(song("c", 100));
    p.onNeedNext = asks;
    await p.start(song("a", 100));
    audios[1].play = function () { this.paused = false; return new Promise<void>(() => undefined); };   // b's deck: pending forever
    audios[0].duration = 100; audios[0].currentTime = 98;
    await vi.advanceTimersByTimeAsync(300);
    expect(p.current.song?.id).toBe("a");                      // still waiting on b
    await vi.advanceTimersByTimeAsync(5200);
    expect(p.current.song?.id).toBe("b");                      // switched anyway, pull released
    expect(errors).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(3500);                   // a's deck released
    expect(audios[0].src).toBe("");
    expect(p.standby.song).toBeNull();
    await vi.advanceTimersByTimeAsync(STUCK_MS + 500);         // b never moves → watchdog skips ahead to c
    expect(asks).toHaveBeenCalledTimes(2);
    expect(p.current.song?.id).toBe("c");
    p.stop();
  });

  it("a playhead that stops moving mid-song skips ahead; a moving one does not", async () => {
    const p = new RadioPlayer((id) => `/audio/${id}`);
    const asks = vi.fn<() => Promise<Song | null>>().mockResolvedValue(song("b", 100));
    p.onNeedNext = asks;
    await p.start(song("a", 100));
    const a = audios[0]; a.duration = 100;
    for (let i = 0; i < 80; i++) { a.currentTime = i * 0.25; await vi.advanceTimersByTimeAsync(250); }   // 20 s of normal playback
    expect(asks).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(STUCK_MS - 1000);        // frozen, but not yet long enough
    expect(asks).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1500);
    expect(asks).toHaveBeenCalledTimes(1);
    expect(p.current.song?.id).toBe("b");
    p.stop();
  });
});
