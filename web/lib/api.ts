export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3021";

export interface SidecarHealth { ok: boolean; url: string; gpu?: string | null; loaded?: boolean | null; error?: string }
export interface Health { ok: boolean; sidecars: Record<"yue2" | "sheetsage" | "clipgrab", SidecarHealth>; llm: { base_url: string; model: string }; library: string }

export interface Tag { label: string; weight: number }
export interface Profile {
  seeds: number;
  tags: Record<string, Tag[]>;
  bpm: { low: number; high: number; center: number } | null;
  keys: string[];
  sections: string[];
  phrases: Record<string, number[]>;
  instrumental: boolean;
  language: string;
  style: string;
  seconds: number | null;
}
export interface Seed {
  id: string; title: string; source: string | null; seconds: number | null; created: number;
  key: string | null; bpm: number | null; sections: string[] | null; style_guess: string | null;
  promoted_sections: string[] | null; warnings: string[] | null; has_score: boolean;
}
export interface Station { id: string; name: string; created: number; profile: Profile | null; settings: { language?: string; covers?: number; blurb?: string; themes?: string[]; playlist_id?: string } | null; seeds: Seed[] }

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`);
  return r.json();
}

export async function getHealth(): Promise<Health> {
  return j(await fetch(`${API_URL}/healthz`, { cache: "no-store" }));
}
export async function listStations(): Promise<Station[]> {
  return j(await fetch(`${API_URL}/stations`, { cache: "no-store" }));
}
export async function getStation(id: string): Promise<Station> {
  return j(await fetch(`${API_URL}/stations/${id}`, { cache: "no-store" }));
}
export async function createStation(name: string, language = "English"): Promise<Station> {
  return j(await fetch(`${API_URL}/stations`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, language }) }));
}
export async function deleteStation(id: string): Promise<void> {
  await j(await fetch(`${API_URL}/stations/${id}`, { method: "DELETE" }));
}
export async function addSeedFile(stationId: string, file: File): Promise<Station> {
  const fd = new FormData();
  fd.append("file", file, file.name);
  return j(await fetch(`${API_URL}/stations/${stationId}/seeds`, { method: "POST", body: fd }));
}
export async function addSeedUrl(stationId: string, url: string): Promise<Station> {
  const fd = new FormData();
  fd.append("url", url);
  return j(await fetch(`${API_URL}/stations/${stationId}/seeds`, { method: "POST", body: fd }));
}
export async function deleteSeed(seedId: string): Promise<Station> {
  return j(await fetch(`${API_URL}/seeds/${seedId}`, { method: "DELETE" }));
}
export function seedAudioUrl(seedId: string): string {
  return `${API_URL}/seeds/${seedId}/audio`;
}

/** One line per sidecar for the status strip: "yue2 · RTX 4090 · idle". */
export function sidecarLine(name: string, h: SidecarHealth): string {
  if (!h.ok) return `${name} · offline`;
  return `${name} · ${h.gpu ?? "gpu?"} · ${h.loaded ? "loaded" : "idle"}`;
}

export interface Plan { mode: "inspired" | "faithful" | "reinterpret" | "hook"; seed_id: string | null; seed_title: string | null; theme: string; bpm: number; key: string | null; mood: string[]; duration_s: number; explain: string; cot: string; created: number }
export interface Gate { ok: boolean; reasons: string[]; seconds: number | null; mean_db: number | null; max_db: number | null; similarity: number | null }
export interface Song { id: string; station_id: string | null; title: string | null; vote: number; tags: Record<string, { label: string; p: number }[]> | null; style: string | null; lyrics: string | null; abc: string | null; plan: Plan | null; seconds: number | null; path: string; liked: boolean; saved: boolean; created: number; status: string; explain: string | null; gate: Gate | null; played: number | null }
export interface Rendering { plan: Plan; stage: string; progress: number; seconds: number; attempt: number }
export interface RadioStatus { state: "stopped" | "warming" | "playing" | "stopping"; now_playing: Song | null; ready: Song[]; rendering: Rendering | null; buffer_target: number; recent: Song[]; events: { t: number; kind: string; [k: string]: unknown }[] }

export async function radioPlay(id: string): Promise<RadioStatus> { return j(await fetch(`${API_URL}/stations/${id}/play`, { method: "POST" })); }
export async function radioStop(id: string): Promise<RadioStatus> { return j(await fetch(`${API_URL}/stations/${id}/stop`, { method: "POST" })); }
/** Pop the next cued song — or a specific cued one (`songId`) the listener clicked in Up next. */
export async function radioNext(id: string, songId?: string): Promise<{ song: Song | null; status: RadioStatus }> {
  return j(await fetch(`${API_URL}/stations/${id}/next${songId ? `?song_id=${encodeURIComponent(songId)}` : ""}`, { method: "POST" }));
}
export async function radioStatus(id: string): Promise<RadioStatus> { return j(await fetch(`${API_URL}/stations/${id}/radio`, { cache: "no-store" })); }
export async function stationSongs(id: string): Promise<Song[]> { return j(await fetch(`${API_URL}/stations/${id}/songs`, { cache: "no-store" })); }
export async function patchSong(id: string, flags: { saved?: boolean; liked?: boolean; vote?: -1 | 0 | 1; title?: string }): Promise<Song> {
  return j(await fetch(`${API_URL}/songs/${id}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify(flags) }));
}
export async function patchSettings(id: string, s: { covers?: number; blurb?: string }): Promise<Station> {
  return j(await fetch(`${API_URL}/stations/${id}/settings`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify(s) }));
}
export function songAudioUrl(id: string): string { return `${API_URL}/songs/${id}/audio`; }

/** "cover of “X” with new words" etc., or the mode when the plan has no text. */
export function planLabel(p: Plan | null | undefined): string {
  if (!p) return "";
  return p.explain || { inspired: "new song in the station's sound", faithful: "cover", reinterpret: "reinterpretation", hook: "hook" }[p.mode];
}

export interface Playlist { id: string; name: string; created: number; items: Song[]; seconds: number }

export async function librarySongs(opts: { saved?: boolean; liked?: boolean } = {}): Promise<Song[]> {
  const q = new URLSearchParams(); if (opts.saved) q.set("saved", "true"); if (opts.liked) q.set("liked", "true");
  return j(await fetch(`${API_URL}/library${q.toString() ? `?${q}` : ""}`, { cache: "no-store" }));
}
export async function importSong(file: File, title?: string): Promise<Song> {
  const fd = new FormData(); fd.append("file", file, file.name); if (title) fd.append("title", title);
  return j(await fetch(`${API_URL}/library/import`, { method: "POST", body: fd }));
}
export async function deleteSong(id: string): Promise<void> { await j(await fetch(`${API_URL}/songs/${id}`, { method: "DELETE" })); }
export async function voteSong(id: string, vote: -1 | 0 | 1): Promise<Song> { return patchSong(id, { vote }); }
export async function listPlaylists(): Promise<Playlist[]> { return j(await fetch(`${API_URL}/playlists`, { cache: "no-store" })); }
export async function getPlaylist(id: string): Promise<Playlist> { return j(await fetch(`${API_URL}/playlists/${id}`, { cache: "no-store" })); }
export async function createPlaylist(name: string): Promise<Playlist> {
  return j(await fetch(`${API_URL}/playlists`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name }) }));
}
export async function deletePlaylist(id: string): Promise<void> { await j(await fetch(`${API_URL}/playlists/${id}`, { method: "DELETE" })); }
export async function addToPlaylist(id: string, songId: string, position?: number): Promise<Playlist> {
  return j(await fetch(`${API_URL}/playlists/${id}/items`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ song_id: songId, position }) }));
}
export async function removeFromPlaylist(id: string, songId: string): Promise<Playlist> { return j(await fetch(`${API_URL}/playlists/${id}/items/${songId}`, { method: "DELETE" })); }
export async function reorderPlaylist(id: string, order: string[]): Promise<Playlist> {
  return j(await fetch(`${API_URL}/playlists/${id}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ order }) }));
}
export async function stationPlaylist(stationId: string): Promise<Playlist> { return j(await fetch(`${API_URL}/stations/${stationId}/playlist`, { cache: "no-store" })); }
export function playlistExportUrl(id: string): string { return `${API_URL}/playlists/${id}/export.zip`; }

/** Move an item within an order (pure). */
export function moveItem<T>(order: T[], from: number, to: number): T[] {
  if (from < 0 || from >= order.length || to < 0 || to >= order.length || from === to) return order;
  const out = order.slice(); const [x] = out.splice(from, 1); out.splice(to, 0, x); return out;
}
