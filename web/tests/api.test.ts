import { describe, expect, it } from "vitest";
import { sidecarLine } from "@/lib/api";

describe("sidecarLine", () => {
  it("summarises health", () => {
    expect(sidecarLine("yue2", { ok: true, url: "u", gpu: "RTX 4090", loaded: false })).toBe("yue2 · RTX 4090 · idle");
    expect(sidecarLine("clipgrab", { ok: false, url: "u", error: "refused" })).toBe("clipgrab · offline");
  });
});

import { deleteSongPrompt, deleteStationPrompt, freshPrompt } from "@/lib/api";

describe("delete prompts", () => {
  it("say what goes and count the songs", () => {
    expect(deleteSongPrompt({ title: "Night Drive", id: "x" })).toBe("Delete “Night Drive” from disk? It leaves every playlist too.");
    expect(deleteSongPrompt({ title: null, id: "abc123" })).toContain("“abc123”");
    expect(deleteStationPrompt({ name: "Rap House", songs: 1 })).toContain("its 1 song are");
    expect(deleteStationPrompt({ name: "Rap House", songs: 17 })).toMatch(/^Delete station “Rap House”\? .*all 17 of its songs/);
    expect(freshPrompt({ name: "Rap House", songs: 0 })).toContain("all 0 of its songs");
    expect(freshPrompt({ name: "Rap House", songs: 3 })).toContain("new list from the same seeds");
  });
});
