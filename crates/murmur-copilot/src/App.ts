import { TranscriptDisplay, TranscriptUpdate } from "./transcript";

declare global {
  interface Window {
    __TAURI__?: {
      core: {
        invoke: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;
      };
      event: {
        listen: (
          event: string,
          handler: (event: { payload: unknown }) => void
        ) => Promise<() => void>;
      };
    };
  }
}

function invoke(cmd: string, args?: Record<string, unknown>): Promise<unknown> {
  return window.__TAURI__!.core.invoke(cmd, args);
}

function listen(
  event: string,
  handler: (event: { payload: unknown }) => void
): Promise<() => void> {
  return window.__TAURI__!.event.listen(event, handler);
}

const FADE_TIMEOUT_MS = 5000;
const SLEEP_TIMEOUT_MS = 10000;

export function initApp(): void {
  const statusIconEl = document.getElementById("status-icon") as HTMLSpanElement;
  const statusEl = document.getElementById("status") as HTMLSpanElement;
  const transcriptEl = document.getElementById("transcript") as HTMLElement;
  const appEl = document.getElementById("app") as HTMLElement;

  const display = new TranscriptDisplay(transcriptEl);

  let fadeTimer: ReturnType<typeof setTimeout> | null = null;
  let sleepTimer: ReturnType<typeof setTimeout> | null = null;
  let isRecording = false;

  function setRecording() {
    isRecording = true;
    statusIconEl.textContent = "🎙";
    statusIconEl.classList.add("recording");
    statusIconEl.title = "Recording";
    statusEl.textContent = "recording";
    resetTimers();
  }

  function setIdle() {
    isRecording = false;
    statusIconEl.textContent = "⏸";
    statusIconEl.classList.remove("recording");
    statusIconEl.title = "Idle";
    statusEl.textContent = "idle";
  }

  function resetTimers() {
    appEl.classList.remove("faded", "asleep");

    if (fadeTimer) clearTimeout(fadeTimer);
    if (sleepTimer) clearTimeout(sleepTimer);

    fadeTimer = setTimeout(() => {
      appEl.classList.add("faded");
    }, FADE_TIMEOUT_MS);

    sleepTimer = setTimeout(async () => {
      appEl.classList.add("asleep");
      if (isRecording) {
        try {
          await invoke("stop_meeting");
        } catch { /* already stopped */ }
      }
      setIdle();
    }, SLEEP_TIMEOUT_MS);
  }

  // ── Real-time transcript updates ──────────────────────────────────
  listen("transcript-update", (event) => {
    const payload = event.payload as TranscriptUpdate;
    display.applyUpdate(payload);
    resetTimers();
  });

  // ── Wake/sleep voice events ───────────────────────────────────────
  listen("copilot-wake", async () => {
    display.clear();
    try {
      await invoke("start_meeting");
      setRecording();
    } catch { /* meeting may already be running */ }
  });

  listen("copilot-sleep", () => {
    setIdle();
  });
}
