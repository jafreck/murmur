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

export function initApp(): void {
  const statusIcon = document.getElementById("status-icon") as HTMLSpanElement;
  const statusEl = document.getElementById("status") as HTMLSpanElement;
  const transcriptEl = document.getElementById("transcript") as HTMLElement;
  const appEl = document.getElementById("app") as HTMLElement;

  const display = new TranscriptDisplay(transcriptEl);

  // ── Fade & sleep on silence ────────────────────────────────────────
  const FADE_TIMEOUT_MS = 5000;
  const SLEEP_TIMEOUT_MS = 10000;
  let fadeTimer: ReturnType<typeof setTimeout> | null = null;
  let sleepTimer: ReturnType<typeof setTimeout> | null = null;
  let isListening = false;

  function setRecordingState(active: boolean) {
    isListening = active;
    if (active) {
      statusIcon.textContent = "🎙";
      statusIcon.classList.add("recording");
      statusEl.textContent = "listening";
      appEl.classList.remove("faded", "asleep");
      resetSilenceTimers();
    } else {
      statusIcon.textContent = "⏸";
      statusIcon.classList.remove("recording");
      statusEl.textContent = "idle";
      if (fadeTimer) clearTimeout(fadeTimer);
      if (sleepTimer) clearTimeout(sleepTimer);
    }
  }

  function resetSilenceTimers() {
    appEl.classList.remove("faded", "asleep");

    if (fadeTimer) clearTimeout(fadeTimer);
    if (sleepTimer) clearTimeout(sleepTimer);

    if (!isListening) return;

    fadeTimer = setTimeout(() => {
      appEl.classList.add("faded");
    }, FADE_TIMEOUT_MS);

    sleepTimer = setTimeout(async () => {
      appEl.classList.add("asleep");
      try {
        await invoke("stop_meeting");
      } catch {
        // Already stopped
      }
      setRecordingState(false);
    }, SLEEP_TIMEOUT_MS);
  }

  // ── Wake/sleep events from backend ─────────────────────────────────
  listen("copilot-wake", async () => {
    try {
      await invoke("start_meeting");
      display.clear();
      setRecordingState(true);
    } catch (err) {
      statusEl.textContent = `error: ${err}`;
    }
  });

  listen("copilot-sleep", () => {
    setRecordingState(false);
  });

  // ── Real-time transcript updates ──────────────────────────────────
  listen("transcript-update", (event) => {
    const payload = event.payload as TranscriptUpdate;
    display.applyUpdate(payload);
    resetSilenceTimers();
  });
}
