import { TranscriptDisplay } from "./transcript";

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
  const btnStart = document.getElementById("btn-start") as HTMLButtonElement;
  const btnStop = document.getElementById("btn-stop") as HTMLButtonElement;
  const statusEl = document.getElementById("status") as HTMLSpanElement;
  const transcriptEl = document.getElementById("transcript") as HTMLElement;

  const display = new TranscriptDisplay(transcriptEl);

  btnStart.addEventListener("click", async () => {
    try {
      btnStart.disabled = true;
      statusEl.textContent = "starting…";
      await invoke("start_meeting");
      btnStop.disabled = false;
      statusEl.textContent = "recording";
    } catch (err) {
      statusEl.textContent = `error: ${err}`;
      btnStart.disabled = false;
    }
  });

  btnStop.addEventListener("click", async () => {
    try {
      btnStop.disabled = true;
      statusEl.textContent = "stopping…";
      const transcript = (await invoke("stop_meeting")) as string;
      statusEl.textContent = "idle";
      btnStart.disabled = false;
      if (transcript) {
        display.setFinalTranscript(transcript);
      }
    } catch (err) {
      statusEl.textContent = `error: ${err}`;
      btnStop.disabled = false;
    }
  });

  // Listen for real-time transcript updates from the Tauri backend.
  listen("transcript-update", (event) => {
    const payload = event.payload as { text: string; replace_chars: number };
    display.applyUpdate(payload.text, payload.replace_chars);
  });
}
