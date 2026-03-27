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

interface AudioDevice {
  name: string;
  is_loopback_hint: boolean;
}

interface LlmStatus {
  available: boolean;
  model: string | null;
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
  const btnStealth = document.getElementById("btn-stealth") as HTMLButtonElement;
  const btnSuggest = document.getElementById("btn-suggest") as HTMLButtonElement;
  const statusEl = document.getElementById("status") as HTMLSpanElement;
  const llmStatusEl = document.getElementById("llm-status") as HTMLSpanElement;
  const transcriptEl = document.getElementById("transcript") as HTMLElement;
  const deviceSelect = document.getElementById("device-select") as HTMLSelectElement;
  const suggestionPanel = document.getElementById("suggestion-panel") as HTMLElement;
  const suggestionContent = document.getElementById("suggestion-content") as HTMLElement;
  const summaryPanel = document.getElementById("summary-panel") as HTMLElement;
  const summaryContent = document.getElementById("summary-content") as HTMLElement;

  const display = new TranscriptDisplay(transcriptEl);

  // ── LLM status check ──────────────────────────────────────────────
  async function checkLlmStatus() {
    try {
      const status = (await invoke("get_llm_status")) as LlmStatus;
      if (status.available) {
        llmStatusEl.textContent = "🟢";
        llmStatusEl.title = `LLM connected: ${status.model}`;
      } else {
        llmStatusEl.textContent = "🔴";
        llmStatusEl.title = "LLM disconnected";
      }
    } catch {
      llmStatusEl.textContent = "🔴";
      llmStatusEl.title = "LLM disconnected";
    }
  }
  checkLlmStatus();

  // ── Populate audio device selector ─────────────────────────────────
  async function loadDevices() {
    try {
      const devices = (await invoke("list_audio_devices")) as AudioDevice[];
      deviceSelect.innerHTML = '<option value="">None (mic only)</option>';
      for (const d of devices) {
        const opt = document.createElement("option");
        opt.value = d.name;
        opt.textContent = d.name + (d.is_loopback_hint ? " ★" : "");
        deviceSelect.appendChild(opt);
      }
    } catch {
      // Device listing may fail in environments without audio
    }
  }
  loadDevices();

  deviceSelect.addEventListener("change", async () => {
    const value = deviceSelect.value || null;
    await invoke("set_system_audio_device", { deviceName: value });
  });

  // ── Meeting controls ───────────────────────────────────────────────
  btnStart.addEventListener("click", async () => {
    try {
      btnStart.disabled = true;
      statusEl.textContent = "starting…";
      await invoke("start_meeting");
      btnStop.disabled = false;
      statusEl.textContent = "recording";
      suggestionPanel.classList.remove("hidden");
      summaryPanel.classList.add("hidden");
      suggestionContent.textContent = "No suggestions yet.";
    } catch (err) {
      statusEl.textContent = `error: ${err}`;
      btnStart.disabled = false;
    }
  });

  btnStop.addEventListener("click", async () => {
    try {
      btnStop.disabled = true;
      statusEl.textContent = "stopping…";
      const entries = (await invoke("stop_meeting")) as { speaker: string; text: string }[];
      statusEl.textContent = "idle";
      btnStart.disabled = false;
      if (entries && entries.length > 0) {
        display.setFinalTranscript(entries);
      }
      suggestionPanel.classList.add("hidden");

      // Auto-generate summary when meeting ends
      try {
        const summary = (await invoke("generate_summary")) as string | null;
        if (summary) {
          summaryContent.textContent = summary;
          summaryPanel.classList.remove("hidden");
        }
      } catch {
        // LLM may not be available — that's fine
      }
    } catch (err) {
      statusEl.textContent = `error: ${err}`;
      btnStop.disabled = false;
    }
  });

  // ── Suggestion button ──────────────────────────────────────────────
  btnSuggest.addEventListener("click", async () => {
    try {
      btnSuggest.disabled = true;
      suggestionContent.textContent = "Thinking…";
      const suggestion = (await invoke("get_suggestion")) as string | null;
      suggestionContent.textContent = suggestion ?? "No suggestions available.";
    } catch (err) {
      suggestionContent.textContent = `Error: ${err}`;
    } finally {
      btnSuggest.disabled = false;
    }
  });

  // ── Stealth toggle ─────────────────────────────────────────────────
  btnStealth.addEventListener("click", async () => {
    try {
      const enabled = (await invoke("toggle_stealth")) as boolean;
      btnStealth.textContent = enabled ? "👁 Stealth: ON" : "👁 Stealth: OFF";
      btnStealth.classList.toggle("active", enabled);
    } catch (err) {
      statusEl.textContent = `error: ${err}`;
    }
  });

  // ── Real-time transcript updates ──────────────────────────────────
  listen("transcript-update", (event) => {
    const payload = event.payload as TranscriptUpdate;
    display.applyUpdate(payload);
  });
}
