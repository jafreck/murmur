/**
 * Meeting history — browse, view, and export past sessions.
 */

declare global {
  interface Window {
    __TAURI__?: {
      core: {
        invoke: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;
      };
    };
  }
}

export interface SessionSummary {
  id: string;
  title: string | null;
  started_at: string;
  ended_at: string;
  duration_secs: number;
  entry_count: number;
}

export interface SavedSession {
  id: string;
  title: string | null;
  started_at: string;
  ended_at: string;
  duration_secs: number;
  transcript: { speaker: string; text: string; timestamp_ms: number }[];
  summary: string | null;
  action_items: string | null;
}

function invoke(cmd: string, args?: Record<string, unknown>): Promise<unknown> {
  return window.__TAURI__!.core.invoke(cmd, args);
}

function formatDuration(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export class MeetingHistory {
  private container: HTMLElement;
  private sessions: SessionSummary[] = [];

  constructor(container: HTMLElement) {
    this.container = container;
  }

  async loadSessions(): Promise<SessionSummary[]> {
    this.sessions = (await invoke("list_sessions")) as SessionSummary[];
    return this.sessions;
  }

  renderSessionList(): void {
    if (this.sessions.length === 0) {
      this.container.innerHTML =
        '<div class="history-empty">No saved meetings yet.</div>';
      return;
    }

    const list = document.createElement("div");
    list.className = "history-list";

    for (const s of this.sessions) {
      const item = document.createElement("div");
      item.className = "history-item";
      item.innerHTML = `
        <div class="history-item-header">
          <span class="history-title">${this.escapeHtml(s.title ?? "Untitled Meeting")}</span>
          <span class="history-date">${formatDate(s.started_at)}</span>
        </div>
        <div class="history-item-meta">
          <span>${formatDuration(s.duration_secs)}</span>
          <span>${s.entry_count} entries</span>
          <button class="btn btn-sm btn-export" data-id="${s.id}">Export</button>
          <button class="btn btn-sm btn-delete" data-id="${s.id}">Delete</button>
        </div>
      `;

      item.addEventListener("click", (e) => {
        const target = e.target as HTMLElement;
        if (target.classList.contains("btn-export")) {
          this.exportSession(s.id);
          return;
        }
        if (target.classList.contains("btn-delete")) {
          this.deleteSession(s.id);
          return;
        }
        this.showSessionDetail(s.id);
      });

      list.appendChild(item);
    }

    this.container.innerHTML = "";
    this.container.appendChild(list);
  }

  async showSessionDetail(id: string): Promise<void> {
    try {
      const session = (await invoke("get_session", { id })) as SavedSession;
      this.renderSessionDetail(session);
    } catch (err) {
      this.container.innerHTML = `<div class="history-error">Error: ${err}</div>`;
    }
  }

  renderSessionDetail(session: SavedSession): void {
    const div = document.createElement("div");
    div.className = "history-detail";

    let html = `
      <button class="btn btn-sm btn-back">← Back</button>
      <h3>${this.escapeHtml(session.title ?? "Untitled Meeting")}</h3>
      <div class="history-detail-meta">
        ${formatDate(session.started_at)} · ${formatDuration(session.duration_secs)}
      </div>
    `;

    if (session.summary) {
      html += `<div class="history-section"><strong>Summary</strong><p>${this.escapeHtml(session.summary)}</p></div>`;
    }
    if (session.action_items) {
      html += `<div class="history-section"><strong>Action Items</strong><p>${this.escapeHtml(session.action_items)}</p></div>`;
    }

    html += `<div class="history-section"><strong>Transcript</strong><div class="history-transcript">`;
    for (const entry of session.transcript) {
      const label = entry.speaker === "user" ? "You" : "Remote";
      html += `<div><strong>${label}:</strong> ${this.escapeHtml(entry.text)}</div>`;
    }
    html += `</div></div>`;

    div.innerHTML = html;

    const backBtn = div.querySelector(".btn-back") as HTMLButtonElement;
    backBtn.addEventListener("click", () => {
      this.renderSessionList();
    });

    this.container.innerHTML = "";
    this.container.appendChild(div);
  }

  async exportSession(id: string): Promise<void> {
    try {
      const markdown = (await invoke("export_session", { id })) as string;
      const blob = new Blob([markdown], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `meeting-${id}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("export failed:", err);
    }
  }

  async deleteSession(id: string): Promise<void> {
    try {
      await invoke("delete_session", { id });
      await this.loadSessions();
      this.renderSessionList();
    } catch (err) {
      console.error("delete failed:", err);
    }
  }

  private escapeHtml(text: string): string {
    const el = document.createElement("span");
    el.textContent = text;
    return el.innerHTML;
  }
}
