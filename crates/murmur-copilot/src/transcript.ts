/**
 * Manages the live transcript display with speaker-labelled entries.
 */

export interface TranscriptUpdate {
  speaker: "user" | "remote";
  text: string;
  replace_chars: number;
}

export class TranscriptDisplay {
  private container: HTMLElement;
  private entries: { speaker: string; text: string }[] = [];
  private userBuffer: string = "";
  private remoteBuffer: string = "";

  constructor(container: HTMLElement) {
    this.container = container;
  }

  /** Apply an incremental update from the streaming transcription engine. */
  applyUpdate(update: TranscriptUpdate): void {
    const isUser = update.speaker === "user";
    let buffer = isUser ? this.userBuffer : this.remoteBuffer;

    if (update.replace_chars > 0) {
      buffer = buffer.slice(0, -update.replace_chars);
    }
    buffer += update.text;

    if (isUser) {
      this.userBuffer = buffer;
    } else {
      this.remoteBuffer = buffer;
    }

    this.render();
  }

  /** Replace the display with final transcript entries. */
  setFinalTranscript(entries: { speaker: string; text: string }[]): void {
    this.entries = entries;
    this.userBuffer = "";
    this.remoteBuffer = "";
    this.render();
  }

  /** Clear the transcript display. */
  clear(): void {
    this.entries = [];
    this.userBuffer = "";
    this.remoteBuffer = "";
    this.render();
  }

  private render(): void {
    const lines: string[] = [];

    for (const entry of this.entries) {
      const label = entry.speaker === "user" ? "You" : "Remote";
      lines.push(`${label}: ${entry.text}`);
    }

    // Show live buffers at the end
    if (this.userBuffer) {
      lines.push(`You: ${this.userBuffer}`);
    }
    if (this.remoteBuffer) {
      lines.push(`Remote: ${this.remoteBuffer}`);
    }

    this.container.textContent = lines.join("\n");
    this.container.scrollTop = this.container.scrollHeight;
  }
}
