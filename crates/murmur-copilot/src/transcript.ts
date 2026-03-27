/**
 * Manages the live transcript display, applying partial-text updates
 * from the streaming transcription engine.
 */
export class TranscriptDisplay {
  private container: HTMLElement;
  private buffer: string = "";

  constructor(container: HTMLElement) {
    this.container = container;
  }

  /** Apply an incremental update from StreamingEvent::PartialText. */
  applyUpdate(text: string, replaceChars: number): void {
    if (replaceChars > 0) {
      this.buffer = this.buffer.slice(0, -replaceChars);
    }
    this.buffer += text;
    this.render();
  }

  /** Replace the display with the final transcript text. */
  setFinalTranscript(text: string): void {
    this.buffer = text;
    this.render();
  }

  /** Clear the transcript display. */
  clear(): void {
    this.buffer = "";
    this.render();
  }

  private render(): void {
    this.container.textContent = this.buffer;
    // Auto-scroll to the bottom so the latest text is always visible.
    this.container.scrollTop = this.container.scrollHeight;
  }
}
