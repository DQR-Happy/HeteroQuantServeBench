import { base } from './api/client';
import type { StreamEvent } from './api/types';

/** Incremental SSE parser: transport chunks and output tokens are unrelated. */
export class SSEParser {
  private buffer = '';
  feed(text: string): StreamEvent[] {
    this.buffer += text;
    const result: StreamEvent[] = [];
    let match: RegExpMatchArray | null;
    while ((match = this.buffer.match(/\r?\n\r?\n/))) {
      const index = match.index!;
      const frame = this.buffer.slice(0, index);
      this.buffer = this.buffer.slice(index + match[0].length);
      const payload = frame
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n');
      if (!payload || payload === '[DONE]') continue;
      const event = JSON.parse(payload) as StreamEvent;
      if (event.event_version !== '1' || !Number.isInteger(event.seq) || !event.request_id)
        throw new Error('不兼容的流式事件');
      result.push(event);
    }
    if (this.buffer.length > 2_000_000) throw new Error('流式事件超出大小限制');
    return result;
  }
}
export async function followRun(
  id: string,
  after: number,
  signal: AbortSignal,
  receive: (event: StreamEvent) => void,
) {
  const response = await fetch(`${base}/requests/${encodeURIComponent(id)}/events?after=${after}`, {
    signal,
    credentials: 'same-origin',
  });
  if (!response.ok || !response.body) throw new Error(`无法订阅事件：HTTP ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  let cursor = after;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      const chunk = decoder.decode(value, { stream: !done });
      for (const event of parser.feed(chunk)) {
        if (event.request_id !== id) throw new Error('事件归属不匹配');
        if (event.seq <= cursor) continue;
        if (event.seq !== cursor + 1) throw new Error('事件序号存在缺口，请刷新快照');
        cursor = event.seq;
        receive(event);
      }
      if (done) break;
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
  return cursor;
}
