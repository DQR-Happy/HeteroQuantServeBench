import { describe, expect, it, vi, afterEach } from 'vitest';
import { followRun, SSEParser } from './streaming';
const event = (seq = 1) => ({
  event_version: '1',
  request_id: 'run_a',
  seq,
  kind: 'output.snapshot',
  data: { text: '你好' },
});
const frame = (seq = 1) => `id: run_a:${seq}\ndata: ${JSON.stringify(event(seq))}\n\n`;
describe('SSE transport', () => {
  afterEach(() => vi.unstubAllGlobals());
  it('preserves arbitrarily fragmented frames and comments', () => {
    const p = new SSEParser();
    const text = ': heartbeat\n\n' + frame();
    const events = [];
    for (const c of text) events.push(...p.feed(c));
    expect(events).toEqual([event()]);
  });
  it('accepts CRLF and multiple frames per network read', () => {
    expect(new SSEParser().feed((frame() + frame(2)).replaceAll('\n', '\r\n'))).toHaveLength(2);
  });
  it('rejects incompatible event versions', () => {
    expect(() => new SSEParser().feed(frame().replace('"1"', '"2"'))).toThrow();
  });
  it('bounds incomplete frames', () => {
    expect(() => new SSEParser().feed('x'.repeat(2000001))).toThrow();
  });
  it('decodes UTF-8 across bytes and ignores replayed sequence numbers', async () => {
    const bytes = new TextEncoder().encode(frame() + frame() + frame(2));
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          new ReadableStream({
            start(c) {
              for (const b of bytes) c.enqueue(new Uint8Array([b]));
              c.close();
            },
          }),
        ),
      ),
    );
    const got: number[] = [];
    expect(await followRun('run_a', 0, new AbortController().signal, (e) => got.push(e.seq))).toBe(
      2,
    );
    expect(got).toEqual([1, 2]);
  });
  it('detects a sequence gap instead of joining inconsistent output', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(frame(2))));
    await expect(followRun('run_a', 0, new AbortController().signal, () => {})).rejects.toThrow(
      '缺口',
    );
  });
  it('rejects an event belonging to another request', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(frame())));
    await expect(followRun('run_other', 0, new AbortController().signal, () => {})).rejects.toThrow(
      '归属',
    );
  });
});
