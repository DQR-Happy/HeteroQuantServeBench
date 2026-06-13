"""SSE wire codec, parser oracle and stream reconstruction (E08-01 §7).

The only thing that proves framing is correct is the *raw bytes*: an SDK that
reads a stream successfully says nothing about `data:` fields, event
separators, sequence monotonicity or the terminal marker.  This module therefore
keeps raw bytes and parsed frames side by side, and provides:

* :func:`encode_data_frame` / :func:`encode_terminal` — the writer side;
* :class:`SseReassembler` — the reader side, tolerant of frames split across
  arbitrary TCP reads (including inside a UTF-8 code point) and of several
  frames arriving in one read;
* :func:`validate_stream` — the framing oracle (sequence, terminal uniqueness
  and position, no normal frame after an error, heartbeat excluded);
* :func:`stream_vs_nonstream` — the paired oracle: delta concatenation must
  equal the non-stream body, including finish reason and usage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Default terminal marker of the frozen subset (see the protocol profile).
DEFAULT_TERMINAL_MARKER = "[DONE]"


@dataclass(frozen=True)
class SseFrame:
    """One parsed SSE event plus the raw bytes it came from."""

    index: int
    raw: bytes
    data: str
    is_terminal: bool = False
    is_heartbeat: bool = False
    payload: Optional[Mapping[str, Any]] = None
    parse_error: str = ""

    def token_ids(self) -> Tuple[int, ...]:
        """Token IDs carried by this frame, if it is a normal content frame."""
        if self.payload is None or self.is_terminal:
            return ()
        choices = self.payload.get("choices") or []
        tokens: List[int] = []
        for choice in choices:
            for key in ("delta", "text"):
                segment = choice.get(key)
                if isinstance(segment, dict) and "token_ids" in segment:
                    tokens.extend(int(item) for item in segment["token_ids"])
        return tuple(tokens)

    def text(self) -> str:
        if self.payload is None or self.is_terminal:
            return ""
        parts: List[str] = []
        for choice in self.payload.get("choices") or []:
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                parts.append(delta["content"])
            elif isinstance(choice.get("text"), str):
                parts.append(choice["text"])
        return "".join(parts)

    def finish_reason(self) -> str:
        if self.payload is None:
            return ""
        for choice in self.payload.get("choices") or []:
            reason = choice.get("finish_reason")
            if reason:
                return str(reason)
        return ""

    def usage(self) -> Optional[Mapping[str, Any]]:
        if self.payload is None:
            return None
        usage = self.payload.get("usage")
        return usage if isinstance(usage, dict) else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "raw": self.raw.decode("utf-8", errors="replace"),
            "data": self.data,
            "is_terminal": self.is_terminal,
            "is_heartbeat": self.is_heartbeat,
            "payload": dict(self.payload) if self.payload else None,
            "parse_error": self.parse_error,
        }


# ── writer side ────────────────────────────────────────────────────────────


def encode_data_frame(payload: Mapping[str, Any], *, data_prefix: str = "data: ") -> bytes:
    return (data_prefix + json.dumps(dict(payload), ensure_ascii=False) + "\n\n").encode("utf-8")


def encode_terminal(*, data_prefix: str = "data: ", marker: str = DEFAULT_TERMINAL_MARKER) -> bytes:
    return (data_prefix + marker + "\n\n").encode("utf-8")


def encode_heartbeat(text: str = "keep-alive") -> bytes:
    """A comment line: valid SSE, never part of the token sequence."""
    return (": " + text + "\n\n").encode("utf-8")


def encode_usage_frame(
    usage: Mapping[str, Any],
    *,
    model: str = "",
    frame_id: str = "",
    data_prefix: str = "data: ",
) -> bytes:
    payload = {
        "id": frame_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [],
        "usage": dict(usage),
    }
    return encode_data_frame(payload, data_prefix=data_prefix)


# ── reader side ────────────────────────────────────────────────────────────


class SseReassembler:
    """Incremental SSE parser that never assumes a frame fits one read."""

    def __init__(self, *, data_prefix: str = "data: ", terminal_marker: str = DEFAULT_TERMINAL_MARKER):
        self._buffer = b""
        self._index = 0
        self.data_prefix = data_prefix
        self.terminal_marker = terminal_marker
        self.frames: List[SseFrame] = []
        self.errors: List[str] = []

    # -- internals ------------------------------------------------------
    def _parse_event(self, raw: bytes) -> SseFrame:
        text = raw.decode("utf-8", errors="replace")
        data_lines = []
        is_heartbeat = False
        for line in text.split("\n"):
            if not line:
                continue
            if line.startswith(":"):
                is_heartbeat = True
                continue
            if line.startswith("data:"):
                value = line[len("data:"):]
                data_lines.append(value[1:] if value.startswith(" ") else value)
            elif line.startswith(("event:", "id:", "retry:")):
                continue
            else:
                self.errors.append(f"unknown SSE field {line[:16]!r}")
        data = "\n".join(data_lines)
        is_terminal = data.strip() == self.terminal_marker
        payload: Optional[Mapping[str, Any]] = None
        parse_error = ""
        if data and not is_terminal:
            try:
                decoded = json.loads(data)
                if isinstance(decoded, dict):
                    payload = decoded
                else:
                    parse_error = "SSE data payload must be a JSON object"
            except json.JSONDecodeError as exc:
                parse_error = f"invalid JSON in SSE data: {exc.msg}"
        frame = SseFrame(
            index=self._index,
            raw=raw,
            data=data,
            is_terminal=is_terminal,
            is_heartbeat=is_heartbeat and not data_lines,
            payload=payload,
            parse_error=parse_error,
        )
        self._index += 1
        return frame

    # -- public API -----------------------------------------------------
    def feed(self, chunk: bytes) -> List[SseFrame]:
        """Consume one TCP read; returns the frames that are now complete."""
        self._buffer += chunk
        produced: List[SseFrame] = []
        while True:
            separator = self._buffer.find(b"\n\n")
            if separator < 0:
                break
            raw = self._buffer[:separator]
            self._buffer = self._buffer[separator + 2 :]
            if not raw.strip():
                continue  # tolerates duplicated blank lines
            frame = self._parse_event(raw)
            self.frames.append(frame)
            produced.append(frame)
        return produced

    def finish(self) -> Dict[str, Any]:
        """Close the stream: leftover bytes are reported, never ignored."""
        trailing = self._buffer
        if trailing.strip():
            self.errors.append(
                f"truncated SSE frame at end of stream: {trailing[:32]!r}"
            )
        self._buffer = b""
        return {
            "frames": len(self.frames),
            "errors": list(self.errors),
            "trailing_bytes": len(trailing),
            "complete": not self.errors,
        }


def parse_stream(raw: bytes, **kwargs: Any) -> Dict[str, Any]:
    """Parse a whole captured stream in one call (fixtures, raw evidence)."""
    reassembler = SseReassembler(**kwargs)
    reassembler.feed(raw)
    report = reassembler.finish()
    report["parsed"] = reassembler.frames
    return report


def validate_stream(
    frames: Sequence[SseFrame],
    *,
    terminal_marker: str = DEFAULT_TERMINAL_MARKER,
    expect_terminal: bool = True,
) -> Dict[str, Any]:
    """The framing oracle of E08-01 §7."""
    problems: List[str] = []
    normal = [frame for frame in frames if not frame.is_terminal and not frame.is_heartbeat]
    terminals = [frame for frame in frames if frame.is_terminal]
    if expect_terminal:
        if len(terminals) != 1:
            problems.append(f"expected exactly one terminal marker, found {len(terminals)}")
        elif frames and frames[-1] is not terminals[0]:
            problems.append("the terminal marker is not the last frame")
    elif terminals:
        problems.append("terminal marker present although the profile does not expect one")
    for frame in frames:
        if frame.is_terminal and frame.parse_error:
            problems.append("terminal marker must not carry a JSON payload")
    expected_index = 0
    for frame in frames:
        if frame.index != expected_index:
            problems.append(
                f"frame sequence is not monotonic: expected {expected_index}, got {frame.index}"
            )
        expected_index = frame.index + 1
        if frame.parse_error:
            problems.append(f"frame {frame.index}: {frame.parse_error}")
        if frame.is_terminal and frame.is_heartbeat:
            problems.append(f"frame {frame.index} is both terminal and heartbeat")
        if frame.raw and not frame.raw.strip():
            problems.append(f"frame {frame.index} is empty")
    if terminal_marker != DEFAULT_TERMINAL_MARKER:
        for frame in terminals:
            if frame.data.strip() != terminal_marker:
                problems.append(f"terminal frame data {frame.data!r} != {terminal_marker!r}")
    for frame in normal:
        if frame.payload is None:
            problems.append(f"normal frame {frame.index} carries no parsed payload")
            continue
        for name in ("id", "object", "model", "choices"):
            if name not in frame.payload:
                problems.append(f"frame {frame.index} is missing {name!r}")
    return {
        "ok": not problems,
        "problems": problems,
        "frames": len(frames),
        "normal_frames": len(normal),
        "heartbeats": len(frames) - len(normal) - len(terminals),
    }


def reconstruct(frames: Sequence[SseFrame]) -> Dict[str, Any]:
    """Rebuild the final answer from deltas (never from the SDK string)."""
    text_parts: List[str] = []
    token_ids: List[int] = []
    finish_reason = ""
    usage: Optional[Mapping[str, Any]] = None
    for frame in frames:
        if frame.is_heartbeat or frame.is_terminal:
            continue
        text_parts.append(frame.text())
        token_ids.extend(frame.token_ids())
        finish_reason = frame.finish_reason() or finish_reason
        usage = frame.usage() or usage
    return {
        "text": "".join(text_parts),
        "token_ids": token_ids,
        "finish_reason": finish_reason,
        "usage": dict(usage) if usage else None,
    }


def stream_vs_nonstream(
    frames: Sequence[SseFrame],
    non_stream_body: Mapping[str, Any],
) -> Dict[str, Any]:
    """Paired oracle: stream concatenation must equal the non-stream answer."""
    problems: List[str] = []
    rebuilt = reconstruct(frames)
    choices = non_stream_body.get("choices") or [{}]
    choice = choices[0] if choices else {}
    final_text = choice.get("text")
    if final_text is None:
        message = choice.get("message") or {}
        final_text = message.get("content", "")
    if rebuilt["text"] != final_text:
        problems.append("streamed text does not equal the non-stream text")
    final_tokens = choice.get("token_ids")
    if final_tokens is not None and list(final_tokens) != rebuilt["token_ids"]:
        problems.append("streamed token IDs do not equal the non-stream token IDs")
    if rebuilt["finish_reason"] != choice.get("finish_reason"):
        problems.append("stream finish reason differs from the non-stream finish reason")
    usage = non_stream_body.get("usage")
    if usage and rebuilt["usage"]:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if rebuilt["usage"].get(name) is not None and rebuilt["usage"][name] != usage.get(name):
                problems.append(f"streamed usage {name} differs from the non-stream usage")
    return {
        "ok": not problems,
        "problems": problems,
        "reconstructed": rebuilt,
        "final_text": final_text,
    }


# ── parser oracle / self-verification (E08-01 step 4, E08-08 step 4) ───────


def fragmentation_cases(raw: bytes, *, splits: Sequence[int] = (1, 3, 7)) -> List[Dict[str, Any]]:
    """Split one captured stream at several byte offsets (TCP read boundaries)."""
    cases: List[Dict[str, Any]] = []
    length = len(raw)
    for split in splits:
        if not 0 < split < length:
            continue
        reassembler = SseReassembler()
        reassembler.feed(raw[:split])
        reassembler.feed(raw[split:])
        report = reassembler.finish()
        report["case"] = f"split_at_{split}"
        report["frames"] = reassembler.frames
        cases.append(report)
    return cases


def parser_oracle_cases(raw: bytes) -> Dict[str, Any]:
    """Legal vs. illegal byte streams a client parser must classify correctly."""
    full = parse_stream(raw)
    baseline_frames = len(full["parsed"])
    fragmented = fragmentation_cases(raw)
    sticky = parse_stream(raw + raw)  # two responses in one read
    truncated = parse_stream(raw[:-2]) if len(raw) > 2 else parse_stream(b"")
    invalid_prefix = parse_stream(b"event: message\nfoo: bar\n\n" + raw)
    doubled_blank = parse_stream(raw.replace(b"\n\n", b"\n\n\n\n", 1))
    return {
        "baseline_frames": baseline_frames,
        "baseline_ok": full["complete"],
        "fragmented_all_ok": all(item["complete"] for item in fragmented),
        "fragmented_counts": [len(item["frames"]) for item in fragmented],
        "sticky_detected_extra_frames": sticky["frames"] > baseline_frames,
        "truncated_reported": not truncated["complete"],
        "unknown_field_reported": bool(invalid_prefix["errors"]),
        "doubled_blank_tolerated": doubled_blank["complete"],
    }


def utf8_boundary_cases(payload_texts: Sequence[str] = ("你好", "🙂", "aí")) -> List[Dict[str, Any]]:
    """A multibyte character split across two reads must still parse."""
    results: List[Dict[str, Any]] = []
    for text in payload_texts:
        frame = encode_data_frame(
            {
                "id": "cmpl-1",
                "object": "chat.completion.chunk",
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
        )
        reassembler = SseReassembler()
        # split inside the multi-byte sequence
        cut = frame.find(text.encode("utf-8")[:1]) + 1
        reassembler.feed(frame[:cut])
        reassembler.feed(frame[cut:])
        report = reassembler.finish()
        rebuild = reconstruct(reassembler.frames)
        results.append(
            {
                "text": text,
                "complete": report["complete"],
                "reconstructed": rebuild["text"],
                "ok": report["complete"] and rebuild["text"] == text,
            }
        )
    return results


def require_no_silent_drop(frames: Sequence[SseFrame], expected_tokens: int) -> None:
    """A stream must never drop a token and keep pretending to be complete."""
    delivered = sum(len(frame.token_ids()) for frame in frames)
    if delivered != expected_tokens:
        raise ConfigError(
            f"stream delivered {delivered} tokens but {expected_tokens} were committed; "
            "silently dropping tokens inside a stream is forbidden (E08-08 §6)",
            details={"delivered": delivered, "expected": expected_tokens},
        )


def frames_from_payloads(payloads: Iterable[Mapping[str, Any]]) -> List[SseFrame]:
    """Helper for fixtures: encode payloads and parse them back."""
    raw = b"".join(encode_data_frame(payload) for payload in payloads) + encode_terminal()
    return list(parse_stream(raw)["parsed"])


__all__ = [
    "DEFAULT_TERMINAL_MARKER",
    "SseFrame",
    "SseReassembler",
    "encode_data_frame",
    "encode_heartbeat",
    "encode_terminal",
    "encode_usage_frame",
    "fragmentation_cases",
    "frames_from_payloads",
    "parse_stream",
    "parser_oracle_cases",
    "reconstruct",
    "require_no_silent_drop",
    "stream_vs_nonstream",
    "utf8_boundary_cases",
    "validate_stream",
]
