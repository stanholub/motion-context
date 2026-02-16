# Video-to-Code Visual Context MCP Server

A local Python MCP server that extracts representative video frames for coding/debugging context.

The server includes a session-based workflow optimized for token usage:
- First call returns only frame references and metadata.
- Follow-up calls return only the frame image(s) you actually need.

It also provides a one-shot workflow (`analyze_video`) that returns representative
frames and metadata in a single call.

## Features

- Coverage-first representative frame selection (scene + visual-change + uniform sampling).
- Scene detection timeout fallback with non-scene sampling to preserve coverage.
- Session extraction output target: `1280x720`; one-shot analysis supports mode presets with `auto` defaulting to `overview` (`overview=640x360`, `precise=1280x720`).
- Duration-aware frame density for one-shot analysis (`overview`: up to `1 FPS`; `precise`: up to `3 FPS` depending on intent).
- One-shot analysis window cap: first `150s` (`2.5 min`) of the input video.
- Adaptive JPEG quality fallback (`95 -> 90 -> 82 -> 74 -> 66 -> 58 -> 50`) when needed for response-size coverage.
- Max `8` frames per session extraction (`get_visual_context` / `get_visual_frames`); one-shot `analyze_video` supports higher dynamic frame counts.
- Temporary frame sessions with automatic expiration and cleanup.

## Requirements

- Python 3.10+
- `pip`

## Quick Start

1. Clone the repository and open it:

```bash
cd <PROJECT_ROOT>
```

2. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

4. Run the MCP server:

```bash
python3 server.py
```

This confirms the server starts. In normal MCP usage, your client launches
`server.py` automatically, so you usually do not run it manually.

## First-Time MCP Setup (Never Used MCP Before)

1. Complete **Quick Start** above.
2. Add this server to your MCP client config:

```json
{
  "mcpServers": {
    "video-context": {
      "command": "python3",
      "args": ["/absolute/path/to/server.py"]
    }
  }
}
```

3. Restart your MCP client (VS Code/Cursor/Claude Desktop) after saving config.
4. Open a chat in your client and run a smoke test prompt:
   - `Use analyze_video on /absolute/path/to/video.mp4 and summarize key actions.`
5. If it works, you should see the tool return frame images + metadata (timestamps, coverage, token fields).

Notes:
- Use absolute paths for both `server.py` and `video_path`.
- The MCP client starts/stops the server process for you.
- If tools are not visible after config changes, fully restart the client.

## MCP Tools

### `analyze_video(video_path: str, question?: str, max_frames?: number, resolution_mode?: "auto" | "precise" | "overview", max_estimated_tokens?: number, strict_evidence?: boolean, auto_tune?: boolean, ensure_end_frame?: boolean)`

One-shot tool for seamless chat usage.

Key output fields:
- representative frame images and timestamps in one response
- sampling metadata (`scene`, `change`, `uniform`, and blended combinations)
- truncation/size metadata
- estimated token metadata (`approx_estimated_tokens`, `token_limited`)
- effective JPEG quality (`jpeg_quality_used`) when quality fallback is applied
- coverage diagnostics (`coverage_level`, `coverage_percentage`, `max_gap_sec`, `tail_gap_sec`, `uncertain_intervals`)
- end-coverage enforcement metadata (`ensure_end_frame`, `end_frame_forced`)
- candidate/collection diagnostics (`candidate_timestamp_count`, `collection_timestamp_count`)

Use this when you want to stay in your current chat and just provide a video path.
For sparse scene detection, the tool automatically blends scene cuts, visual-change
sampling, and uniform coverage to preserve long-flow context.

Resolution modes:
- `auto` (default): always selects `overview`; use `resolution_mode="precise"` when you explicitly need high-detail analysis
- `precise`: `1280x720`, duration-based target density (`~2.0..3.0 FPS` by question intent), max `3 FPS`
- `overview`: `640x360` (longest side capped at `672px`), duration-based target density (`1 FPS`), max `1 FPS`
- for a `14s` clip this means defaults are at least `14` frames in `overview`, and typically `28..42` in `precise`
- no small static frame cap is used when duration is known; cap is duration-derived from mode FPS policy
- videos longer than `150s` are truncated to first `150s` for one-shot analysis
- backward-compatible aliases are accepted: `detail -> precise`, `balanced -> overview`, `flow -> overview`, `long -> overview`

Token guard:
- per-mode starting budgets: `precise=42000`, `overview=16000`
- hard ceiling for any request: `75000`
- per-frame estimate: `ceil(width * height / 512) + 48` tokens
- effective frame cap from tokens: `floor(effective_max_estimated_tokens / estimated_tokens_per_frame)`
- if `max_estimated_tokens` is omitted, server can raise the mode default budget to support duration-driven frame targets
- if exceeded, frames are reduced automatically and `token_limited` is returned
- session-based tools (`get_visual_frame`, `get_visual_frames`) also enforce cumulative
  session budget (`75000` by default) and stop once exhausted

Anti-hallucination guard:
- `strict_evidence` defaults to `true`
- when enabled, consumers should only describe visible evidence and mark uncovered
  intervals as unknown (see `uncertain_intervals`)
- for short, action-heavy clips, use `resolution_mode="precise"` and increase
  `max_frames`/`max_estimated_tokens` as needed for micro-interactions

Coverage guard:
- extraction is coverage-first: candidate timestamps are ordered to prioritize
  start/end and temporal spread before dense middle frames
- start (`0s`) and end-of-video timestamps are explicitly anchored in sampling
- when response-size budget is tight, oversized candidates are skipped (not hard-stop),
  so later timestamps can still be included
- `ensure_end_frame=true` (default) attempts to add/replace a frame near the video end
  if tail coverage is missing
- coverage thresholds are stricter by profile (no `4s` recommendation):
  `precise <= 1.6s`, `overview <= 2.5s`, `session <= 2.2s` on longer videos
- inspect `tail_gap_sec` and `coverage_percentage`; if `tail_gap_sec` is above
  `recommended_max_gap_sec`, treat the ending as uncertain

Auto guideline guard:
- `auto_tune` defaults to `true`
- the server inspects your question intent (`low`/`medium`/`high`) and auto-raises
  too-low `max_frames` and `max_estimated_tokens` to safer minimums
- this means even if the LLM picks conservative values (for example `max_frames=16`)
  the server can still boost settings for comprehensive action-level analysis
- set `auto_tune=false` only if you explicitly want strict manual control

### `get_visual_context(video_path: str)`

Creates a temporary frame session and returns metadata only.

Key output fields:
- `session_id`
- session token budget state (`session_estimated_tokens_budget`, `session_estimated_tokens_used`, `session_estimated_tokens_remaining`)
- `frames[]` with `frame_id`, timestamp, dimensions, file path, and byte size
- sampling method (`scene`, `change`, `uniform`, or blends like `scene+change+uniform`)
- truncation/size metadata
- coverage diagnostics (`coverage_level`, `coverage_percentage`, `tail_gap_sec`, `uncertain_intervals`)
- end-coverage metadata (`ensure_end_frame`, `end_frame_forced`)

### `get_visual_frame(session_id: str, frame_id: str)`

Returns a single frame image for one `frame_id`.
Consumes from session token budget.

### `get_visual_frames(session_id: str, frame_ids?: string[], max_frames?: number, max_estimated_tokens?: number)`

Returns multiple frame images in one call.

Behavior:
- If `frame_ids` is omitted, returns all frames in the session (up to limits).
- Hard cap: returns at most `8` frames per call (`MAX_FRAMES`).
- Enforces `max_frames`, response-size budget, and token budget.
- Sets `truncated: true` if not all requested frames fit.
- Consumes from session token budget and stops once exhausted.

### `cleanup_visual_context(session_id: str)`

Deletes temporary files for a session immediately.

## MCP Prompts

### `quick_video_review(video_path: str, question?: str, resolution_mode?: "auto" | "precise" | "overview", max_estimated_tokens?: number, strict_evidence?: boolean, auto_tune?: boolean, ensure_end_frame?: boolean)`

Prompt shortcut that asks the client to call `analyze_video` with sensible defaults.

## Recommended Workflow

1. Call `get_visual_context` with a video path.
2. Inspect returned metadata and pick relevant frame IDs.
3. Call `get_visual_frame` or `get_visual_frames`.
4. Call `cleanup_visual_context` when done.

## Seamless One-Message Workflow

If your chat client supports MCP tools in the current conversation, use `analyze_video` directly:

1. In your current Copilot/Claude chat, send:
   - `Analyze /absolute/path/to/video.mp4 and explain what broke around login.`
2. The model can call `analyze_video` once and answer immediately from returned frames.
3. Optional: ask follow-up questions in the same chat with the same video path.

You can still use the session-based tools when you need tighter token control.

## Example MCP Calls

### One-shot analysis (auto mode)

```json
{
  "tool": "analyze_video",
  "args": {
    "video_path": "/absolute/path/to/video.mp4",
    "question": "What UI changes happen after clicking Save?",
    "ensure_end_frame": true
  }
}
```

### One-shot analysis (overview mode for user journey)

```json
{
  "tool": "analyze_video",
  "args": {
    "video_path": "/absolute/path/to/video.mp4",
    "question": "Where does the onboarding animation stutter?",
    "resolution_mode": "overview",
    "max_frames": 22,
    "max_estimated_tokens": 26000,
    "strict_evidence": true,
    "auto_tune": true,
    "ensure_end_frame": true
  }
}
```

### Extract references

```json
{
  "tool": "get_visual_context",
  "args": {
    "video_path": "/absolute/path/to/video.mp4"
  }
}
```

### Fetch one frame

```json
{
  "tool": "get_visual_frame",
  "args": {
    "session_id": "abc123def456",
    "frame_id": "frame_0"
  }
}
```

### Fetch multiple frames

```json
{
  "tool": "get_visual_frames",
  "args": {
    "session_id": "abc123def456",
    "frame_ids": ["frame_0", "frame_2", "frame_4"],
    "max_frames": 6,
    "max_estimated_tokens": 8000
  }
}
```

### Cleanup session

```json
{
  "tool": "cleanup_visual_context",
  "args": {
    "session_id": "abc123def456"
  }
}
```

## VS Code / Cursor Setup

Add an MCP server entry using either a virtualenv Python binary or your system Python.
If this is your first MCP setup, open VS Code/Cursor settings and search for `MCP`
or `Model Context Protocol`, then add a server with the values below.

### Option A: Use venv Python

- `command`: `<PROJECT_ROOT>/.venv/bin/python`
- `args`: [`<PROJECT_ROOT>/server.py`]

### Option B: Use system Python

- `command`: `python3`
- `args`: [`<PROJECT_ROOT>/server.py`]

Example MCP config:

```json
{
  "mcpServers": {
    "video-context": {
      "command": "python3",
      "args": [
        "/absolute/path/to/server.py"
      ]
    }
  }
}
```

Then restart VS Code/Cursor (or reload MCP servers from the client UI).
Open a chat and run a quick check:
- `Use analyze_video on /absolute/path/to/video.mp4 and list timestamps only.`

Tip: if your MCP client surfaces prompts, use the `quick_video_review` prompt for a fast one-shot flow.

## Claude Desktop Setup

Edit `claude_desktop_config.json` and add:

- macOS (typical): `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows (typical): `%APPDATA%\Claude\claude_desktop_config.json`
- Linux (typical): `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "video-context": {
      "command": "python3",
      "args": [
        "/absolute/path/to/server.py"
      ]
    }
  }
}
```

Then restart Claude Desktop.

## Troubleshooting

- `Video file not found`: verify `video_path` is valid and absolute.
- `Session expired`: call `get_visual_context` again.
- Missing frame files: run extraction again (temporary files may have been cleaned).
- Slow scene detection on long videos: server continues with change/uniform sampling automatically.
