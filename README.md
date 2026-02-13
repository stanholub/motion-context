# Video-to-Code Visual Context MCP Server

A local Python MCP server that extracts representative video frames for coding/debugging context.

The server is optimized to reduce token usage:
- First call returns only frame references and metadata.
- Follow-up calls return only the frame image(s) you actually need.

## Features

- Scene-based frame selection with PySceneDetect.
- Fallback to uniform sampling (every 3 seconds) if scene detection fails or times out.
- 720p output (`1280x720`) with JPEG quality `95`.
- Max `8` frames per extraction session.
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

## MCP Tools

### `get_visual_context(video_path: str)`

Creates a temporary frame session and returns metadata only.

Returns:
- `session_id`
- `frames[]` with `frame_id`, timestamp, dimensions, file path, and byte size
- sampling method (`scene` or `uniform`)
- truncation/size metadata

### `get_visual_frame(session_id: str, frame_id: str)`

Returns a single frame image for one `frame_id`.

### `get_visual_frames(session_id: str, frame_ids?: string[], max_frames?: number)`

Returns multiple frame images in one call.

Behavior:
- If `frame_ids` is omitted, returns all frames in the session (up to limits).
- Enforces `max_frames` and response-size budget.
- Sets `truncated: true` if not all requested frames fit.

### `cleanup_visual_context(session_id: str)`

Deletes temporary files for a session immediately.

## Recommended Workflow

1. Call `get_visual_context` with a video path.
2. Inspect returned metadata and pick relevant frame IDs.
3. Call `get_visual_frame` or `get_visual_frames`.
4. Call `cleanup_visual_context` when done.

## Example MCP Calls

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
    "max_frames": 6
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

Then restart your MCP server from VS Code/Cursor.

## Claude Desktop Setup

Edit `claude_desktop_config.json` and add:

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
- Slow scene detection on long videos: server falls back to uniform sampling automatically.
