# Video-to-Code Visual Context MCP Server

This project provides a local MCP server that extracts visual context from videos using scene detection and returns up to 8 high-quality frames as Base64-encoded JPEGs.

## Requirements

- Python 3.10+ recommended
- `pip` available in your environment

## Install Dependencies

```bash
python -m pip install -r /Users/stanislavholub/work/side-projects/motion-context/requirements.txt
```

## Run The Server

```bash
python /Users/stanislavholub/work/side-projects/motion-context/server.py
```

The server will start and expose the MCP tool `get_visual_context(video_path: str)`.

## Tool Behavior

- Uses PySceneDetect to find scene changes and returns one representative frame per scene.
- Falls back to uniform sampling (one frame every 3 seconds) if no scenes are detected or detection is too slow.
- Outputs frames at 1280x720, JPEG quality 95, as MCP image content blocks (so clients can pass them as images instead of text).
- Limits output to 8 frames and keeps the response under 1MB.

## Example MCP Call

```json
{
  "tool": "get_visual_context",
  "args": {
    "video_path": "/absolute/path/to/video.mp4"
  }
}
```

## Add To VS Code / Cursor

1. Open VS Code/Cursor Settings → MCP Servers → Add Server.
2. Use this configuration:
   - **Command**: `/Users/stanislavholub/work/side-projects/motion-context/venv/bin/python`
   - **Args**: `/Users/stanislavholub/work/side-projects/motion-context/server.py`
3. Save and restart VS Code/Cursor.

Alternatively, you can manually edit your MCP settings file (`~/.vscode/mcp.json` or `~/Library/Application Support/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`):

```json
{
  "mcpServers": {
    "video-context": {
      "command": "/Users/stanislavholub/work/side-projects/motion-context/venv/bin/python",
      "args": [
        "/Users/stanislavholub/work/side-projects/motion-context/server.py"
      ]
    }
  }
}
```

## Add To Claude Desktop

1. Open `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Add this server configuration:

```json
{
  "mcpServers": {
    "video-context": {
      "command": "/Users/stanislavholub/work/side-projects/motion-context/venv/bin/python",
      "args": [
        "/Users/stanislavholub/work/side-projects/motion-context/server.py"
      ]
    }
  }
}
```

3. Restart Claude Desktop.

## Troubleshooting

- If you see `Video file not found`, verify the `video_path` is correct and absolute.
- If extraction is slow on very long videos, the tool will automatically switch to fast uniform sampling.
