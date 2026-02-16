# Video-to-Code Visual Context MCP Server

A local Python MCP server that extracts representative frames from videos so coding agents can reason about UI behavior, regressions, and flow changes from visual evidence.

## Use Cases

- Debug product issues from screen recordings without manually scrubbing videos.
- Review UI/UX flows and identify where state changes, glitches, or broken interactions occur.
- Generate grounded, frame-based summaries for QA, engineering handoff, and incident analysis.

## Setup

1. Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

2. Add the MCP server to VS Code/Cursor:

```json
{
  "mcpServers": {
    "motion-context": {
      "command": "/path/to/motion-context/venv/bin/python",
      "args": ["/path/to/motion-context/server.py"]
    }
  }
}
```

3. If you use Claude Code, add the same server from CLI:

```bash
claude mcp add-json motion-context '{"type":"stdio","command":"/path/to/motion-context/venv/bin/python","args":["/path/to/motion-context/server.py"]}'
```

4. Restart your MCP client and run `analyze_video` on an absolute `video_path`.

For full documentation (tools, workflows, advanced configuration, and troubleshooting), see [AGENTS.md](AGENTS.md).
