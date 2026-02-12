#!/bin/bash
# Start the Video Visual Context MCP Server

cd "$(dirname "$0")"
source venv/bin/activate
python server.py
