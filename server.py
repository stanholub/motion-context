from __future__ import annotations

import os
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import cv2
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent
from scenedetect import ContentDetector, SceneManager, open_video

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
JPEG_QUALITY = 95
MAX_FRAMES = 8
UNIFORM_INTERVAL_SEC = 3.0
SCENE_DETECT_TIMEOUT_SEC = 12.0
MAX_RESPONSE_BYTES = 1_000_000
SIZE_BUDGET_BYTES = 950_000
ESTIMATED_PER_FRAME_OVERHEAD = 220

mcp = FastMCP("Video Visual Context")


def _open_capture(video_path: str) -> Tuple[Optional[cv2.VideoCapture], Optional[float], Optional[float], Optional[str]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, "Unable to open video file."

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = None
    if frame_count and frame_count > 0 and fps > 0:
        duration_sec = frame_count / fps

    return cap, fps, duration_sec, None


def _resize_to_720p(frame):
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None

    scale = min(TARGET_WIDTH / width, TARGET_HEIGHT / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

    top = (TARGET_HEIGHT - new_height) // 2
    bottom = TARGET_HEIGHT - new_height - top
    left = (TARGET_WIDTH - new_width) // 2
    right = TARGET_WIDTH - new_width - left

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def _encode_jpeg_bytes(frame) -> Optional[bytes]:
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
            cv2.IMWRITE_JPEG_OPTIMIZE,
            1,
        ],
    )
    if not success:
        return None
    return buffer.tobytes()


def _estimate_base64_size(byte_len: int) -> int:
    return ((byte_len + 2) // 3) * 4


def _read_frame_at(cap: cv2.VideoCapture, timestamp_sec: float, fps: float):
    if timestamp_sec < 0:
        timestamp_sec = 0.0

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
    ok, frame = cap.read()
    if ok:
        return frame

    frame_index = int(round(timestamp_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if ok:
        return frame

    return None


def _select_evenly(values: List[float], max_count: int) -> List[float]:
    if len(values) <= max_count:
        return values
    if max_count <= 1:
        return [values[0]]

    step = (len(values) - 1) / (max_count - 1)
    indices = []
    seen = set()
    for i in range(max_count):
        idx = int(round(i * step))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return [values[i] for i in indices]


def _detect_scenes_with_timeout(
    video_path: str,
    timeout_sec: float,
    frame_skip: int,
) -> Tuple[Optional[List[Tuple[Any, Any]]], bool, Optional[str]]:
    result: Dict[str, Any] = {"scenes": None, "error": None, "timed_out": False}
    holder: Dict[str, Any] = {}

    def _run_detection():
        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            holder["scene_manager"] = scene_manager
            scene_manager.add_detector(ContentDetector())
            scene_manager.detect_scenes(video=video, frame_skip=frame_skip)
            result["scenes"] = scene_manager.get_scene_list(start_in_scene=True)
        except Exception as exc:  # pragma: no cover - safety net
            result["error"] = str(exc)

    thread = Thread(target=_run_detection, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        result["timed_out"] = True
        scene_manager = holder.get("scene_manager")
        if scene_manager is not None:
            try:
                scene_manager.stop()
            except Exception:
                pass

    return result["scenes"], result["timed_out"], result["error"]


def _scene_timestamps(video_path: str, duration_sec: Optional[float], fps: float):
    frame_skip = 0
    if duration_sec is not None:
        if duration_sec > 600:
            frame_skip = max(1, int(fps // 2))
        elif duration_sec > 300:
            frame_skip = max(1, int(fps // 4))

    scenes, timed_out, error = _detect_scenes_with_timeout(
        video_path=video_path,
        timeout_sec=SCENE_DETECT_TIMEOUT_SEC,
        frame_skip=frame_skip,
    )

    if timed_out:
        return None, "timeout"
    if error:
        return None, "error"
    if not scenes or len(scenes) <= 1:
        return None, "no_scenes"

    timestamps: List[float] = []
    for start, end in scenes:
        try:
            mid = (start.get_seconds() + end.get_seconds()) / 2.0
        except Exception:
            mid = float(start.get_seconds())
        timestamps.append(mid)

    return timestamps, "ok"


def _uniform_timestamps(duration_sec: Optional[float]) -> List[float]:
    if duration_sec is None or duration_sec <= 0:
        return [i * UNIFORM_INTERVAL_SEC for i in range(MAX_FRAMES)]

    times: List[float] = []
    t = 0.0
    while t <= duration_sec + 1e-3:
        times.append(t)
        t += UNIFORM_INTERVAL_SEC

    if not times:
        times = [0.0]

    return times


def _collect_frames(
    cap: cv2.VideoCapture,
    fps: float,
    timestamps: List[float],
) -> Tuple[List[Dict[str, Any]], List[Any], int, bool]:
    frames: List[Dict[str, Any]] = []
    content_blocks: List[Any] = []
    total_size = 0
    truncated = False

    for timestamp_sec in timestamps:
        if len(frames) >= MAX_FRAMES:
            break

        frame = _read_frame_at(cap, timestamp_sec, fps)
        if frame is None:
            continue

        resized = _resize_to_720p(frame)
        if resized is None:
            continue

        jpeg_bytes = _encode_jpeg_bytes(resized)
        if jpeg_bytes is None:
            continue

        estimated_size = _estimate_base64_size(len(jpeg_bytes)) + ESTIMATED_PER_FRAME_OVERHEAD
        if total_size + estimated_size > SIZE_BUDGET_BYTES:
            truncated = True
            break

        image_content = Image(data=jpeg_bytes, format="jpeg").to_image_content()
        content_blocks.append(image_content)

        frames.append(
            {
                "timestamp_sec": round(float(timestamp_sec), 3),
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
                "image_index": len(content_blocks) - 1,
                "jpeg_bytes": len(jpeg_bytes),
            }
        )
        total_size += estimated_size

    if total_size > MAX_RESPONSE_BYTES:
        truncated = True

    return frames, content_blocks, total_size, truncated


@mcp.tool
def get_visual_context(video_path: str) -> ToolResult:
    if not os.path.exists(video_path):
        return ToolResult(
            content=[TextContent(type="text", text="Video file not found.")],
            structured_content={
                "ok": False,
                "error": "Video file not found.",
                "video_path": video_path,
            },
        )

    if not os.path.isfile(video_path):
        return ToolResult(
            content=[TextContent(type="text", text="Video path is not a file.")],
            structured_content={
                "ok": False,
                "error": "Video path is not a file.",
                "video_path": video_path,
            },
        )

    cap, fps, duration_sec, error = _open_capture(video_path)
    if error or cap is None or fps is None:
        return ToolResult(
            content=[TextContent(type="text", text=error or "Failed to open video.")],
            structured_content={
                "ok": False,
                "error": error or "Failed to open video.",
                "video_path": video_path,
            },
        )

    method = "uniform"
    scene_status = "skipped"
    timestamps: List[float] = []

    scene_timestamps, scene_status = _scene_timestamps(video_path, duration_sec, fps)
    if scene_timestamps:
        method = "scene"
        timestamps = scene_timestamps
    else:
        method = "uniform"
        timestamps = _uniform_timestamps(duration_sec)

    timestamps = _select_evenly(timestamps, MAX_FRAMES)

    frames, content_blocks, approx_bytes, truncated = _collect_frames(cap, fps, timestamps)
    cap.release()

    if not frames:
        return ToolResult(
            content=[TextContent(type="text", text="Failed to extract frames from video.")],
            structured_content={
                "ok": False,
                "error": "Failed to extract frames from video.",
                "video_path": video_path,
            },
        )

    summary = (
        f"Extracted {len(frames)} frame(s) via {method} sampling "
        f"(scene_detection={scene_status})."
    )

    return ToolResult(
        content=[TextContent(type="text", text=summary), *content_blocks],
        structured_content={
            "ok": True,
            "method": method,
            "scene_detection": scene_status,
            "frame_count": len(frames),
            "approx_bytes": approx_bytes,
            "truncated": truncated,
            "duration_sec": round(duration_sec, 3) if duration_sec else None,
            "frames": frames,
        },
    )


if __name__ == "__main__":
    mcp.run()
