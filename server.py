from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from threading import Lock
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
SESSION_TTL_SEC = 3600
MAX_SESSIONS = 20
TEMP_ROOT_DIR = os.path.join(tempfile.gettempdir(), "mcp-video-context")
SERVER_VERSION = "ref-sessions-v2"

mcp = FastMCP("Video Visual Context")
SESSION_LOCK = Lock()
SESSION_STORE: Dict[str, Dict[str, Any]] = {}


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


def _ensure_temp_root() -> None:
    os.makedirs(TEMP_ROOT_DIR, exist_ok=True)


def _delete_session_dir(session_dir: str) -> None:
    if os.path.isdir(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)


def _cleanup_sessions() -> None:
    now = time.time()
    remove_ids: List[str] = []

    with SESSION_LOCK:
        for session_id, session in SESSION_STORE.items():
            created_at = float(session.get("created_at", 0))
            if now - created_at > SESSION_TTL_SEC:
                remove_ids.append(session_id)

        if len(SESSION_STORE) - len(remove_ids) > MAX_SESSIONS:
            keep_count = MAX_SESSIONS
            sessions_sorted = sorted(
                SESSION_STORE.items(),
                key=lambda item: float(item[1].get("created_at", 0)),
            )
            extra = len(SESSION_STORE) - len(remove_ids) - keep_count
            for session_id, _ in sessions_sorted:
                if extra <= 0:
                    break
                if session_id in remove_ids:
                    continue
                remove_ids.append(session_id)
                extra -= 1

        removed_dirs = []
        for session_id in remove_ids:
            removed = SESSION_STORE.pop(session_id, None)
            if removed:
                removed_dirs.append(str(removed.get("session_dir", "")))

    for session_dir in removed_dirs:
        _delete_session_dir(session_dir)


def _create_session(
    video_path: str,
    method: str,
    scene_status: str,
    duration_sec: Optional[float],
    frames_with_bytes: List[Dict[str, Any]],
    approx_bytes: int,
    truncated: bool,
) -> Dict[str, Any]:
    _cleanup_sessions()
    _ensure_temp_root()

    session_id = uuid.uuid4().hex[:12]
    session_dir = os.path.join(TEMP_ROOT_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    frames: List[Dict[str, Any]] = []
    for idx, frame in enumerate(frames_with_bytes):
        frame_id = f"frame_{idx}"
        timestamp_sec = float(frame["timestamp_sec"])
        file_name = f"{idx:02d}_{int(timestamp_sec * 1000):06d}.jpg"
        file_path = os.path.join(session_dir, file_name)

        with open(file_path, "wb") as out:
            out.write(frame["jpeg_data"])

        frames.append(
            {
                "frame_id": frame_id,
                "frame_index": idx,
                "timestamp_sec": timestamp_sec,
                "width": int(frame["width"]),
                "height": int(frame["height"]),
                "jpeg_bytes": int(frame["jpeg_bytes"]),
                "file_path": file_path,
            }
        )

    session = {
        "session_id": session_id,
        "session_dir": session_dir,
        "created_at": time.time(),
        "expires_at": time.time() + SESSION_TTL_SEC,
        "video_path": video_path,
        "method": method,
        "scene_detection": scene_status,
        "duration_sec": duration_sec,
        "approx_bytes": approx_bytes,
        "truncated": truncated,
        "frames": frames,
    }

    with SESSION_LOCK:
        SESSION_STORE[session_id] = session

    return session


def _get_session(session_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    _cleanup_sessions()

    with SESSION_LOCK:
        session = SESSION_STORE.get(session_id)

    if session is None:
        return None, "Session not found."

    if time.time() > float(session.get("expires_at", 0)):
        _remove_session(session_id)
        return None, "Session expired. Run get_visual_context again."

    return session, None


def _remove_session(session_id: str) -> bool:
    with SESSION_LOCK:
        removed = SESSION_STORE.pop(session_id, None)
    if not removed:
        return False
    _delete_session_dir(str(removed.get("session_dir", "")))
    return True


def _get_frame_from_session(
    session: Dict[str, Any],
    frame_id: str,
) -> Optional[Dict[str, Any]]:
    return next((item for item in session["frames"] if item["frame_id"] == frame_id), None)


def _read_frame_bytes(frame: Dict[str, Any]) -> Tuple[Optional[bytes], Optional[str]]:
    file_path = str(frame["file_path"])
    if not os.path.isfile(file_path):
        return None, "Frame file missing. Re-run get_visual_context."

    with open(file_path, "rb") as image_file:
        return image_file.read(), None


def _resolve_frame_ids(
    session: Dict[str, Any],
    frame_ids: Optional[List[str]],
) -> List[str]:
    if not frame_ids:
        return [str(frame["frame_id"]) for frame in session["frames"]]

    seen = set()
    resolved: List[str] = []
    for frame_id in frame_ids:
        frame_id_str = str(frame_id)
        if frame_id_str in seen:
            continue
        seen.add(frame_id_str)
        resolved.append(frame_id_str)
    return resolved


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
) -> Tuple[List[Dict[str, Any]], int, bool]:
    frames: List[Dict[str, Any]] = []
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

        frames.append(
            {
                "timestamp_sec": round(float(timestamp_sec), 3),
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
                "jpeg_bytes": len(jpeg_bytes),
                "jpeg_data": jpeg_bytes,
            }
        )
        total_size += estimated_size

    if total_size > MAX_RESPONSE_BYTES:
        truncated = True

    return frames, total_size, truncated


@mcp.tool
def get_visual_context(video_path: str) -> ToolResult:
    """Extract key frame references from a video and create a temporary frame session.

    Uses scene detection first, then falls back to uniform sampling if needed.
    Returns only lightweight metadata and frame references. Fetch actual image
    bytes with `get_visual_frame`.
    """
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

    frames_with_bytes, approx_bytes, truncated = _collect_frames(cap, fps, timestamps)
    cap.release()

    if not frames_with_bytes:
        return ToolResult(
            content=[TextContent(type="text", text="Failed to extract frames from video.")],
            structured_content={
                "ok": False,
                "error": "Failed to extract frames from video.",
                "video_path": video_path,
            },
        )

    session = _create_session(
        video_path=video_path,
        method=method,
        scene_status=scene_status,
        duration_sec=duration_sec,
        frames_with_bytes=frames_with_bytes,
        approx_bytes=approx_bytes,
        truncated=truncated,
    )

    summary = (
        f"Extracted {len(session['frames'])} frame reference(s). "
        "Call get_visual_frame for one frame or get_visual_frames for a batch."
    )

    return ToolResult(
        content=[TextContent(type="text", text=summary)],
        structured_content={
            "ok": True,
            "server_version": SERVER_VERSION,
            "session_id": session["session_id"],
            "session_expires_at": int(session["expires_at"]),
            "session_ttl_sec": SESSION_TTL_SEC,
            "method": method,
            "scene_detection": scene_status,
            "frame_count": len(session["frames"]),
            "approx_bytes": approx_bytes,
            "truncated": truncated,
            "duration_sec": round(duration_sec, 3) if duration_sec else None,
            "frames": session["frames"],
        },
    )


@mcp.tool
def get_visual_frame(session_id: str, frame_id: str) -> ToolResult:
    """Return one JPEG frame image from an existing visual-context session.

    Use the `session_id` and `frame_id` values returned by `get_visual_context`.
    """
    session, error = _get_session(session_id)
    if error:
        return ToolResult(
            content=[TextContent(type="text", text=error)],
            structured_content={"ok": False, "error": error, "session_id": session_id},
        )

    assert session is not None

    frame = _get_frame_from_session(session, frame_id)
    if frame is None:
        return ToolResult(
            content=[TextContent(type="text", text="Frame not found in session.")],
            structured_content={
                "ok": False,
                "error": "Frame not found in session.",
                "session_id": session_id,
                "frame_id": frame_id,
            },
        )

    image_bytes, frame_read_error = _read_frame_bytes(frame)
    if frame_read_error:
        return ToolResult(
            content=[TextContent(type="text", text=frame_read_error)],
            structured_content={
                "ok": False,
                "error": frame_read_error,
                "session_id": session_id,
                "frame_id": frame_id,
            },
        )
    assert image_bytes is not None

    return ToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Frame {frame_id} from session {session_id}.",
            ),
            Image(data=image_bytes, format="jpeg").to_image_content(),
        ],
        structured_content={
            "ok": True,
            "server_version": SERVER_VERSION,
            "session_id": session_id,
            "frame_id": frame_id,
            "frame_index": frame["frame_index"],
            "timestamp_sec": frame["timestamp_sec"],
            "width": frame["width"],
            "height": frame["height"],
            "jpeg_bytes": frame["jpeg_bytes"],
            "file_path": frame["file_path"],
        },
    )


@mcp.tool
def get_visual_frames(
    session_id: str,
    frame_ids: Optional[List[str]] = None,
    max_frames: int = MAX_FRAMES,
) -> ToolResult:
    """Return multiple JPEG frame images from a visual-context session.

    If `frame_ids` is omitted, returns all session frames up to `max_frames`
    and the response-size budget.
    """
    session, error = _get_session(session_id)
    if error:
        return ToolResult(
            content=[TextContent(type="text", text=error)],
            structured_content={"ok": False, "error": error, "session_id": session_id},
        )

    assert session is not None

    if max_frames <= 0:
        return ToolResult(
            content=[TextContent(type="text", text="max_frames must be greater than 0.")],
            structured_content={
                "ok": False,
                "error": "max_frames must be greater than 0.",
                "session_id": session_id,
            },
        )

    requested_ids = _resolve_frame_ids(session, frame_ids)
    selected_ids = requested_ids[: min(max_frames, MAX_FRAMES)]

    image_blocks: List[Any] = []
    returned_frames: List[Dict[str, Any]] = []
    missing_frame_ids: List[str] = []
    missing_files: List[str] = []
    approx_response_bytes = 0
    truncated = False

    for current_frame_id in selected_ids:
        frame = _get_frame_from_session(session, current_frame_id)
        if frame is None:
            missing_frame_ids.append(current_frame_id)
            continue

        image_bytes, frame_read_error = _read_frame_bytes(frame)
        if frame_read_error:
            missing_files.append(current_frame_id)
            continue
        assert image_bytes is not None

        estimated_size = _estimate_base64_size(len(image_bytes)) + ESTIMATED_PER_FRAME_OVERHEAD
        if approx_response_bytes + estimated_size > SIZE_BUDGET_BYTES:
            truncated = True
            break

        image_blocks.append(Image(data=image_bytes, format="jpeg").to_image_content())
        returned_frames.append(
            {
                "frame_id": current_frame_id,
                "frame_index": frame["frame_index"],
                "timestamp_sec": frame["timestamp_sec"],
                "width": frame["width"],
                "height": frame["height"],
                "jpeg_bytes": frame["jpeg_bytes"],
                "file_path": frame["file_path"],
            }
        )
        approx_response_bytes += estimated_size

    if len(selected_ids) > len(returned_frames) + len(missing_frame_ids) + len(missing_files):
        truncated = True

    summary = (
        f"Returned {len(returned_frames)} frame image(s) from session {session_id}. "
        f"Requested {len(selected_ids)} frame(s)."
    )

    return ToolResult(
        content=[TextContent(type="text", text=summary), *image_blocks],
        structured_content={
            "ok": True,
            "server_version": SERVER_VERSION,
            "session_id": session_id,
            "requested_frame_count": len(selected_ids),
            "returned_frame_count": len(returned_frames),
            "approx_response_bytes": approx_response_bytes,
            "truncated": truncated,
            "missing_frame_ids": missing_frame_ids,
            "missing_files": missing_files,
            "frames": returned_frames,
        },
    )


@mcp.tool
def cleanup_visual_context(session_id: str) -> Dict[str, Any]:
    """Delete temporary frame files for a visual-context session."""
    removed = _remove_session(session_id)
    if not removed:
        return {
            "ok": False,
            "error": "Session not found.",
            "session_id": session_id,
            "server_version": SERVER_VERSION,
        }
    return {"ok": True, "session_id": session_id, "server_version": SERVER_VERSION}


if __name__ == "__main__":
    mcp.run()
