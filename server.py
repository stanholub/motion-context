from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
import math
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
ANALYZE_MAX_RESPONSE_BYTES = 8_500_000
ANALYZE_SIZE_BUDGET_BYTES = 8_000_000
ESTIMATED_PER_FRAME_OVERHEAD = 220
VISION_TOKENS_PER_PIXEL_DIVISOR = 512
ESTIMATED_PER_FRAME_METADATA_TOKENS = 48
MAX_ESTIMATED_TOKENS_HARD_CAP = 75_000
DEFAULT_GET_VISUAL_FRAMES_MAX_ESTIMATED_TOKENS = 24_000
DEFAULT_SESSION_MAX_ESTIMATED_TOKENS = 75_000
MIN_TIMESTAMP_GAP_SEC = 0.18
JPEG_QUALITY_FALLBACKS = [90, 82, 74, 66, 58, 50]
TIMESTAMP_CANDIDATE_MULTIPLIER = 3
TIMESTAMP_CANDIDATE_EXTRA = 24
MAX_TIMESTAMP_CANDIDATES = 1200
CHANGE_SAMPLE_MIN_FPS = 2.0
CHANGE_SAMPLE_MAX_FPS = 8.0
CHANGE_SCORE_MIN_RATIO = 0.25
CHANGE_TIMESTAMP_MIN_GAP_SEC = 0.35
MAX_CHANGE_SAMPLES = 6000
MAX_UNCERTAIN_INTERVALS = 8
SESSION_TTL_SEC = 3600
MAX_SESSIONS = 20
TEMP_ROOT_DIR = os.path.join(tempfile.gettempdir(), "mcp-video-context")
SERVER_VERSION = "ref-sessions-v11"
AUTO_OVERVIEW_MODE_MIN_DURATION_SEC = 90.0
MAX_ANALYZE_DURATION_SEC = 150.0
ANALYZE_RESOLUTION_PRESETS: Dict[str, Dict[str, int]] = {
    "precise": {
        "width": 1280,
        "height": 720,
        "default_max_frames": 24,
        "max_frames_cap": 180,
        "default_max_estimated_tokens": 42_000,
    },
    "overview": {
        "width": 640,
        "height": 360,
        "default_max_frames": 60,
        "max_frames_cap": 150,
        "default_max_estimated_tokens": 16_000,
    },
}
ANALYZE_RESOLUTION_MODE_ALIASES: Dict[str, str] = {
    "flow": "overview",
    "balanced": "overview",
    "detail": "precise",
    "long": "overview",
    "low_detail": "overview",
    "high_detail": "precise",
}
MAX_RECOMMENDED_GAP_BY_PROFILE: Dict[str, Tuple[float, float, float]] = {
    # (<=30s, <=120s, >120s) recommended max gap in seconds
    "precise": (0.75, 1.15, 1.6),
    "overview": (1.25, 1.8, 2.5),
    "session": (1.0, 1.6, 2.2),
}
TARGET_FPS_BY_MODE_AND_INTENSITY: Dict[str, Dict[str, float]] = {
    "precise": {"low": 2.0, "medium": 2.5, "high": 3.0},
    "overview": {"low": 1.0, "medium": 1.0, "high": 1.0},
}
MAX_FPS_BY_MODE: Dict[str, float] = {
    "precise": 3.0,
    "overview": 1.0,
}
ANALYSIS_INTENSITY_HIGH_KEYWORDS = [
    "every",
    "all actions",
    "comprehensive",
    "step-by-step",
    "step by step",
    "click",
    "scroll",
    "hover",
    "selection",
    "deselect",
    "timeline",
    "sequence",
    "debug",
    "regression",
    "exactly",
]
ANALYSIS_INTENSITY_LOW_KEYWORDS = [
    "brief",
    "short summary",
    "high-level",
    "high level",
    "quick summary",
]
AUTO_MIN_FRAMES_BY_INTENSITY: Dict[str, Dict[str, int]] = {
    "precise": {"low": 12, "medium": 18, "high": 24},
    "overview": {"low": 8, "medium": 12, "high": 16},
}
AUTO_MIN_TOKENS_BY_INTENSITY: Dict[str, Dict[str, int]] = {
    "precise": {"low": 28_000, "medium": 42_000, "high": 58_000},
    "overview": {"low": 10_000, "medium": 16_000, "high": 22_000},
}

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


def _probe_video_duration(video_path: str) -> Optional[float]:
    cap, _, duration_sec, error = _open_capture(video_path)
    if cap is not None:
        cap.release()
    if error:
        return None
    return duration_sec


def _resize_to_target(frame, target_width: int, target_height: int):
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None

    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def _encode_jpeg_bytes(frame, jpeg_quality: int = JPEG_QUALITY) -> Optional[bytes]:
    effective_quality = min(max(int(jpeg_quality), 1), 100)
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            effective_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE,
            1,
        ],
    )
    if not success:
        return None
    return buffer.tobytes()


def _estimate_base64_size(byte_len: int) -> int:
    return ((byte_len + 2) // 3) * 4


def _estimate_image_tokens(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return 1
    pixels = width * height
    return max(1, (pixels + VISION_TOKENS_PER_PIXEL_DIVISOR - 1) // VISION_TOKENS_PER_PIXEL_DIVISOR)


def _estimate_frame_tokens(width: int, height: int) -> int:
    return _estimate_image_tokens(width, height) + ESTIMATED_PER_FRAME_METADATA_TOKENS


def _infer_analysis_intensity(question: str) -> str:
    lowered = (question or "").lower()
    if any(keyword in lowered for keyword in ANALYSIS_INTENSITY_HIGH_KEYWORDS):
        return "high"
    if any(keyword in lowered for keyword in ANALYSIS_INTENSITY_LOW_KEYWORDS):
        return "low"
    return "medium"


def _ensure_temp_root() -> None:
    os.makedirs(TEMP_ROOT_DIR, exist_ok=True)


def _normalize_session_token_budget(raw_budget: Any) -> int:
    try:
        budget = int(raw_budget)
    except Exception:
        budget = DEFAULT_SESSION_MAX_ESTIMATED_TOKENS
    return min(max(1, budget), MAX_ESTIMATED_TOKENS_HARD_CAP)


def _normalize_session_tokens_used(raw_used: Any, budget: int) -> int:
    try:
        used = int(raw_used)
    except Exception:
        used = 0
    return min(max(0, used), budget)


def _get_session_token_usage(session_id: str) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    with SESSION_LOCK:
        session = SESSION_STORE.get(session_id)
        if session is None:
            return None, "Session not found."

        budget = _normalize_session_token_budget(session.get("estimated_tokens_budget"))
        used = _normalize_session_tokens_used(session.get("estimated_tokens_used"), budget)
        session["estimated_tokens_budget"] = budget
        session["estimated_tokens_used"] = used

    return {
        "estimated_tokens_budget": budget,
        "estimated_tokens_used": used,
        "estimated_tokens_remaining": budget - used,
    }, None


def _consume_session_tokens(session_id: str, estimated_tokens: int) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    if estimated_tokens < 0:
        return None, "estimated_tokens must be greater than or equal to 0."

    with SESSION_LOCK:
        session = SESSION_STORE.get(session_id)
        if session is None:
            return None, "Session not found."

        budget = _normalize_session_token_budget(session.get("estimated_tokens_budget"))
        used = _normalize_session_tokens_used(session.get("estimated_tokens_used"), budget)
        remaining = budget - used

        if estimated_tokens > remaining:
            session["estimated_tokens_budget"] = budget
            session["estimated_tokens_used"] = used
            return (
                {
                    "estimated_tokens_budget": budget,
                    "estimated_tokens_used": used,
                    "estimated_tokens_remaining": remaining,
                },
                "Session estimated token budget reached. "
                "Run get_visual_context again to start a fresh session budget.",
            )

        used += estimated_tokens
        session["estimated_tokens_budget"] = budget
        session["estimated_tokens_used"] = used

    return {
        "estimated_tokens_budget": budget,
        "estimated_tokens_used": used,
        "estimated_tokens_remaining": budget - used,
    }, None


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
        "estimated_tokens_budget": DEFAULT_SESSION_MAX_ESTIMATED_TOKENS,
        "estimated_tokens_used": 0,
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

    frame_index = int(timestamp_sec * fps)
    if frame_index < 0:
        frame_index = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if ok:
        return frame

    # Some codecs fail exactly at the tail; probe a few previous frame indices.
    if fps > 0:
        for fallback_delta in (1, 2, 3, 5, 8, 13):
            fallback_index = frame_index - fallback_delta
            if fallback_index < 0:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_index)
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


def _uniform_timestamps(duration_sec: Optional[float], max_frames: int = MAX_FRAMES) -> List[float]:
    if duration_sec is None or duration_sec <= 0:
        return [i * UNIFORM_INTERVAL_SEC for i in range(max_frames)]

    if max_frames <= 1:
        return [0.0]

    target_interval = duration_sec / float(max_frames - 1)
    step = min(UNIFORM_INTERVAL_SEC, max(target_interval, 0.2))

    times: List[float] = []
    t = 0.0
    while t <= duration_sec + 1e-3:
        times.append(t)
        t += step

    if not times:
        times = [0.0]

    if times[-1] < duration_sec:
        times.append(duration_sec)

    return times


def _dedupe_sorted_timestamps(values: List[float], min_gap_sec: float = MIN_TIMESTAMP_GAP_SEC) -> List[float]:
    if not values:
        return []

    ordered = sorted(max(0.0, float(value)) for value in values)
    deduped: List[float] = []
    for current in ordered:
        if not deduped or current - deduped[-1] >= min_gap_sec:
            deduped.append(current)
    return deduped


def _clip_timestamps(values: Optional[List[float]], max_duration_sec: Optional[float]) -> List[float]:
    if not values:
        return []

    if max_duration_sec is None or max_duration_sec <= 0:
        return _dedupe_sorted_timestamps(values)

    clipped = [min(max(0.0, float(value)), float(max_duration_sec)) for value in values]
    return _dedupe_sorted_timestamps(clipped)


def _change_timestamps(
    video_path: str,
    fps: float,
    duration_sec: Optional[float],
    target_count: int,
    max_duration_sec: Optional[float] = None,
) -> List[float]:
    if target_count <= 0:
        return []

    cap, _, _, error = _open_capture(video_path)
    if error or cap is None:
        return []

    sample_fps = min(CHANGE_SAMPLE_MAX_FPS, max(CHANGE_SAMPLE_MIN_FPS, fps / 4.0))
    step_frames = max(1, int(round(fps / sample_fps)))

    scored: List[Tuple[float, float]] = []
    prev_small = None
    frame_index = 0
    sampled_frames = 0

    try:
        while sampled_frames < MAX_CHANGE_SAMPLES:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % step_frames != 0:
                frame_index += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            timestamp_sec = frame_index / fps if fps > 0 else float(sampled_frames)
            if max_duration_sec is not None and timestamp_sec > max_duration_sec + 1e-6:
                break

            if prev_small is not None:
                diff = cv2.absdiff(small, prev_small)
                score = float(diff.mean())
                scored.append((timestamp_sec, score))

            prev_small = small
            frame_index += 1
            sampled_frames += 1
    finally:
        cap.release()

    if not scored:
        return []

    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    top_score = ranked[0][1]
    min_score = top_score * CHANGE_SCORE_MIN_RATIO
    selected: List[float] = []
    max_candidates = max(target_count * 3, target_count + 6)

    for timestamp_sec, score in ranked[:max_candidates]:
        if score < min_score and len(selected) >= max(4, target_count // 3):
            break
        if all(abs(timestamp_sec - existing) >= CHANGE_TIMESTAMP_MIN_GAP_SEC for existing in selected):
            selected.append(timestamp_sec)
            if len(selected) >= target_count:
                break

    clamp_duration = duration_sec
    if max_duration_sec is not None and max_duration_sec > 0:
        if clamp_duration is None:
            clamp_duration = max_duration_sec
        else:
            clamp_duration = min(clamp_duration, max_duration_sec)

    if clamp_duration is not None and clamp_duration > 0:
        selected = [min(max(0.0, timestamp), clamp_duration) for timestamp in selected]

    return _dedupe_sorted_timestamps(selected, min_gap_sec=CHANGE_TIMESTAMP_MIN_GAP_SEC)


def _coverage_diagnostics(
    timestamps: List[float],
    duration_sec: Optional[float],
    profile: str = "session",
) -> Dict[str, Any]:
    ordered = sorted(max(0.0, float(ts)) for ts in timestamps)
    if not ordered:
        known_duration = float(duration_sec) if duration_sec and duration_sec > 0 else None
        return {
            "coverage_level": "low",
            "max_gap_sec": None,
            "avg_gap_sec": None,
            "recommended_max_gap_sec": None,
            "uncertain_intervals": [],
            "uncertain_duration_sec": round(known_duration, 3) if known_duration is not None else None,
            "coverage_percentage": 0.0 if known_duration is not None else None,
            "tail_gap_sec": round(known_duration, 3) if known_duration is not None else None,
        }

    effective_duration = float(duration_sec) if duration_sec and duration_sec > 0 else ordered[-1]
    if effective_duration < ordered[-1]:
        effective_duration = ordered[-1]

    threshold_profile = str(profile).strip().lower()
    short_gap, medium_gap, long_gap = MAX_RECOMMENDED_GAP_BY_PROFILE.get(
        threshold_profile,
        MAX_RECOMMENDED_GAP_BY_PROFILE["session"],
    )

    if effective_duration <= 30:
        recommended_max_gap_sec = short_gap
    elif effective_duration <= 120:
        recommended_max_gap_sec = medium_gap
    else:
        recommended_max_gap_sec = long_gap

    intervals: List[Tuple[float, float]] = []
    previous = 0.0
    for current in ordered:
        intervals.append((previous, current))
        previous = current
    intervals.append((previous, effective_duration))

    gaps = [max(0.0, end - start) for start, end in intervals]
    max_gap_sec = max(gaps) if gaps else 0.0
    avg_gap_sec = sum(gaps) / len(gaps) if gaps else 0.0

    uncertain: List[Dict[str, float]] = []
    uncertain_duration_sec = 0.0
    for start, end in intervals:
        gap = max(0.0, end - start)
        if gap > recommended_max_gap_sec:
            uncertain_duration_sec += gap
            if len(uncertain) < MAX_UNCERTAIN_INTERVALS:
                uncertain.append(
                    {
                        "start_sec": round(start, 3),
                        "end_sec": round(end, 3),
                        "gap_sec": round(gap, 3),
                    }
                )

    if not uncertain:
        coverage_level = "high"
    elif len(uncertain) <= 2:
        coverage_level = "medium"
    else:
        coverage_level = "low"

    coverage_percentage = 100.0
    if effective_duration > 0:
        coverage_percentage = max(0.0, min(100.0, 100.0 * (1.0 - (uncertain_duration_sec / effective_duration))))

    tail_gap_sec = max(0.0, effective_duration - ordered[-1])

    return {
        "coverage_level": coverage_level,
        "max_gap_sec": round(max_gap_sec, 3),
        "avg_gap_sec": round(avg_gap_sec, 3),
        "recommended_max_gap_sec": round(recommended_max_gap_sec, 3),
        "uncertain_intervals": uncertain,
        "uncertain_duration_sec": round(uncertain_duration_sec, 3),
        "coverage_percentage": round(coverage_percentage, 2),
        "tail_gap_sec": round(tail_gap_sec, 3),
    }


def _build_coverage_timestamps(
    scene_timestamps: Optional[List[float]],
    change_timestamps: Optional[List[float]],
    duration_sec: Optional[float],
    target_count: int,
) -> Tuple[List[float], str]:
    uniform = _dedupe_sorted_timestamps(_uniform_timestamps(duration_sec, max_frames=max(target_count * 2, target_count)))
    scene_only = _dedupe_sorted_timestamps(scene_timestamps or [])
    change_only = _dedupe_sorted_timestamps(change_timestamps or [], min_gap_sec=CHANGE_TIMESTAMP_MIN_GAP_SEC)
    anchors: List[float] = [0.0]
    if duration_sec is not None and duration_sec > 0:
        anchors.append(float(duration_sec))
    anchors = _dedupe_sorted_timestamps(anchors, min_gap_sec=0.01)

    if not scene_only and not change_only:
        with_anchors = _dedupe_sorted_timestamps(uniform + anchors, min_gap_sec=0.01)
        return _select_evenly(with_anchors, target_count), "uniform"

    selected = _dedupe_sorted_timestamps(scene_only + change_only + anchors, min_gap_sec=0.01)
    pool = [
        candidate
        for candidate in uniform
        if all(abs(candidate - existing) >= MIN_TIMESTAMP_GAP_SEC for existing in selected)
    ]
    need = max(0, target_count - len(selected))
    if need > 0 and pool:
        selected.extend(_select_evenly(pool, need))

    if len(selected) < target_count:
        for candidate in uniform:
            if len(selected) >= target_count:
                break
            selected.append(candidate)

    selected = _dedupe_sorted_timestamps(selected, min_gap_sec=0.01)

    if len(selected) > target_count:
        seed_set = {round(ts, 3) for ts in scene_only + change_only + anchors}
        non_seed = [ts for ts in selected if round(ts, 3) not in seed_set]
        keep = _dedupe_sorted_timestamps(scene_only + change_only + anchors, min_gap_sec=0.01)
        need = max(0, target_count - len(keep))
        if need > 0:
            keep.extend(_select_evenly(non_seed, need))
        selected = _dedupe_sorted_timestamps(keep, min_gap_sec=0.01)

    if len(selected) < target_count:
        existing = _dedupe_sorted_timestamps(selected, min_gap_sec=0.01)
        for candidate in uniform:
            if len(existing) >= target_count:
                break
            if all(abs(candidate - current) >= 0.01 for current in existing):
                existing.append(candidate)
        selected = _dedupe_sorted_timestamps(existing, min_gap_sec=0.01)

    selected = _select_evenly(selected, min(target_count, len(selected)))

    seed_keys = {round(ts, 3) for ts in scene_only + change_only + anchors}
    method_parts: List[str] = []
    if scene_only:
        method_parts.append("scene")
    if change_only:
        method_parts.append("change")
    if any(round(ts, 3) not in seed_keys for ts in selected):
        method_parts.append("uniform")
    if not method_parts:
        method_parts = ["uniform"]

    return selected, "+".join(method_parts)


def _coverage_priority_index_order(count: int) -> List[int]:
    if count <= 0:
        return []
    if count == 1:
        return [0]

    order: List[int] = [0, count - 1]
    seen = {0, count - 1}
    intervals: List[Tuple[int, int]] = [(0, count - 1)]

    while intervals:
        intervals.sort(key=lambda item: item[1] - item[0], reverse=True)
        left, right = intervals.pop(0)
        if right - left <= 1:
            continue

        mid = (left + right) // 2
        if mid in seen:
            alternatives = [mid + 1, mid - 1]
            replacement = next(
                (
                    candidate
                    for candidate in alternatives
                    if left < candidate < right and candidate not in seen
                ),
                None,
            )
            if replacement is None:
                continue
            mid = replacement

        order.append(mid)
        seen.add(mid)
        intervals.append((left, mid))
        intervals.append((mid, right))

    if len(order) < count:
        for idx in range(count):
            if idx in seen:
                continue
            order.append(idx)
            seen.add(idx)

    return order


def _collection_order_timestamps(
    candidate_timestamps: List[float],
    target_count: int,
) -> List[float]:
    ordered = _dedupe_sorted_timestamps(candidate_timestamps, min_gap_sec=0.001)
    if not ordered or target_count <= 0:
        return []

    index_order = _coverage_priority_index_order(len(ordered))
    primary_limit = min(target_count, len(ordered))
    primary_indices = index_order[:primary_limit]
    primary_set = set(primary_indices)
    backup_indices = [idx for idx in index_order if idx not in primary_set]
    final_indices = primary_indices + backup_indices
    return [ordered[idx] for idx in final_indices]


def _sanitize_collection_timestamps(
    timestamps: List[float],
    duration_sec: Optional[float],
    fps: float,
) -> List[float]:
    if not timestamps:
        return []

    max_seek_sec: Optional[float] = None
    if duration_sec is not None and duration_sec > 0:
        frame_margin = (1.0 / fps) if fps > 0 else 0.04
        max_seek_sec = max(0.0, float(duration_sec) - frame_margin)

    sanitized: List[float] = []
    seen = set()
    for timestamp_sec in timestamps:
        value = max(0.0, float(timestamp_sec))
        if max_seek_sec is not None:
            value = min(value, max_seek_sec)
        key = round(value, 6)
        if key in seen:
            continue
        seen.add(key)
        sanitized.append(value)

    return sanitized


def _estimate_frame_payload_size_bytes(jpeg_bytes_len: int) -> int:
    return _estimate_base64_size(jpeg_bytes_len) + ESTIMATED_PER_FRAME_OVERHEAD


def _frame_payload_size(frame: Dict[str, Any]) -> int:
    return _estimate_frame_payload_size_bytes(int(frame["jpeg_bytes"]))


def _capture_encoded_frame(
    cap: cv2.VideoCapture,
    fps: float,
    timestamp_sec: float,
    target_width: int,
    target_height: int,
    jpeg_quality: int,
) -> Optional[Dict[str, Any]]:
    frame = _read_frame_at(cap, timestamp_sec, fps)
    if frame is None:
        return None

    resized = _resize_to_target(frame, target_width=target_width, target_height=target_height)
    if resized is None:
        return None

    jpeg_bytes = _encode_jpeg_bytes(resized, jpeg_quality=jpeg_quality)
    if jpeg_bytes is None:
        return None

    return {
        "timestamp_sec": round(float(timestamp_sec), 3),
        "width": target_width,
        "height": target_height,
        "jpeg_bytes": len(jpeg_bytes),
        "jpeg_data": jpeg_bytes,
    }


def _has_nearby_timestamp(frames: List[Dict[str, Any]], timestamp_sec: float, tolerance_sec: float) -> bool:
    return any(abs(float(frame["timestamp_sec"]) - timestamp_sec) <= tolerance_sec for frame in frames)


def _select_replacement_index_for_required_frame(frames: List[Dict[str, Any]]) -> Optional[int]:
    if len(frames) <= 2:
        return None

    indexed = sorted(
        [(idx, float(frame["timestamp_sec"])) for idx, frame in enumerate(frames)],
        key=lambda item: item[1],
    )
    if len(indexed) <= 2:
        return None

    best_idx: Optional[int] = None
    best_density = float("inf")
    for position in range(1, len(indexed) - 1):
        current_idx, current_ts = indexed[position]
        _, prev_ts = indexed[position - 1]
        _, next_ts = indexed[position + 1]
        left_gap = max(0.0, current_ts - prev_ts)
        right_gap = max(0.0, next_ts - current_ts)
        density = min(left_gap, right_gap)
        if density < best_density:
            best_density = density
            best_idx = current_idx

    return best_idx


def _enforce_end_frame(
    cap: cv2.VideoCapture,
    fps: float,
    frames: List[Dict[str, Any]],
    approx_bytes: int,
    duration_sec: Optional[float],
    max_frames: int,
    target_width: int,
    target_height: int,
    jpeg_quality: int,
    size_budget_bytes: int,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    if duration_sec is None or duration_sec <= 0:
        return frames, approx_bytes, False

    tail_tolerance = max(0.25, (1.5 / fps) if fps > 0 else 0.25)
    target_tail_timestamp = max(0.0, float(duration_sec) - ((1.0 / fps) if fps > 0 else 0.04))
    if _has_nearby_timestamp(frames, target_tail_timestamp, tail_tolerance):
        return frames, approx_bytes, False

    required = _capture_encoded_frame(
        cap=cap,
        fps=fps,
        timestamp_sec=target_tail_timestamp,
        target_width=target_width,
        target_height=target_height,
        jpeg_quality=jpeg_quality,
    )
    if required is None:
        return frames, approx_bytes, False

    required_size = _frame_payload_size(required)
    updated_frames = list(frames)
    updated_bytes = approx_bytes

    if len(updated_frames) < max_frames and updated_bytes + required_size <= size_budget_bytes:
        updated_frames.append(required)
        updated_bytes += required_size
        return updated_frames, updated_bytes, True

    replace_index = _select_replacement_index_for_required_frame(updated_frames)
    if replace_index is None:
        return frames, approx_bytes, False

    replaced_size = _frame_payload_size(updated_frames[replace_index])
    new_total = updated_bytes - replaced_size + required_size
    if new_total > size_budget_bytes:
        return frames, approx_bytes, False

    updated_frames[replace_index] = required
    return updated_frames, new_total, True


def _collect_frames(
    cap: cv2.VideoCapture,
    fps: float,
    timestamps: List[float],
    max_frames: int = MAX_FRAMES,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    jpeg_quality: int = JPEG_QUALITY,
    size_budget_bytes: int = SIZE_BUDGET_BYTES,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
) -> Tuple[List[Dict[str, Any]], int, bool, int]:
    frames: List[Dict[str, Any]] = []
    total_size = 0
    truncated = False
    skipped_for_size_budget = 0

    for timestamp_sec in timestamps:
        if len(frames) >= max_frames:
            break

        encoded = _capture_encoded_frame(
            cap=cap,
            fps=fps,
            timestamp_sec=timestamp_sec,
            target_width=target_width,
            target_height=target_height,
            jpeg_quality=jpeg_quality,
        )
        if encoded is None:
            continue

        estimated_size = _frame_payload_size(encoded)
        if total_size + estimated_size > size_budget_bytes:
            truncated = True
            skipped_for_size_budget += 1
            continue

        frames.append(encoded)
        total_size += estimated_size

    if total_size > max_response_bytes:
        truncated = True

    return frames, total_size, truncated, skipped_for_size_budget


def _extract_representative_frames(
    video_path: str,
    max_frames: int,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    max_frames_cap: int = MAX_FRAMES,
    max_estimated_tokens: Optional[int] = None,
    ensure_end_frame: bool = True,
    coverage_profile: str = "session",
    max_duration_sec: Optional[float] = None,
    size_budget_bytes: int = SIZE_BUDGET_BYTES,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not os.path.exists(video_path):
        return None, "Video file not found."

    if not os.path.isfile(video_path):
        return None, "Video path is not a file."

    if max_frames <= 0:
        return None, "max_frames must be greater than 0."

    if target_width <= 0 or target_height <= 0:
        return None, "target resolution must be greater than 0."

    if max_frames_cap <= 0:
        return None, "max_frames_cap must be greater than 0."
    if size_budget_bytes <= 0:
        return None, "size_budget_bytes must be greater than 0."
    if max_response_bytes <= 0:
        return None, "max_response_bytes must be greater than 0."
    if max_duration_sec is not None and max_duration_sec <= 0:
        return None, "max_duration_sec must be greater than 0."

    effective_max_estimated_tokens: Optional[int] = None
    if max_estimated_tokens is not None:
        if max_estimated_tokens <= 0:
            return None, "max_estimated_tokens must be greater than 0."
        effective_max_estimated_tokens = min(max_estimated_tokens, MAX_ESTIMATED_TOKENS_HARD_CAP)

    base_effective_max_frames = min(max_frames, max_frames_cap)
    estimated_tokens_per_frame = _estimate_frame_tokens(target_width, target_height)
    token_limited = False

    if effective_max_estimated_tokens is None:
        effective_max_frames = base_effective_max_frames
    else:
        token_frame_cap = effective_max_estimated_tokens // estimated_tokens_per_frame
        if token_frame_cap < 1:
            return (
                None,
                "max_estimated_tokens is too low for requested resolution. "
                "Increase it or lower resolution_mode.",
            )
        effective_max_frames = min(base_effective_max_frames, token_frame_cap)
        token_limited = effective_max_frames < base_effective_max_frames

    candidate_target_count = min(
        MAX_TIMESTAMP_CANDIDATES,
        max(
            effective_max_frames,
            effective_max_frames * TIMESTAMP_CANDIDATE_MULTIPLIER,
            effective_max_frames + TIMESTAMP_CANDIDATE_EXTRA,
        ),
    )

    cap, fps, source_duration_sec, error = _open_capture(video_path)
    if error or cap is None or fps is None:
        return None, error or "Failed to open video."

    analysis_duration_sec = source_duration_sec
    duration_limited = False
    if max_duration_sec is not None:
        if analysis_duration_sec is None:
            analysis_duration_sec = max_duration_sec
        elif analysis_duration_sec > max_duration_sec:
            analysis_duration_sec = max_duration_sec
            duration_limited = True

    try:
        method = "uniform"
        scene_status = "skipped"

        scene_timestamps, scene_status = _scene_timestamps(video_path, analysis_duration_sec, fps)
        scene_timestamps = _clip_timestamps(scene_timestamps, analysis_duration_sec)
        change_timestamps = _change_timestamps(
            video_path=video_path,
            fps=fps,
            duration_sec=analysis_duration_sec,
            target_count=max(candidate_target_count, effective_max_frames + 4),
            max_duration_sec=analysis_duration_sec,
        )
        change_timestamps = _clip_timestamps(change_timestamps, analysis_duration_sec)
        timestamps, method = _build_coverage_timestamps(
            scene_timestamps=scene_timestamps,
            change_timestamps=change_timestamps,
            duration_sec=analysis_duration_sec,
            target_count=candidate_target_count,
        )
        collection_timestamps = _collection_order_timestamps(
            candidate_timestamps=timestamps,
            target_count=effective_max_frames,
        )
        collection_timestamps = _sanitize_collection_timestamps(
            timestamps=collection_timestamps,
            duration_sec=analysis_duration_sec,
            fps=fps,
        )

        quality_candidates = [JPEG_QUALITY]
        quality_candidates.extend(fallback for fallback in JPEG_QUALITY_FALLBACKS if fallback < JPEG_QUALITY)

        best_frames_with_bytes: List[Dict[str, Any]] = []
        best_approx_bytes = 0
        best_truncated = False
        best_skipped_for_size_budget = 0
        best_quality = JPEG_QUALITY
        best_count = -1
        best_tail_gap = float("inf")

        for candidate_quality in quality_candidates:
            frames_with_bytes, approx_bytes, truncated, skipped_for_size_budget = _collect_frames(
                cap,
                fps,
                collection_timestamps,
                max_frames=effective_max_frames,
                target_width=target_width,
                target_height=target_height,
                jpeg_quality=candidate_quality,
                size_budget_bytes=size_budget_bytes,
                max_response_bytes=max_response_bytes,
            )

            frame_count = len(frames_with_bytes)
            candidate_tail_gap = (
                max(0.0, float(analysis_duration_sec) - float(max(frame["timestamp_sec"] for frame in frames_with_bytes)))
                if frames_with_bytes and analysis_duration_sec is not None and analysis_duration_sec > 0
                else float("inf")
            )
            should_replace = False
            if frame_count > best_count:
                should_replace = True
            elif frame_count == best_count:
                if candidate_tail_gap + 1e-6 < best_tail_gap:
                    should_replace = True
                elif candidate_tail_gap == best_tail_gap and candidate_quality > best_quality:
                    should_replace = True
            if should_replace:
                best_frames_with_bytes = frames_with_bytes
                best_approx_bytes = approx_bytes
                best_truncated = truncated
                best_skipped_for_size_budget = skipped_for_size_budget
                best_quality = candidate_quality
                best_count = frame_count
                best_tail_gap = candidate_tail_gap

            if frame_count >= effective_max_frames:
                break
            if not truncated:
                break

        frames_with_bytes = sorted(
            best_frames_with_bytes,
            key=lambda item: float(item.get("timestamp_sec", 0.0)),
        )
        approx_bytes = best_approx_bytes
        truncated = best_truncated
        jpeg_quality_used = best_quality
        end_frame_forced = False

        if ensure_end_frame:
            frames_with_bytes, approx_bytes, end_frame_forced = _enforce_end_frame(
                cap=cap,
                fps=fps,
                frames=frames_with_bytes,
                approx_bytes=approx_bytes,
                duration_sec=analysis_duration_sec,
                max_frames=effective_max_frames,
                target_width=target_width,
                target_height=target_height,
                jpeg_quality=jpeg_quality_used,
                size_budget_bytes=size_budget_bytes,
            )
            if end_frame_forced:
                frames_with_bytes = sorted(
                    frames_with_bytes,
                    key=lambda item: float(item.get("timestamp_sec", 0.0)),
                )
                truncated = True
        else:
            end_frame_forced = False
    finally:
        cap.release()

    if not frames_with_bytes:
        return None, "Failed to extract frames from video."

    frame_timestamps = [float(frame["timestamp_sec"]) for frame in frames_with_bytes]
    coverage = _coverage_diagnostics(frame_timestamps, analysis_duration_sec, profile=coverage_profile)

    return (
        {
            "method": method,
            "scene_detection": scene_status,
            "duration_sec": analysis_duration_sec,
            "source_duration_sec": source_duration_sec,
            "duration_limited": duration_limited,
            "approx_bytes": approx_bytes,
            "truncated": truncated,
            "frames": frames_with_bytes,
            "effective_max_frames": effective_max_frames,
            "candidate_timestamp_count": len(timestamps),
            "candidate_target_count": candidate_target_count,
            "collection_timestamp_count": len(collection_timestamps),
            "target_width": target_width,
            "target_height": target_height,
            "max_frames_cap": max_frames_cap,
            "base_effective_max_frames": base_effective_max_frames,
            "effective_max_estimated_tokens": effective_max_estimated_tokens,
            "estimated_tokens_per_frame": estimated_tokens_per_frame,
            "approx_estimated_tokens": estimated_tokens_per_frame * len(frames_with_bytes),
            "token_limited": token_limited,
            "jpeg_quality_used": jpeg_quality_used,
            "ensure_end_frame": ensure_end_frame,
            "end_frame_forced": end_frame_forced,
            "skipped_for_size_budget": best_skipped_for_size_budget,
            "scene_timestamp_count": len(scene_timestamps or []),
            "change_timestamp_count": len(change_timestamps),
            "coverage": coverage,
        },
        None,
    )


def _resolve_analyze_settings(
    resolution_mode: str,
    max_frames: Optional[int],
    max_estimated_tokens: Optional[int],
    question: str,
    auto_tune: bool,
    duration_sec_hint: Optional[float],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw_mode = str(resolution_mode).strip().lower()
    if not raw_mode:
        raw_mode = "auto"

    mode_auto_selected = False
    mode_selection_reason: Optional[str] = None
    requested_mode = raw_mode

    if raw_mode == "auto":
        mode_auto_selected = True
        if duration_sec_hint is not None and duration_sec_hint >= AUTO_OVERVIEW_MODE_MIN_DURATION_SEC:
            mode = "overview"
            mode_selection_reason = (
                f"auto-selected overview mode for duration >= {AUTO_OVERVIEW_MODE_MIN_DURATION_SEC:.0f}s"
            )
        else:
            mode = "precise"
            mode_selection_reason = (
                f"auto-selected precise mode for duration < {AUTO_OVERVIEW_MODE_MIN_DURATION_SEC:.0f}s"
                if duration_sec_hint is not None
                else "auto-selected precise mode (duration unknown)"
            )
    else:
        mode = ANALYZE_RESOLUTION_MODE_ALIASES.get(raw_mode, raw_mode)
        if raw_mode in ANALYZE_RESOLUTION_MODE_ALIASES:
            mode_selection_reason = f"mode alias '{raw_mode}' mapped to '{mode}'"

    preset = ANALYZE_RESOLUTION_PRESETS.get(mode)
    if preset is None:
        canonical_modes = ", ".join(sorted(ANALYZE_RESOLUTION_PRESETS.keys()))
        alias_modes = ", ".join(sorted(ANALYZE_RESOLUTION_MODE_ALIASES.keys()))
        return (
            None,
            "resolution_mode must be one of: auto, "
            f"{canonical_modes} (legacy aliases: {alias_modes}).",
        )

    intensity = _infer_analysis_intensity(question)
    duration_sec_planned = (
        min(float(duration_sec_hint), MAX_ANALYZE_DURATION_SEC)
        if duration_sec_hint is not None and duration_sec_hint > 0
        else None
    )
    target_fps = float(TARGET_FPS_BY_MODE_AND_INTENSITY[mode][intensity])
    max_fps = float(MAX_FPS_BY_MODE[mode])

    duration_based_default_frames = (
        max(1, int(math.ceil(duration_sec_planned * target_fps)))
        if duration_sec_planned is not None
        else None
    )
    duration_based_max_frames = (
        max(1, int(math.ceil(duration_sec_planned * max_fps)))
        if duration_sec_planned is not None
        else None
    )

    default_max_frames = (
        int(duration_based_default_frames)
        if duration_based_default_frames is not None
        else int(preset["default_max_frames"])
    )
    max_frames_cap = (
        int(duration_based_max_frames)
        if duration_based_max_frames is not None
        else int(preset["max_frames_cap"])
    )

    if max_frames is None:
        requested_max_frames_raw = default_max_frames
    else:
        if max_frames <= 0:
            return None, "max_frames must be greater than 0."
        requested_max_frames_raw = max_frames

    default_max_estimated_tokens = int(preset["default_max_estimated_tokens"])
    if max_estimated_tokens is None:
        requested_max_estimated_tokens_raw = default_max_estimated_tokens
    else:
        if max_estimated_tokens <= 0:
            return None, "max_estimated_tokens must be greater than 0."
        requested_max_estimated_tokens_raw = max_estimated_tokens

    auto_min_frames = int(AUTO_MIN_FRAMES_BY_INTENSITY[mode][intensity])
    auto_min_tokens = int(AUTO_MIN_TOKENS_BY_INTENSITY[mode][intensity])

    auto_adjusted_max_frames = False
    auto_adjusted_max_estimated_tokens = False
    duration_adjusted_max_frames = False
    duration_adjusted_max_estimated_tokens = False

    requested_max_frames = requested_max_frames_raw
    requested_max_estimated_tokens = requested_max_estimated_tokens_raw
    estimated_tokens_per_frame = _estimate_frame_tokens(
        width=int(preset["width"]),
        height=int(preset["height"]),
    )

    if requested_max_frames > max_frames_cap:
        requested_max_frames = max_frames_cap
        duration_adjusted_max_frames = True

    # When caller does not specify a token budget, provision enough tokens for
    # the frame target so duration-driven defaults are actually reachable.
    if max_estimated_tokens is None:
        min_tokens_for_requested_frames = int(math.ceil(requested_max_frames * estimated_tokens_per_frame * 1.1))
        if requested_max_estimated_tokens < min_tokens_for_requested_frames:
            requested_max_estimated_tokens = min_tokens_for_requested_frames
            duration_adjusted_max_estimated_tokens = True

    if auto_tune:
        tuned_frame_floor = min(max_frames_cap, auto_min_frames)
        if requested_max_frames < tuned_frame_floor:
            requested_max_frames = tuned_frame_floor
            auto_adjusted_max_frames = True

        min_tokens_for_frames = int(requested_max_frames * estimated_tokens_per_frame * 1.15)
        tuned_token_floor = max(auto_min_tokens, min_tokens_for_frames)
        if requested_max_estimated_tokens < tuned_token_floor:
            requested_max_estimated_tokens = tuned_token_floor
            auto_adjusted_max_estimated_tokens = True

    effective_max_estimated_tokens = min(requested_max_estimated_tokens, MAX_ESTIMATED_TOKENS_HARD_CAP)

    return (
        {
            "resolution_mode": mode,
            "requested_resolution_mode": requested_mode,
            "mode_auto_selected": mode_auto_selected,
            "mode_selection_reason": mode_selection_reason,
            "duration_sec_hint": duration_sec_hint,
            "duration_sec_planned": duration_sec_planned,
            "analysis_duration_cap_sec": MAX_ANALYZE_DURATION_SEC,
            "target_fps": target_fps,
            "max_fps": max_fps,
            "target_width": int(preset["width"]),
            "target_height": int(preset["height"]),
            "default_max_frames": default_max_frames,
            "requested_max_frames": requested_max_frames,
            "requested_max_frames_raw": requested_max_frames_raw,
            "max_frames_cap": max_frames_cap,
            "default_max_estimated_tokens": default_max_estimated_tokens,
            "requested_max_estimated_tokens": requested_max_estimated_tokens,
            "requested_max_estimated_tokens_raw": requested_max_estimated_tokens_raw,
            "effective_max_estimated_tokens": effective_max_estimated_tokens,
            "analysis_intensity": intensity,
            "auto_tune": auto_tune,
            "duration_adjusted_max_frames": duration_adjusted_max_frames,
            "duration_adjusted_max_estimated_tokens": duration_adjusted_max_estimated_tokens,
            "auto_adjusted_max_frames": auto_adjusted_max_frames,
            "auto_adjusted_max_estimated_tokens": auto_adjusted_max_estimated_tokens,
            "auto_min_frames": auto_min_frames,
            "auto_min_estimated_tokens": auto_min_tokens,
        },
        None,
    )


@mcp.tool
def get_visual_context(video_path: str) -> ToolResult:
    """Extract key frame references from a video and create a temporary frame session.

    Uses scene detection first, then falls back to uniform sampling if needed.
    Returns only lightweight metadata and frame references. Fetch actual image
    bytes with `get_visual_frame`.
    """
    extraction, error = _extract_representative_frames(
        video_path=video_path,
        max_frames=MAX_FRAMES,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        max_frames_cap=MAX_FRAMES,
        max_estimated_tokens=None,
        ensure_end_frame=True,
        coverage_profile="session",
    )
    if error:
        return ToolResult(
            content=[TextContent(type="text", text=error or "Failed to open video.")],
            structured_content={
                "ok": False,
                "error": error or "Failed to open video.",
                "video_path": video_path,
            },
        )
    assert extraction is not None

    session = _create_session(
        video_path=video_path,
        method=str(extraction["method"]),
        scene_status=str(extraction["scene_detection"]),
        duration_sec=extraction["duration_sec"],
        frames_with_bytes=list(extraction["frames"]),
        approx_bytes=int(extraction["approx_bytes"]),
        truncated=bool(extraction["truncated"]),
    )

    summary = (
        f"Extracted {len(session['frames'])} frame reference(s). "
        "Call get_visual_frame for one frame or get_visual_frames for a batch."
    )
    session_tokens = {
        "session_estimated_tokens_budget": int(session.get("estimated_tokens_budget", DEFAULT_SESSION_MAX_ESTIMATED_TOKENS)),
        "session_estimated_tokens_used": int(session.get("estimated_tokens_used", 0)),
        "session_estimated_tokens_remaining": int(session.get("estimated_tokens_budget", DEFAULT_SESSION_MAX_ESTIMATED_TOKENS))
        - int(session.get("estimated_tokens_used", 0)),
    }

    return ToolResult(
        content=[TextContent(type="text", text=summary)],
        structured_content={
            "ok": True,
            "server_version": SERVER_VERSION,
            "session_id": session["session_id"],
            "session_expires_at": int(session["expires_at"]),
            "session_ttl_sec": SESSION_TTL_SEC,
            "method": extraction["method"],
            "scene_detection": extraction["scene_detection"],
            "frame_count": len(session["frames"]),
            "candidate_timestamp_count": extraction["candidate_timestamp_count"],
            "candidate_target_count": extraction["candidate_target_count"],
            "collection_timestamp_count": extraction["collection_timestamp_count"],
            "scene_timestamp_count": extraction["scene_timestamp_count"],
            "change_timestamp_count": extraction["change_timestamp_count"],
            "coverage_level": extraction["coverage"]["coverage_level"],
            "max_gap_sec": extraction["coverage"]["max_gap_sec"],
            "avg_gap_sec": extraction["coverage"]["avg_gap_sec"],
            "recommended_max_gap_sec": extraction["coverage"]["recommended_max_gap_sec"],
            "uncertain_intervals": extraction["coverage"]["uncertain_intervals"],
            "uncertain_duration_sec": extraction["coverage"]["uncertain_duration_sec"],
            "coverage_percentage": extraction["coverage"]["coverage_percentage"],
            "tail_gap_sec": extraction["coverage"]["tail_gap_sec"],
            "approx_bytes": extraction["approx_bytes"],
            "truncated": extraction["truncated"],
            "jpeg_quality_used": extraction["jpeg_quality_used"],
            "ensure_end_frame": extraction["ensure_end_frame"],
            "end_frame_forced": extraction["end_frame_forced"],
            "skipped_for_size_budget": extraction["skipped_for_size_budget"],
            "duration_sec": (
                round(float(extraction["duration_sec"]), 3)
                if extraction["duration_sec"]
                else None
            ),
            **session_tokens,
            "frames": session["frames"],
        },
    )


@mcp.tool
def analyze_video(
    video_path: str,
    question: str = "Summarize the key visual events in this video.",
    max_frames: Optional[int] = None,
    resolution_mode: str = "auto",
    max_estimated_tokens: Optional[int] = None,
    strict_evidence: bool = True,
    auto_tune: bool = True,
    ensure_end_frame: bool = True,
) -> ToolResult:
    """One-shot video context for existing chats.

    Returns representative frame images and timestamp metadata in one call,
    so clients can answer immediately without manual multi-step orchestration.
    """
    settings, settings_error = _resolve_analyze_settings(
        resolution_mode=resolution_mode,
        max_frames=max_frames,
        max_estimated_tokens=max_estimated_tokens,
        question=question,
        auto_tune=auto_tune,
        duration_sec_hint=_probe_video_duration(video_path),
    )
    if settings_error:
        return ToolResult(
            content=[TextContent(type="text", text=settings_error)],
            structured_content={
                "ok": False,
                "error": settings_error,
                "video_path": video_path,
            },
        )
    assert settings is not None

    extraction, error = _extract_representative_frames(
        video_path=video_path,
        max_frames=int(settings["requested_max_frames"]),
        target_width=int(settings["target_width"]),
        target_height=int(settings["target_height"]),
        max_frames_cap=int(settings["max_frames_cap"]),
        max_estimated_tokens=int(settings["effective_max_estimated_tokens"]),
        ensure_end_frame=bool(ensure_end_frame),
        coverage_profile=str(settings["resolution_mode"]),
        max_duration_sec=MAX_ANALYZE_DURATION_SEC,
        size_budget_bytes=ANALYZE_SIZE_BUDGET_BYTES,
        max_response_bytes=ANALYZE_MAX_RESPONSE_BYTES,
    )
    if error:
        return ToolResult(
            content=[TextContent(type="text", text=error)],
            structured_content={
                "ok": False,
                "error": error,
                "video_path": video_path,
            },
        )
    assert extraction is not None

    frame_images: List[Any] = []
    frame_refs: List[Dict[str, Any]] = []

    for idx, frame in enumerate(extraction["frames"]):
        frame_id = f"frame_{idx}"
        frame_images.append(Image(data=frame["jpeg_data"], format="jpeg").to_image_content())
        frame_refs.append(
            {
                "frame_id": frame_id,
                "frame_index": idx,
                "timestamp_sec": frame["timestamp_sec"],
                "width": frame["width"],
                "height": frame["height"],
                "jpeg_bytes": frame["jpeg_bytes"],
            }
        )

    summary_parts = [
        f"Question: {question}",
        (
            f"Resolution mode: {settings['resolution_mode']} "
            f"({settings['target_width']}x{settings['target_height']})."
        ),
        (
            f"Returned {len(frame_refs)} representative frame image(s) "
            f"using {extraction['method']} sampling."
        ),
        "Use timestamps as citations in the answer.",
    ]
    if strict_evidence:
        summary_parts.append(
            "Strict evidence mode: only describe what is visible in returned frames. "
            "Do not infer unseen UI actions, backend results, or navigation state."
        )
    if settings["mode_auto_selected"] and settings["mode_selection_reason"]:
        summary_parts.append(f"Mode selection: {settings['mode_selection_reason']}.")
    elif settings["requested_resolution_mode"] != settings["resolution_mode"]:
        summary_parts.append(
            f"Mode alias '{settings['requested_resolution_mode']}' resolved to '{settings['resolution_mode']}'."
        )
    if extraction["duration_limited"] and extraction["source_duration_sec"] is not None:
        summary_parts.append(
            "Duration cap applied: analyzed first "
            f"{MAX_ANALYZE_DURATION_SEC:.0f}s of {round(float(extraction['source_duration_sec']), 3)}s source video."
        )
    if max_frames is None:
        summary_parts.append(
            f"max_frames defaulted to {settings['default_max_frames']} for this mode."
        )
    if settings["duration_adjusted_max_frames"]:
        summary_parts.append(
            f"max_frames capped to {settings['max_frames_cap']} for {settings['resolution_mode']} mode."
        )
    if max_estimated_tokens is None:
        if settings["duration_adjusted_max_estimated_tokens"]:
            summary_parts.append(
                "Token guard raised mode default budget "
                f"({settings['default_max_estimated_tokens']}) to "
                f"{settings['requested_max_estimated_tokens']} for duration-based frame density."
            )
        else:
            summary_parts.append(
                "Token guard used mode default budget "
                f"({settings['default_max_estimated_tokens']} tokens)."
            )
    if int(settings["requested_max_estimated_tokens"]) > MAX_ESTIMATED_TOKENS_HARD_CAP:
        summary_parts.append(
            f"max_estimated_tokens capped to hard limit ({MAX_ESTIMATED_TOKENS_HARD_CAP})."
        )
    if settings["auto_tune"]:
        summary_parts.append(
            f"Auto-tune profile: {settings['analysis_intensity']}."
        )
    if settings["auto_adjusted_max_frames"]:
        summary_parts.append(
            "Auto-tune increased max_frames from "
            f"{settings['requested_max_frames_raw']} to {settings['requested_max_frames']}."
        )
    if settings["auto_adjusted_max_estimated_tokens"]:
        summary_parts.append(
            "Auto-tune increased max_estimated_tokens from "
            f"{settings['requested_max_estimated_tokens_raw']} to "
            f"{settings['requested_max_estimated_tokens']}."
        )
    if extraction["token_limited"]:
        summary_parts.append(
            "Frame count was reduced by token budget."
        )
    if int(extraction["jpeg_quality_used"]) < JPEG_QUALITY:
        summary_parts.append(
            f"JPEG quality reduced to {extraction['jpeg_quality_used']} to preserve coverage."
        )
    if extraction["coverage"]["uncertain_intervals"]:
        summary_parts.append(
            f"Coverage uncertainty detected in {len(extraction['coverage']['uncertain_intervals'])} interval(s); "
            "mark those periods as unknown."
        )
    coverage_percentage = extraction["coverage"]["coverage_percentage"]
    if coverage_percentage is not None:
        summary_parts.append(f"Coverage estimate: {coverage_percentage}%.")
    tail_gap = extraction["coverage"]["tail_gap_sec"]
    recommended_gap = extraction["coverage"]["recommended_max_gap_sec"]
    if (
        tail_gap is not None
        and recommended_gap is not None
        and tail_gap > recommended_gap
    ):
        summary_parts.append(
            f"Tail coverage is uncertain (last gap {tail_gap}s > recommended {recommended_gap}s)."
        )
    if len(frame_refs) < int(extraction["effective_max_frames"]):
        summary_parts.append(
            f"Returned fewer frames than requested target ({len(frame_refs)}/{extraction['effective_max_frames']})."
        )
    if extraction["skipped_for_size_budget"] > 0:
        summary_parts.append(
            f"Skipped {extraction['skipped_for_size_budget']} candidate frame(s) due to response-size budget."
        )
    if extraction["end_frame_forced"]:
        summary_parts.append(
            "End-frame guard replaced/added a frame near video end to reduce tail uncertainty."
        )

    return ToolResult(
        content=[TextContent(type="text", text=" ".join(summary_parts)), *frame_images],
        structured_content={
            "ok": True,
            "server_version": SERVER_VERSION,
            "video_path": video_path,
            "question": question,
            "resolution_mode": settings["resolution_mode"],
            "requested_resolution_mode": settings["requested_resolution_mode"],
            "mode_auto_selected": settings["mode_auto_selected"],
            "mode_selection_reason": settings["mode_selection_reason"],
            "duration_sec_hint": (
                round(float(settings["duration_sec_hint"]), 3)
                if settings["duration_sec_hint"] is not None
                else None
            ),
            "duration_sec_planned": (
                round(float(settings["duration_sec_planned"]), 3)
                if settings["duration_sec_planned"] is not None
                else None
            ),
            "analysis_duration_cap_sec": settings["analysis_duration_cap_sec"],
            "target_fps": settings["target_fps"],
            "max_fps": settings["max_fps"],
            "target_width": settings["target_width"],
            "target_height": settings["target_height"],
            "method": extraction["method"],
            "scene_detection": extraction["scene_detection"],
            "strict_evidence": strict_evidence,
            "auto_tune": settings["auto_tune"],
            "ensure_end_frame": extraction["ensure_end_frame"],
            "end_frame_forced": extraction["end_frame_forced"],
            "analysis_intensity": settings["analysis_intensity"],
            "duration_sec": (
                round(float(extraction["duration_sec"]), 3)
                if extraction["duration_sec"]
                else None
            ),
            "source_duration_sec": (
                round(float(extraction["source_duration_sec"]), 3)
                if extraction["source_duration_sec"] is not None
                else None
            ),
            "duration_limited": extraction["duration_limited"],
            "default_max_frames": settings["default_max_frames"],
            "requested_max_frames": settings["requested_max_frames"],
            "requested_max_frames_raw": settings["requested_max_frames_raw"],
            "max_frames_cap": settings["max_frames_cap"],
            "default_max_estimated_tokens": settings["default_max_estimated_tokens"],
            "requested_max_estimated_tokens": settings["requested_max_estimated_tokens"],
            "requested_max_estimated_tokens_raw": settings["requested_max_estimated_tokens_raw"],
            "effective_max_estimated_tokens": settings["effective_max_estimated_tokens"],
            "duration_adjusted_max_frames": settings["duration_adjusted_max_frames"],
            "duration_adjusted_max_estimated_tokens": settings["duration_adjusted_max_estimated_tokens"],
            "auto_adjusted_max_frames": settings["auto_adjusted_max_frames"],
            "auto_adjusted_max_estimated_tokens": settings["auto_adjusted_max_estimated_tokens"],
            "auto_min_frames": settings["auto_min_frames"],
            "auto_min_estimated_tokens": settings["auto_min_estimated_tokens"],
            "effective_max_frames": extraction["effective_max_frames"],
            "returned_frame_count": len(frame_refs),
            "candidate_timestamp_count": extraction["candidate_timestamp_count"],
            "candidate_target_count": extraction["candidate_target_count"],
            "collection_timestamp_count": extraction["collection_timestamp_count"],
            "scene_timestamp_count": extraction["scene_timestamp_count"],
            "change_timestamp_count": extraction["change_timestamp_count"],
            "coverage_level": extraction["coverage"]["coverage_level"],
            "max_gap_sec": extraction["coverage"]["max_gap_sec"],
            "avg_gap_sec": extraction["coverage"]["avg_gap_sec"],
            "recommended_max_gap_sec": extraction["coverage"]["recommended_max_gap_sec"],
            "uncertain_intervals": extraction["coverage"]["uncertain_intervals"],
            "uncertain_duration_sec": extraction["coverage"]["uncertain_duration_sec"],
            "coverage_percentage": extraction["coverage"]["coverage_percentage"],
            "tail_gap_sec": extraction["coverage"]["tail_gap_sec"],
            "approx_bytes": extraction["approx_bytes"],
            "jpeg_quality_used": extraction["jpeg_quality_used"],
            "skipped_for_size_budget": extraction["skipped_for_size_budget"],
            "estimated_tokens_per_frame": extraction["estimated_tokens_per_frame"],
            "approx_estimated_tokens": extraction["approx_estimated_tokens"],
            "token_limited": extraction["token_limited"],
            "truncated": extraction["truncated"],
            "frames": frame_refs,
        },
    )


@mcp.prompt(name="quick_video_review")
def quick_video_review(
    video_path: str,
    question: str = "Summarize the key visual events in this video.",
    resolution_mode: str = "auto",
    max_estimated_tokens: Optional[int] = None,
    strict_evidence: bool = True,
    auto_tune: bool = True,
    ensure_end_frame: bool = True,
) -> str:
    """Prompt shortcut for one-shot video analysis in MCP clients."""
    max_tokens_note = (
        f"- max_estimated_tokens: {max_estimated_tokens}\n"
        if max_estimated_tokens is not None
        else "- max_estimated_tokens: omit to use mode default\n"
    )
    return (
        "Use the `analyze_video` tool with these arguments and then answer:\n"
        f"- video_path: {video_path}\n"
        f"- question: {question}\n"
        f"- resolution_mode: {resolution_mode}\n"
        "- max_frames: omit to use mode default\n"
        f"{max_tokens_note}"
        f"- strict_evidence: {strict_evidence}\n"
        f"- auto_tune: {auto_tune}\n"
        f"- ensure_end_frame: {ensure_end_frame}\n"
        "Include timestamps as evidence. If evidence is missing between timestamps, explicitly say unknown."
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

    estimated_tokens = _estimate_frame_tokens(int(frame["width"]), int(frame["height"]))
    session_tokens, token_error = _consume_session_tokens(session_id, estimated_tokens)
    if token_error:
        assert session_tokens is not None
        return ToolResult(
            content=[TextContent(type="text", text=token_error)],
            structured_content={
                "ok": False,
                "error": token_error,
                "session_id": session_id,
                "frame_id": frame_id,
                "estimated_tokens_for_frame": estimated_tokens,
                "session_estimated_tokens_budget": session_tokens["estimated_tokens_budget"],
                "session_estimated_tokens_used": session_tokens["estimated_tokens_used"],
                "session_estimated_tokens_remaining": session_tokens["estimated_tokens_remaining"],
            },
        )
    assert session_tokens is not None

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
            "estimated_tokens_for_frame": estimated_tokens,
            "session_estimated_tokens_budget": session_tokens["estimated_tokens_budget"],
            "session_estimated_tokens_used": session_tokens["estimated_tokens_used"],
            "session_estimated_tokens_remaining": session_tokens["estimated_tokens_remaining"],
            "file_path": frame["file_path"],
        },
    )


@mcp.tool
def get_visual_frames(
    session_id: str,
    frame_ids: Optional[List[str]] = None,
    max_frames: int = MAX_FRAMES,
    max_estimated_tokens: int = DEFAULT_GET_VISUAL_FRAMES_MAX_ESTIMATED_TOKENS,
) -> ToolResult:
    """Return multiple JPEG frame images from a visual-context session.

    If `frame_ids` is omitted, returns all session frames up to `max_frames`
    and the response-size/token budget.
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

    if max_estimated_tokens <= 0:
        return ToolResult(
            content=[TextContent(type="text", text="max_estimated_tokens must be greater than 0.")],
            structured_content={
                "ok": False,
                "error": "max_estimated_tokens must be greater than 0.",
                "session_id": session_id,
            },
        )

    session_tokens, session_tokens_error = _get_session_token_usage(session_id)
    if session_tokens_error:
        return ToolResult(
            content=[TextContent(type="text", text=session_tokens_error)],
            structured_content={
                "ok": False,
                "error": session_tokens_error,
                "session_id": session_id,
            },
        )
    assert session_tokens is not None

    effective_max_estimated_tokens = min(
        max_estimated_tokens,
        MAX_ESTIMATED_TOKENS_HARD_CAP,
        session_tokens["estimated_tokens_remaining"],
    )
    if effective_max_estimated_tokens <= 0:
        error = (
            "Session estimated token budget reached. "
            "Run get_visual_context again to start a fresh session budget."
        )
        return ToolResult(
            content=[TextContent(type="text", text=error)],
            structured_content={
                "ok": False,
                "error": error,
                "session_id": session_id,
                "requested_max_estimated_tokens": max_estimated_tokens,
                "effective_max_estimated_tokens": 0,
                "session_estimated_tokens_budget": session_tokens["estimated_tokens_budget"],
                "session_estimated_tokens_used": session_tokens["estimated_tokens_used"],
                "session_estimated_tokens_remaining": session_tokens["estimated_tokens_remaining"],
            },
        )

    requested_ids = _resolve_frame_ids(session, frame_ids)
    selected_ids = requested_ids[: min(max_frames, MAX_FRAMES)]

    image_blocks: List[Any] = []
    returned_frames: List[Dict[str, Any]] = []
    missing_frame_ids: List[str] = []
    missing_files: List[str] = []
    approx_response_bytes = 0
    approx_estimated_tokens = 0
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
        estimated_tokens = _estimate_frame_tokens(int(frame["width"]), int(frame["height"]))

        if approx_response_bytes + estimated_size > SIZE_BUDGET_BYTES:
            truncated = True
            break
        if approx_estimated_tokens + estimated_tokens > effective_max_estimated_tokens:
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
        approx_estimated_tokens += estimated_tokens

    if len(selected_ids) > len(returned_frames) + len(missing_frame_ids) + len(missing_files):
        truncated = True

    summary = (
        f"Returned {len(returned_frames)} frame image(s) from session {session_id}. "
        f"Requested {len(selected_ids)} frame(s)."
    )
    if max_estimated_tokens > MAX_ESTIMATED_TOKENS_HARD_CAP:
        summary += f" max_estimated_tokens capped to {MAX_ESTIMATED_TOKENS_HARD_CAP}."

    updated_session_tokens, consume_error = _consume_session_tokens(session_id, approx_estimated_tokens)
    if consume_error:
        assert updated_session_tokens is not None
        return ToolResult(
            content=[TextContent(type="text", text=consume_error)],
            structured_content={
                "ok": False,
                "error": consume_error,
                "session_id": session_id,
                "requested_max_estimated_tokens": max_estimated_tokens,
                "effective_max_estimated_tokens": effective_max_estimated_tokens,
                "approx_estimated_tokens": approx_estimated_tokens,
                "session_estimated_tokens_budget": updated_session_tokens["estimated_tokens_budget"],
                "session_estimated_tokens_used": updated_session_tokens["estimated_tokens_used"],
                "session_estimated_tokens_remaining": updated_session_tokens["estimated_tokens_remaining"],
            },
        )
    assert updated_session_tokens is not None

    return ToolResult(
        content=[TextContent(type="text", text=summary), *image_blocks],
        structured_content={
            "ok": True,
            "server_version": SERVER_VERSION,
            "session_id": session_id,
            "requested_frame_count": len(selected_ids),
            "returned_frame_count": len(returned_frames),
            "approx_response_bytes": approx_response_bytes,
            "requested_max_estimated_tokens": max_estimated_tokens,
            "effective_max_estimated_tokens": effective_max_estimated_tokens,
            "approx_estimated_tokens": approx_estimated_tokens,
            "session_estimated_tokens_budget": updated_session_tokens["estimated_tokens_budget"],
            "session_estimated_tokens_used": updated_session_tokens["estimated_tokens_used"],
            "session_estimated_tokens_remaining": updated_session_tokens["estimated_tokens_remaining"],
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
