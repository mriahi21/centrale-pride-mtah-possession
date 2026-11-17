"""Video/audio probing with local tools only (OpenCV + the ffmpeg binary shipped by imageio-ffmpeg)."""
import re
import shutil
import subprocess

import poss  # noqa: F401


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("no ffmpeg available: pip install imageio-ffmpeg")
        return exe


def probe_video(path):
    """{frames, fps, width, height, has_audio, audio_desc, duration_s, audio_start_s} -- no pixel data kept."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False, "error": "cannot open video"}
    info = {"ok": True, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                       text=True, errors="replace")
    err = p.stderr
    audio = re.findall(r"Stream #\d+:\d+.*?Audio: ([^\n]+)", err)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+).*?start: ([\d.\-]+)", err)
    info["has_audio"] = bool(audio)
    info["audio_desc"] = audio[0].strip() if audio else ""
    if m:
        info["duration_s"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        info["audio_start_s"] = float(m.group(4))
    return info
