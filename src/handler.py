"""RunPod Serverless handler for ProPainter video inpainting (caption removal)."""
import base64
import os
import subprocess
import sys
import tempfile
import time

import runpod

PROPAINTER_DIR = "/app/ProPainter"
sys.path.insert(0, PROPAINTER_DIR)

MODEL_LOADED = False


def ensure_model():
    global MODEL_LOADED
    if MODEL_LOADED:
        return
    for w in ["ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth"]:
        p = os.path.join(PROPAINTER_DIR, "weights", w)
        if not os.path.isfile(p):
            raise RuntimeError(f"Missing weight: {p}")
    MODEL_LOADED = True
    print("[worker] weights verified")


def handler(job):
    ensure_model()
    inp = job["input"]

    video_b64 = inp.get("video")
    mask_b64 = inp.get("mask")
    if not video_b64 or not mask_b64:
        return {"error": "Both 'video' and 'mask' are required (base64)."}

    neighbor_length = inp.get("neighbor_length", 10)
    ref_stride = inp.get("ref_stride", 10)
    raft_iter = inp.get("raft_iter", 20)
    subvideo_length = inp.get("subvideo_length", 80)
    fp16 = inp.get("fp16", True)

    with tempfile.TemporaryDirectory(prefix="pp_") as td:
        vid_path = os.path.join(td, "input.mp4")
        mask_dir = os.path.join(td, "masks")
        out_dir = os.path.join(td, "output")
        os.makedirs(mask_dir)
        os.makedirs(out_dir)

        with open(vid_path, "wb") as f:
            f.write(base64.b64decode(video_b64))
        print(f"[worker] video: {os.path.getsize(vid_path) / 1048576:.2f} MB")

        mask_png = base64.b64decode(mask_b64)

        # Get frame count
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", vid_path],
            capture_output=True, text=True,
        )
        try:
            n_frames = int(probe.stdout.strip())
        except ValueError:
            probe2 = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate,duration", "-of", "csv=p=0", vid_path],
                capture_output=True, text=True,
            )
            parts = probe2.stdout.strip().split(",")
            fps_str = parts[0] if parts else "30/1"
            dur_str = parts[1] if len(parts) > 1 else "5"
            num, den = fps_str.split("/") if "/" in fps_str else (fps_str, "1")
            n_frames = int(float(dur_str) * float(num) / float(den))

        print(f"[worker] {n_frames} frames, writing static mask copies")
        for i in range(n_frames):
            with open(os.path.join(mask_dir, f"{i:05d}.png"), "wb") as f:
                f.write(mask_png)

        t0 = time.time()
        cmd = [
            sys.executable,
            os.path.join(PROPAINTER_DIR, "inference_propainter.py"),
            "--video", vid_path,
            "--mask", mask_dir,
            "--output", out_dir,
            "--neighbor_length", str(neighbor_length),
            "--ref_stride", str(ref_stride),
            "--raft_iter", str(raft_iter),
            "--subvideo_length", str(subvideo_length),
            "--save_frames", "False",
        ]
        if fp16:
            cmd.append("--fp16")

        print("[worker] running ProPainter...")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0

        if proc.returncode != 0:
            print(f"[worker] STDERR: {proc.stderr[-2000:]}")
            return {"error": f"ProPainter failed (exit {proc.returncode}): {proc.stderr[-500:]}"}

        print(f"[worker] inference done in {dt:.1f}s")

        result_path = None
        for root, _, files in os.walk(out_dir):
            for fn in files:
                if fn.endswith(".mp4"):
                    result_path = os.path.join(root, fn)
                    break
            if result_path:
                break

        if not result_path or not os.path.isfile(result_path):
            return {"error": "ProPainter produced no output video"}

        print(f"[worker] output: {os.path.getsize(result_path) / 1048576:.2f} MB")
        with open(result_path, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode("ascii")

    return {"video": result_b64, "inference_time": round(dt, 1), "frames": n_frames}


runpod.serverless.start({"handler": handler})
