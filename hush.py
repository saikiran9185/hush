"""Hush: remove a sound you name from DaVinci Resolve clips, using SAM Audio on this Mac.

How it fits together:
  1. In Resolve you mark In/Out around the sound and run Workspace > Scripts > Hush - Remove a Sound.
     That script drops one request file per clip into ~/Library/Application Support/Hush/inbox.
  2. launchd sees the inbox change and runs this file. It asks what the sound is (the `ask` panel),
     removes it from the marked range with SAM Audio, and writes the clip to ~/Movies/Hush.
  3. A ".done" file tells the waiting Resolve script to put the clean clip on the timeline.

Without Resolve:
  python hush.py clip.mov --start 3 --end 12 --remove "car horn" --out clean.wav
"""

import argparse
import contextlib
import csv
import ctypes
import gc
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUPPORT = Path.home() / "Library/Application Support/Hush"
INBOX, STATUS, MEDIA = SUPPORT / "inbox", SUPPORT / "status", Path.home() / "Movies/Hush"
ASK = HERE / ".build/Hush Ask.app/Contents/MacOS/ask"
MODEL = "mlx-community/sam-audio-small-fp16"
SR = 48000
FFMPEG = next((p for p in ("/opt/homebrew/bin/ffmpeg", str(Path.home() / ".local/bin/ffmpeg"), "/usr/local/bin/ffmpeg")
               if os.path.exists(p)), "ffmpeg")

# <id>-<k>of<n>_s<clip start>_e<clip end>_a<marked start>_b<marked end>.csv, all in source seconds
REQUEST = re.compile(r"^(\d+)-(\d+)of(\d+)_s([\d.]+)_e([\d.]+)_a([\d.]+)_b([\d.]+)\.csv$")
MESSAGE = re.compile(r"^(\d+)-msg_(\w+)\.edl$")
MESSAGES = {
    "noclip": "Select a clip in Resolve (or mark In and Out over it), then run Hush again.",
    "retimed": "Hush can't clean clips with speed changes. Reset the clip speed and try again.",
}


# --- audio -----------------------------------------------------------------------------------

def read_audio(path, start, end):
    """Decode [start, end) seconds of any audio/video file to stereo float32 at 48 kHz."""
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-ss", f"{start:.6f}", "-to", f"{end:.6f}", "-i", path,
           "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
    if out.returncode:
        raise RuntimeError(f"Couldn't read the audio: {out.stderr.decode(errors='ignore').strip()[-200:]}")
    audio = np.frombuffer(out.stdout, np.float32).reshape(-1, 2)
    want = round((end - start) * SR)  # decoders are off by a few ms; Resolve needs the exact length
    return np.pad(audio, ((0, max(0, want - len(audio))), (0, 0)))[:want]


def write_wav(path, audio):
    import soundfile as sf

    tmp = path.with_name(f".{path.name}.part")  # the Resolve script must never see a half-written file
    sf.write(tmp, np.clip(audio, -1, 1), SR, subtype="PCM_24", format="WAV")
    os.replace(tmp, path)


# --- SAM Audio -------------------------------------------------------------------------------

def free_memory_percent():
    """macOS's own "memory free percentage" (what `memory_pressure` prints)."""
    value, size = ctypes.c_int(100), ctypes.c_size_t(4)
    ctypes.CDLL("libc.dylib").sysctlbyname(b"kern.memorystatus_level", ctypes.byref(value), ctypes.byref(size), None, 0)
    return value.value


def limit_memory():
    import mlx.core as mx
    import mlx_audio.sts.models.sam_audio.model as sam

    # SAM Audio wires up to the GPU's whole working set (5.7 GB on an 8 GB Mac). Wired memory can't be
    # paged out, and with Resolve open that froze the display. Keep it pageable and cap MLX at 40% of RAM.
    sam.wired_limit = lambda model: contextlib.nullcontext()
    mx.set_memory_limit(int(mx.device_info()["memory_size"] * 0.4))
    mx.set_cache_limit(256 << 20)


def load_model():
    from mlx_audio.sts import SAMAudio

    limit_memory()
    return SAMAudio.from_pretrained(MODEL)


def encode_prompt(prompt):
    """Turn the sound's name into SAM Audio's text features.

    Done before the main model loads, so the ~900 MB T5 encoder and SAM Audio never share memory.
    """
    import mlx.core as mx
    from mlx_audio.sts.models.sam_audio.config import T5EncoderConfig
    from mlx_audio.sts.models.sam_audio.text_encoder import T5TextEncoder

    limit_memory()
    text, mask = T5TextEncoder(T5EncoderConfig())([prompt])
    mx.eval(text, mask)
    gc.collect()
    mx.clear_cache()
    return text, mask


def separate(model, mono, text, chunk=2.0, overlap=0.5, on_chunk=None):
    """Split mono audio into (the encoded sound, everything else).

    Works in 2 s chunks with crossfades: memory grows with chunk length.
    """
    import mlx.core as mx

    text, mask = text
    n, size, fade = len(mono), int(chunk * SR), int(overlap * SR)
    starts = range(0, max(n - fade, 1), size - fade)
    target, rest, weight = (np.zeros(n, np.float32) for _ in range(3))
    for i, a in enumerate(starts):
        while free_memory_percent() < 12:  # never push the Mac into heavy swapping
            gc.collect()
            mx.clear_cache()
            time.sleep(0.5)
        b = min(a + size, n)
        mx.random.seed(42 + i)
        result = model.separate(mx.array(mono[a:b])[None, None], [""], ode_opt={"method": "midpoint", "step_size": 1 / 16},
                                ode_decode_chunk_size=10, _text_features=text, _text_mask=mask)
        w = np.ones(b - a, np.float32)  # crossfade the overlaps
        if a > 0:
            w[:fade] = np.linspace(0, 1, fade)
        if b < n:
            w[-fade:] = np.linspace(1, 0, fade)
        for out, part in ((target, result.target), (rest, result.residual)):
            out[a:b] += np.array(part[0]).reshape(-1)[: b - a] * w
        weight[a:b] += w
        mx.clear_cache()
        if on_chunk:
            on_chunk(i + 1, len(starts))
    return target / np.maximum(weight, 1e-6), rest / np.maximum(weight, 1e-6)


def clean(model, path, start, end, mark_a, mark_b, text, keep=False, pad=0.3):
    """Clean the marked part [mark_a, mark_b] of the clip [start, end]; the rest stays untouched.

    Returns stereo audio for the whole clip, so it can replace the original clip one-for-one.
    """
    audio = read_audio(path, start, end)
    a = max(0, round((mark_a - start - pad) * SR))
    b = min(len(audio), round((mark_b - start + pad) * SR))
    sound, rest = separate(model, audio[a:b].mean(axis=1), text)
    cleaned = sound if keep else rest

    mix = np.ones(b - a, np.float32)  # 30 ms fades where cleaned audio meets the untouched original
    ramp = min(int(0.03 * SR), (b - a) // 2)
    if a > 0:
        mix[:ramp] = np.linspace(0, 1, ramp)
    if b < len(audio):
        mix[-ramp:] = np.linspace(1, 0, ramp)
    out = audio.copy()
    out[a:b] = audio[a:b] * (1 - mix[:, None]) + cleaned[:, None] * mix[:, None]
    return out


# --- talking to the Resolve script -------------------------------------------------------------

def ask(*args):
    """Show the Hush panel. Returns (mode, name) or None if cancelled."""
    result = subprocess.run([str(ASK), *args], capture_output=True, text=True)
    mode, _, name = result.stdout.strip().partition("\t")
    return (mode, name) if result.returncode == 0 and name else None


def media_path(csv_file):
    """Resolve writes clip metadata as UTF-16 CSV; "Clip Directory" + "File Name" is the media file."""
    with open(csv_file, encoding="utf-16", newline="") as f:
        row = next(csv.DictReader(f))
    return os.path.join(row["Clip Directory"], row["File Name"])


def read_inbox(pending):
    """Collect request files into `pending` (id -> request); return the requests that are complete."""
    for f in sorted(INBOX.iterdir()):
        age = time.time() - f.stat().st_mtime
        if age > 60:  # the Resolve script gave up long ago
            f.unlink()
            continue
        if age < 0.2:  # let Resolve finish writing
            continue
        if m := REQUEST.match(f.name):
            req = pending.setdefault(m[1], {"clips": [], "messages": [], "seen": time.time()})
            req["total"] = int(m[3])
            s, e, a, b = map(float, m.group(4, 5, 6, 7))
            req["clips"].append({"k": int(m[2]), "path": media_path(f), "start": s, "end": e, "a": a, "b": b})
        elif m := MESSAGE.match(f.name):
            pending.setdefault(m[1], {"clips": [], "messages": [], "seen": time.time()})["messages"].append(m[2])
        f.unlink()
    ready = [rid for rid, r in pending.items()
             if len(r["clips"]) == r.get("total", -1) or time.time() - r["seen"] > 2]
    return [(rid, pending.pop(rid)) for rid in ready]


def handle(rid, req, model):
    """One request from Resolve. The .ack file tells the script we're on it; removing it means we're finished."""
    ack = STATUS / f"{rid}.ack"
    ack.touch()
    try:
        return work(rid, req, model)
    finally:
        time.sleep(1)  # let the script pick up the last .done first
        ack.unlink(missing_ok=True)


def work(rid, req, model):
    if not req["clips"]:
        ask("--message", "\n".join(MESSAGES.get(code, code) for code in req["messages"]))
        return model
    clips = sorted(req["clips"], key=lambda c: c["k"])
    marked = sum(c["b"] - c["a"] for c in clips)
    name = Path(clips[0]["path"]).stem if len(clips) == 1 else f"{len(clips)} clips"
    answer = ask(f"{name} · {marked:.1f} s")
    if not answer:
        (STATUS / f"{rid}.cancel").touch()
        return model
    mode, prompt = answer
    text = encode_prompt(prompt)
    model = model or load_model()
    for c in clips:
        tag = f"{rid}-{c['k']}"
        try:
            out = clean(model, c["path"], c["start"], c["end"], c["a"], c["b"], text, keep=mode == "keep")
            label = f"{Path(c['path']).stem} – {'only' if mode == 'keep' else 'no'} {prompt}".replace("/", "-")[:90]
            write_wav(MEDIA / f"{tag}__{label}.wav", out)
            (STATUS / f"{tag}.done").touch()
        except Exception as e:
            print(f"{tag}: {e}", file=sys.stderr)
            (STATUS / f"{tag}.failed").touch()
            text = "Not enough free memory. Close some apps and try again." if "memory" in str(e).lower() else str(e)
            ask("--message", f"Hush couldn't clean {Path(c['path']).name}:\n{text}")
    return model


def serve():
    """Run by launchd when the inbox changes: handle requests, then exit after 20 s of quiet."""
    os.nice(10)  # Resolve first
    for d in (INBOX, STATUS, MEDIA):
        d.mkdir(parents=True, exist_ok=True)
    for f in STATUS.iterdir():  # leftovers from a run that was killed
        if f.suffix == ".ack" or time.time() - f.stat().st_mtime > 86400:
            f.unlink()
    pending, model, quiet_since = {}, None, time.time()
    while time.time() - quiet_since < 20:
        ready = read_inbox(pending)
        for rid, req in ready:
            model = handle(rid, req, model)
        if ready or pending:
            quiet_since = time.time()
        time.sleep(0.3)


def main():
    if len(sys.argv) == 1:
        return serve()
    p = argparse.ArgumentParser(description="Remove a sound you name from an audio or video file.")
    p.add_argument("input")
    p.add_argument("--remove", help="the sound to remove, e.g. 'car horn'")
    p.add_argument("--keep", help="keep only this sound instead, e.g. 'person speaking'")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, help="defaults to the end of the file")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if not (args.remove or args.keep):
        p.error("say what to --remove (or --keep)")
    end = args.end
    if end is None:
        probe = subprocess.run([FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", args.input], capture_output=True, text=True)
        end = float(probe.stdout.strip())
    text = encode_prompt(args.keep or args.remove)
    out = clean(load_model(), args.input, args.start, end, args.start, end, text, keep=bool(args.keep))
    write_wav(Path(args.out).resolve(), out)


if __name__ == "__main__":
    main()
