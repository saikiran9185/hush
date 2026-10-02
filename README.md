# Hush

**Turn any recording into sound layers you can see, hear and edit.**

Open a recording and Hush shows what's inside it: voice, car horns, traffic, wind, music and so on, plus exactly when each one happens. Click a sound, press **Extract**, and it becomes its own layer. Mute that layer and the sound is gone from the recording.

It runs entirely on your Mac. Nothing is uploaded.

## What you can do

**In Hush Studio (any recording: MP3, WAV, voice memo, video):**
- **See what's in it.** A sound map shows each sound type in its own row, lit up wherever it happens.
- **Pull a sound out.** Click a row, then **Extract**. That sound becomes its own layer.
- **Pull out anything by name.** Type "dog barking" or "fan", then **Extract**. Drag across the timeline first to search only that part.
- **Split any layer:**
  - **Lows · Mids · Highs**, by pitch range.
  - **Hits · Sustained**, which separates sudden sounds from steady ones.
  - **Loud · Quiet**, which separates the loud moments from the background.
  - **Drag a box** on the pitch view to cut out exactly that patch of sound.
- **Mix:** each layer has mute, solo and volume.
- **Export** every layer as a WAV, plus your mix.

**In DaVinci Resolve:** select a clip, then go to **Workspace › Scripts › Hush - Extract Layers**. Each sound Hush finds lands on its own audio track, in sync and named after the sound (Horns, Traffic, Wind…). The voice stays on **Voice & rest**. Mute a track to remove that sound. The original clip's audio is switched off, not deleted, and the clip stays orange while Hush works.

## Install

You need an Apple Silicon Mac, macOS 14 or later, and these tools:

```bash
xcode-select --install          # Swift compiler, if you don't have Xcode
brew install uv ffmpeg
```

Then:

```bash
git clone <this repo> Hush
cd Hush
./install.sh                     # the first run downloads the AI model (about 2 GB)
```

## Use

```bash
./studio.sh                      # opens Hush Studio in your browser
./studio.sh my-recording.m4a     # opens it with a recording
```

In Resolve, use the script under **Workspace › Scripts**. The installer puts it there.

## How it works

1. **Listening.** Apple's built-in sound classifier (SoundAnalysis) listens in 1-second steps and reports what it hears, for example "speech 96%, car horn 80%". That becomes the sound map. A minute of audio takes about a second, and nothing needs downloading.
2. **Pulling a sound out.** [SAM Audio](https://github.com/facebookresearch/sam-audio), Meta's model that separates sounds by name, runs only on the moments where that sound was found. It returns two things: the sound itself and everything else. The sound becomes a new layer, and "Everything else" loses it.
3. **Instant splits.** Ordinary signal processing (spectrogram masks) splits a layer by pitch range, by hits vs sustained sounds, by loudness, or by the box you drag. These parts add back up to the original exactly.
4. **Resolve.** Resolve's free version can't run Python or show windows from a script. So the Resolve script hands the clip over through a file Resolve writes itself, and macOS starts Hush. Hush maps and splits the clip, then the script places each layer on its own track as soon as it's ready.

```
recording ──► sound map (Apple, ~1 s/min) ──► click a sound ──► SAM Audio ──► new layer
                                         └──► drag / split ──► DSP masks ──► new layers
```

| File | What it is |
| --- | --- |
| `studio/server.py` + `studio/index.html` | Hush Studio: the visual editor |
| `studio/soundmap.swift` | The sound map (Apple's classifier) |
| `hush.py` | The engine: SAM Audio, memory safety, the Resolve job runner |
| `Hush - Extract Layers.lua` | The Resolve script |
| `ask.swift` | The small message panel Resolve uses for errors |
| `install.sh`, `studio.sh` | Install / uninstall, open Studio |

## Good to know

- **Phone recordings and voice memos work.** The AI works on any audio. A clean recording separates best. When two sounds are loud at the same moment and at the same pitch, the result can have artifacts, so listen with solo before you export.
- **On an 8 GB Mac it's safe but not instant.** Pulling a sound out takes about 3× the length of the moments it works on. Hush keeps its memory capped, pauses if the Mac runs low, and unloads the AI after 3 idle minutes.
- **The sound map names what a sound sounds like.** For something unusual, type its name or drag a box.
- **Licensing.** Hush's code is yours. The SAM Audio model weights are under Meta's [SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE). Read it before you sell anything built on them.

## Uninstall

```bash
./install.sh --uninstall         # your exported audio in ~/Movies/Hush is kept
```
