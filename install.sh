#!/bin/bash
# Install Hush:  ./install.sh        Remove it:  ./install.sh --uninstall
set -euo pipefail
cd "$(dirname "$0")"
HUSH=$(pwd)
SCRIPTS="$HOME/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility"
AGENT="$HOME/Library/LaunchAgents/com.hush.resolve.plist"

if [ "${1:-}" = "--uninstall" ]; then
  launchctl bootout "gui/$(id -u)" "$AGENT" 2>/dev/null || true
  rm -rf "$AGENT" "$SCRIPTS/Hush - Extract Layers.lua" "$HOME/Library/Application Support/Hush" .venv .build
  echo "Hush removed. Your cleaned audio in ~/Movies/Hush was kept."
  exit
fi

command -v uv >/dev/null || { echo "Hush needs uv. Install it with: brew install uv"; exit 1; }
command -v ffmpeg >/dev/null || [ -x "$HOME/.local/bin/ffmpeg" ] || { echo "Hush needs ffmpeg. Install it with: brew install ffmpeg"; exit 1; }

echo "1/4  Python and the SAM Audio model (the first time downloads about 2 GB)"
uv venv --quiet --allow-existing .venv --python 3.12
uv pip install --quiet --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "
from huggingface_hub import snapshot_download as get
get('mlx-community/sam-audio-small-fp16')
get('t5-base', allow_patterns=['*.json', 'model.safetensors', 'spiece.model'])" 2>/dev/null

echo "2/4  The sound map and the message panel"
APP=".build/Hush Ask.app"
rm -rf "$APP" && mkdir -p "$APP/Contents/MacOS"
swiftc -O ask.swift -o "$APP/Contents/MacOS/ask"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>com.hush.ask</string>
  <key>CFBundleName</key><string>Hush</string>
  <key>CFBundleExecutable</key><string>ask</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSUIElement</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$APP" 2>/dev/null
swiftc -O studio/soundmap.swift -o .build/soundmap  # Studio's sound map

echo "3/4  The Resolve script"
mkdir -p "$SCRIPTS"
rm -f "$SCRIPTS/Hush - Remove a Sound.lua"  # older version
cp "Hush - Extract Layers.lua" "$SCRIPTS/"

echo "4/4  The background job (macOS starts it only when Resolve sends work)"
mkdir -p "$HOME/Library/Application Support/Hush/inbox" "$HOME/Library/LaunchAgents"
cat > "$AGENT" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.hush.resolve</string>
  <key>ProgramArguments</key><array><string>$HUSH/.venv/bin/python</string><string>$HUSH/hush.py</string></array>
  <key>WatchPaths</key><array><string>$HOME/Library/Application Support/Hush/inbox</string></array>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/Hush.log</string>
</dict></plist>
PLIST
launchctl bootout "gui/$(id -u)" "$AGENT" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$AGENT"

echo
echo "Done!"
echo "  Studio (any recording):  ./studio.sh"
echo "  In Resolve:              select a clip, then Workspace > Scripts > Hush - Extract Layers"
