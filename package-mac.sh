#!/usr/bin/env bash
# package-mac.sh — build bsnobs, create BSnoBS.app bundle, and package as DMG
set -euo pipefail
cd "$(dirname "$0")"

APP_NAME="BSnoBS"
BIN_NAME="bsnobs"
BUNDLE="dist/${APP_NAME}.app"
RESOURCES="${BUNDLE}/Contents/Resources"
MACOS="${BUNDLE}/Contents/MacOS"

# ── 1. Build ─────────────────────────────────────────────────────────────────
echo "==> Building release binary…"
cargo build --release

# ── 2. Assemble .app bundle ───────────────────────────────────────────────────
echo "==> Assembling ${BUNDLE}…"
rm -rf "${BUNDLE}"
mkdir -p "${MACOS}" "${RESOURCES}"

cp "target/release/${BIN_NAME}" "${MACOS}/${BIN_NAME}"
cp Info.plist                   "${BUNDLE}/Contents/Info.plist"

# ── 3. Icon ───────────────────────────────────────────────────────────────────
if [ -f icon.png ]; then
    echo "==> Converting icon.png → icon.icns…"
    ICONSET="$(mktemp -d)/icon.iconset"
    mkdir -p "${ICONSET}"
    sips -z 16   16   icon.png --out "${ICONSET}/icon_16x16.png"      >/dev/null
    sips -z 32   32   icon.png --out "${ICONSET}/icon_16x16@2x.png"   >/dev/null
    sips -z 32   32   icon.png --out "${ICONSET}/icon_32x32.png"       >/dev/null
    sips -z 64   64   icon.png --out "${ICONSET}/icon_32x32@2x.png"   >/dev/null
    sips -z 128  128  icon.png --out "${ICONSET}/icon_128x128.png"     >/dev/null
    sips -z 256  256  icon.png --out "${ICONSET}/icon_128x128@2x.png" >/dev/null
    sips -z 256  256  icon.png --out "${ICONSET}/icon_256x256.png"     >/dev/null
    sips -z 512  512  icon.png --out "${ICONSET}/icon_256x256@2x.png" >/dev/null
    sips -z 512  512  icon.png --out "${ICONSET}/icon_512x512.png"     >/dev/null
    sips -z 1024 1024 icon.png --out "${ICONSET}/icon_512x512@2x.png" >/dev/null
    iconutil -c icns "${ICONSET}" -o "${RESOURCES}/icon.icns"
    echo "    icon.icns written."
else
    echo "    (no icon.png found — skipping icon)"
fi

# ── 4. DMG ────────────────────────────────────────────────────────────────────
DMG="dist/${APP_NAME}.dmg"
rm -f "${DMG}"

echo "==> Creating ${DMG}…"
if command -v create-dmg &>/dev/null; then
    create-dmg \
        --volname "${APP_NAME}" \
        --volicon "${RESOURCES}/icon.icns" \
        --window-pos 200 100 \
        --window-size 540 380 \
        --icon-size 100 \
        --icon "${APP_NAME}.app" 140 180 \
        --hide-extension "${APP_NAME}.app" \
        --app-drop-link 400 180 \
        "${DMG}" \
        "dist/" || [ $? -eq 2 ]
else
    hdiutil create \
        -volname "${APP_NAME}" \
        -srcfolder "${BUNDLE}" \
        -ov -format UDZO \
        "${DMG}"
fi

echo ""
echo "Done!  →  ${DMG}"
