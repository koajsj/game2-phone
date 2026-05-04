# Posture Pulse

Mobile-first posture analysis web app for GitHub Pages.

## Features

- Real-time pose tracking with MediaPipe Pose Landmarker
- Live posture score and quality status (`Good` / `Watch` / `Alert`)
- Core metrics:
  - Head drift angle
  - Shoulder balance delta
  - Trunk tilt angle
  - Motion stability
- Session analytics:
  - Session duration
  - Good posture ratio
  - Alert duration
- Local-only processing: camera frames never leave the browser

## Tech

- HTML/CSS/JavaScript (no build step)
- `@mediapipe/tasks-vision` loaded via CDN

## Run locally

Because camera APIs require secure context, use `https` or localhost:

```powershell
cd C:\Users\Administrator\Desktop\posture-analyzer
python -m http.server 5500
```

Then open `http://localhost:5500`.

## Deploy to GitHub Pages

1. Push all files to your repo default branch.
2. In GitHub repo settings, enable Pages from `Deploy from a branch`.
3. Select `/ (root)` and branch `main`.
4. Open the generated `https://<username>.github.io/<repo>/` URL.

## Privacy

All detection runs client-side in the browser.
