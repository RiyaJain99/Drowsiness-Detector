<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&height=220&color=0:0F131D,50:171B26,100:262A35&text=DrowsyGuard&fontSize=48&fontColor=FF5451&fontAlignY=38&desc=Real-time%20driver%20drowsiness%20monitoring%20%E2%80%94%20100%25%20in%20the%20browser&descAlignY=58&descSize=18&animation=fadeIn"/>

<p>
<img src="https://img.shields.io/badge/React_19-20232A?style=for-the-badge&logo=react&logoColor=61DAFB"/>
<img src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white"/>
<img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white"/>
<img src="https://img.shields.io/badge/Tailwind_CSS_4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white"/>
<img src="https://img.shields.io/badge/MediaPipe-FF6F00?style=for-the-badge&logo=google&logoColor=white"/>
</p>
<p>
<img src="https://img.shields.io/badge/100%25_client--side-4EDEA3?style=for-the-badge"/>
<img src="https://img.shields.io/badge/no_video_upload-4EDEA3?style=for-the-badge"/>
<img src="https://img.shields.io/badge/no_server-4EDEA3?style=for-the-badge"/>
<img src="https://img.shields.io/badge/license-ISC-373737?style=for-the-badge"/>
</p>

**No server, no video upload, no Python environment to reproduce — open the page and it works.**

</div>

---

### Contents

[Quick start](#quick-start) · [Architecture](#architecture) · [Detection](#detection) · [Voice control](#voice-control) · [Interface](#interface) · [Limits](#limits)

---

## Quick start

```bash
npm install
npm run dev      # http://localhost:5173
npm test         # 43 unit tests over the detection logic
npm run build
```

Camera access requires `localhost` or HTTPS. `npm run setup` (run automatically by `dev` and `build`) vendors the MediaPipe WASM runtime out of `node_modules` and downloads the face landmarker model into `public/`, so the deployed app has no runtime CDN dependency.

## Architecture

Facial landmarks come from MediaPipe FaceLandmarker — 478 points, WASM with a GPU delegate, at camera framerate. Everything above that layer is this project's own logic, and it is deliberately split so the interesting part can be tested without a browser:

```
src/lib/geometry.ts             pure maths — EAR, MAR, head pitch, trimmed mean
src/lib/drowsiness.ts           the state machine; takes `now` as an argument, never reads a clock
src/lib/voice.ts                voice command parsing and spoken responses — pure, no mic access
src/lib/config.ts               every threshold in one place, with the reasoning
src/lib/frame.ts                frame luminance sampling
src/hooks/useMonitor.ts         camera + model + rAF loop; the only impure layer
src/hooks/useVoiceAssistant.ts  microphone + speech recognition glue
src/components/                 presentation
```

`DrowsinessAnalyzer` knows nothing about cameras or canvases. Its tests drive several minutes of simulated driving in under 60 ms.

## Detection

**Eye aspect ratio.** Six landmarks per eye give vertical lid separation over horizontal eye width:

```
EAR = (|p2−p6| + |p3−p5|) / (2·|p1−p4|)
```

Dividing by eye width is the point — the value is independent of distance from the camera and of face size, so one threshold survives the driver shifting in their seat.

**Per-driver calibration.** Published implementations hard-code `EAR < 0.25`, which really means "0.25 worked for the author's eyes". This samples four seconds of open-eye baseline, takes a *trimmed* mean so a blink during calibration cannot drag it down, and sets `threshold = baseline × 0.78`. There is a test covering a driver whose open eyes read 0.19, where the fixed cutoff would report them as permanently asleep.

**PERCLOS.** Proportion of eyes-closed time over a rolling 60-second window. This, not instantaneous EAR, is the metric validated against driving impairment in the fatigue literature — a single low frame is noise. Above 15% is drowsy, above 30% critical.

**Blink versus microsleep.** A closure at or under 400 ms is an ordinary blink. Past 500 ms it is a microsleep and escalates while the eyes are *still shut*, not after they reopen. A detector that alarms on normal blinking is one that gets switched off on day one.

**Yawns.** Mouth aspect ratio held above 0.55 for 900 ms. The hold requirement is what separates a yawn from speech, which opens the mouth briefly and repeatedly.

**Head pitch.** Pitch rotation foreshortens the vertical face axis while leaving the interocular distance alone, and a rigid segment rotated by θ projects to cos(θ) of its length — so inverting that ratio against the calibrated neutral pose recovers the angle in degrees, no solvePnP required. Direction comes from the forehead-to-jaw balance, which shifts as the head tilts. A nod must hold 800 ms, because checking the mirrors dips the head for far less.

**Light level.** Mean Rec. 601 luma of a 32×32 downsample of each frame. Tracking degrades badly in the dark and the driver is told so rather than being handed a confident reading built on a bad signal.

**Hysteresis.** States must dwell 2.5 s before de-escalating, so readings sitting on the threshold do not flicker the display between alert and drowsy.

**Acknowledgement.** In the critical state the primary control becomes *I am awake*, which mutes repeat alarms for 60 s while leaving the display red. The alarm itself repeats on an interval rather than firing once — without that, a system that cannot be acknowledged is a system that gets muted permanently on day one.

## Voice control

Say **"Riri"** any time monitoring is running:

| You say | Riri does |
|---|---|
| "Riri" (alone) | Replies "Yes?" — acknowledges being addressed, nothing more |
| "Riri, what's my PERCLOS?" | Speaks the current PERCLOS, yawn count, blink rate, microsleep count, or head-nod count |
| "Riri, how am I doing?" | Speaks a full status summary |
| "Riri, I'm up" *(while critical)* | Silences the repeating alarm, same as tapping **I am awake** |

Everything requires the wake word — a bare metric keyword or "I'm up" said in passing conversation shouldn't make the app talk over you or silence a real alarm. Recognition only runs while monitoring is active (not always-on in the background) and only acts on a finalized transcript, not the in-progress guesses speech recognition produces while you're still talking.

Built on the browser's Web Speech API, so it's Chrome/Edge only for now (see [Limits](#limits)).

## Interface

A Material Design 3 dark cockpit HUD. A two-column dashboard on wide screens — camera viewfinder and EAR waveform on the left, PERCLOS, telemetry, and the incident timeline on the right — collapsing to one column on narrow ones. A single CSS custom property `--state` carries the verdict to every accent — viewport border, PERCLOS figure, trace line, control deck — so the whole cockpit re-themes without a conditional class anywhere in the component tree.

The EAR trace is the centrepiece: 20 seconds of history with the calibrated threshold drawn in, a gradient fill under the live curve, and closure runs shaded, so the algorithm can be watched deciding rather than trusted. PERCLOS marks both thresholds on its bar rather than hiding them in a colour change, and each telemetry cell shows the normal band next to the number so a reading like "6 blinks/min" is interpretable.

Motion is deliberately limited to what's actually informative: the header's status beacon pulses only in the critical state, and the microphone indicator pulses only while it's genuinely listening. It is tempting to pulse everything red when the alarm fires, but a strobing alert aimed at someone already impaired is a hazard rather than a warning — colour, border weight and type size carry the hierarchy instead. `prefers-reduced-motion` turns all of it off.

Keyboard focus is visible throughout and state changes are announced through one atomic `role="status"` region rather than competing live regions.

## Limits

Single face only. Degrades in very low light and against strong reflections on glasses — the light readout warns about the first case but cannot fix it.

Pitch is estimated from a foreshortening ratio rather than a full 3-D pose solve, so it conflates pitch with any motion that changes apparent face height; yaw and roll are not modelled at all. A proper solvePnP against a canonical 3-D face model would give all three axes and is the obvious next step.

Voice control needs the Web Speech API (Chrome/Edge; not implemented in Firefox or Safari) and, unlike the on-device face tracking, sends audio to a cloud speech-recognition service — it degrades to unavailable rather than partially working elsewhere, and the touch controls cover everything it does.

The "rest stop" action opens a Maps search rather than routing to a vetted stopping place.
