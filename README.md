# Hand Controller

Hand Controller is a Windows desktop-control project built in verified stages. **Stages 1 and 2 are complete:** it tracks all 21 hand landmarks with the MediaPipe Tasks API, draws them over a mirrored webcam preview, and maps the index fingertip to a smoothed system cursor with click, drag, right-click, and scroll gestures.

The PySide6 dashboard, configurable gesture bindings, and packaging remain separate later stages so the webcam and gesture baseline can be debugged first.

## Setup

Use a virtual environment running **Python 3.13**. Python 3.13 is installed on this computer; do not create the environment with Python 3.14 because the pinned native packages do not provide its required wheels.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the official MediaPipe Hand Landmarker model into the expected location:

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" `
  -OutFile "assets\models\hand_landmarker.task"
```

Then run:

```powershell
python main.py
```

The preview is mirrored by default, so moving your hand right moves the cursor right. The gold rectangle is the active tracking area: its edges correspond to the edges of your display. Press `q` while the preview is focused to stop. PyAutoGUI's upper-left-corner failsafe remains enabled.

## Stage 2 gestures

The current gesture name appears at the top of the preview.

| Hand pose | Action |
| --- | --- |
| Index finger extended | Move cursor |
| Bring thumb and index finger together, then release | Left-click |
| Hold thumb/index pinch for 0.45 seconds, move, then release | Drag and drop |
| Middle finger extended alone for 0.22 seconds | Right-click |
| Index and middle fingers extended; move hand up/down | Scroll up/down |
| Open palm | No assigned action |
| Closed fist | No assigned action |

Open and closed hands do not pause, disable, click, or release a drag. For safety, tracking loss, pressing `q`, and application shutdown release a held left mouse button. A brief pinch is ignored; a pinch must last at least 0.06 seconds before it can click.

## Calculation reference

MediaPipe supplies each of the 21 landmarks as normalized `x`, `y`, and `z` coordinates. `x` and `y` normally lie between `0` and `1`; the code mirrors the camera image before inference, so the coordinates already match the mirrored preview. For drawing, each normalized point becomes a camera pixel with `pixel_x = round(x × (frame_width - 1))` and `pixel_y = round(y × (frame_height - 1))`.

### Cursor mapping and sensitivity

For index-tip coordinate `(x, y)`, the application first clamps it to the configured active area `[left, right] × [top, bottom]`. It then converts it to coordinates relative to that area:

```text
rx = (clamp(x, left, right) - left) / (right - left)
ry = (clamp(y, top, bottom) - top) / (bottom - top)
```

Sensitivity `s` expands or contracts movement around the screen centre, followed by another clamp to `[0, 1]`:

```text
rx' = clamp((rx - 0.5) × s + 0.5, 0, 1)
ry' = clamp((ry - 0.5) × s + 0.5, 0, 1)
screen_x = rx' × (screen_width  - 1)
screen_y = ry' × (screen_height - 1)
```

At `s = 1`, the active-area edge maps directly to the corresponding screen edge. Values above `1` make the cursor travel farther for the same hand movement.

### Cursor smoothing

The cursor position is filtered with an exponential moving average. For each axis independently:

```text
filtered_position = previous_position + α × (target_position - previous_position)
```

`α` is the `smoothing` setting. Higher values follow the hand more quickly; lower values reject more jitter but add visible latency. The first detected target is used directly, and the filter initially starts at the current system cursor position.

### Landmark distance and finger extension

The 3D Euclidean distance between landmarks `a` and `b` is:

```text
d(a, b) = √((ax - bx)² + (ay - by)² + (az - bz)²)
```

For each non-thumb finger, the program calculates the angle at its PIP joint from the MCP joint, PIP joint, and fingertip:

```text
θ = arccos(clamp((u · v) / (|u| × |v|), -1, 1))
u = MCP - PIP
v = TIP - PIP
```

A finger counts as extended only when both conditions are true:

```text
θ ≥ 155°
d(TIP, WRIST) > 1.05 × d(PIP, WRIST)
```

This combines joint geometry and outward fingertip position instead of relying only on a single 2D distance.

### Pinch, click, and drag timing

Pinch distance is scaled by the user’s apparent palm size, so it is less sensitive to distance from the camera:

```text
palm_size = d(WRIST, MIDDLE_MCP)
pinch_ratio = d(THUMB_TIP, INDEX_TIP) / palm_size
pinch = pinch_ratio ≤ pinch_threshold
```

The default `pinch_threshold` is `0.35`. A pinch held for at least `0.06 s` then released produces one left-click. A pinch held for at least `0.45 s` emits one mouse-down event; releasing it emits one mouse-up event. Thumb/index geometry is checked before other finger poses, allowing a user to curl their other fingers during a drag.

### Right-click debounce

Right-click requires only the middle finger to be extended. The pose must remain valid continuously for `right_click_debounce_ms` (default `220 ms`) before one right-click is emitted. It is latched until the pose changes, preventing repeated right-clicks every frame.

### Two-finger scroll and flick boost

Scroll mode requires index and middle fingers extended while ring and pinky are folded. Its vertical hand position is the average fingertip height:

```text
y = (index_tip_y + middle_tip_y) / 2
Δy = previous_y - current_y
velocity = |Δy| / max(Δtime_seconds, 0.001)
```

Normal scroll contribution is `Δy × scroll_speed`. Fast motions receive a capped multiplier:

```text
multiplier = 1                                      when velocity ≤ threshold
multiplier = min(1 + (velocity - threshold) × gain,
                 max_multiplier)                    otherwise
scroll_contribution = Δy × scroll_speed × multiplier
```

The contribution is added to a fractional accumulator. `int(accumulator)` is sent to the operating system as whole scroll steps, and the remainder is retained for the next frame. A positive step scrolls up; a negative step scrolls down. This preserves slow fine control while making quick two-finger flicks travel farther.

### FPS display and timestamps

The preview FPS is calculated per frame as:

```text
FPS = 1 / max(current_time - previous_time, 0.0001)
```

MediaPipe video timestamps use `round(performance_counter × 1000)` milliseconds and are forced to increase by at least one millisecond between detections. This meets the task API’s monotonic-timestamp requirement.

## Settings

Edit [`config/settings.json`](config/settings.json) before launching:

- `camera_index`: use `1` or another index if the wrong webcam opens.
- `active_area`: normalized camera region that spans the full display. A smaller region makes movement easier but less precise.
- `cursor_sensitivity`: movement scale around the display centre.
- `smoothing`: `0.1` is smoother/slower; `1.0` disables filtering.
- `pinch_threshold`: pinch proximity relative to palm size; raise it if a pinch does not register, lower it to reduce accidental clicks.
- `drag_hold_seconds`: time to hold a pinch before drag begins.
- `right_click_debounce_ms`: time the middle-finger gesture must remain stable before right-clicking.
- `scroll_speed`: scroll steps for the same vertical hand movement.
- `scroll_flick_velocity_threshold`: minimum two-finger movement speed before a flick boost applies.
- `scroll_flick_gain` and `scroll_flick_max_multiplier`: how strongly a fast two-finger flick accelerates scrolling.
- MediaPipe confidence values: increase them to reduce weak detections; decrease them if detection is unreliable.

## Validation

The coordinate mapping logic is independent of hardware and can be tested with:

```powershell
python -m unittest discover -s tests -v
```

## Troubleshooting

- **NumPy tries to compile / no compiler found:** the virtual environment was created with Python 3.14. Delete only the `.venv` folder, recreate it with `py -3.13 -m venv .venv`, and reinstall the requirements.
- **Model missing:** download `hand_landmarker.task` to `assets/models` using the command above.
- **Webcam cannot open:** close another application using it, then adjust `camera_index`.
- **Cursor moves too much:** keep your index fingertip inside the gold rectangle and lower `smoothing` or `cursor_sensitivity`.
- **Emergency stop:** move the cursor to the top-left corner of the screen to trigger PyAutoGUI's failsafe, or press `q` in the preview.

## Planned stages

1. Completed — hand tracking and cursor movement.
2. Completed — click, drag, right-click, scroll, and safety debouncing.
3. Responsive PySide6 dashboard with a threaded camera worker.
4. Persistent configuration and customizable gesture bindings.
5. Profiling, optimization, tests, and Windows executable packaging.
