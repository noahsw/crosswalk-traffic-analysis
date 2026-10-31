# Crosswalk traffic study

Turns fixed-camera video of an uncontrolled crosswalk (for example, shot from a balcony
with an iPhone) into evidence for a pedestrian safety request to a city, such as a
pedestrian-activated flashing beacon (RRFB):

- `annotated.mp4`: the video with vehicle boxes, estimated speeds, a live date/time
  stamp and running counters. **No audio**, and the phone's GPS and other metadata
  are removed.
- `vehicles.csv`, `pedestrians.csv`, `summary.md`, `summary.json` and two charts.

Vehicle counting is the main goal. Speeds are secondary and stay labeled as
**estimates** until the speed trap is tape-measured and checked (see below).

## Setup (once)

```bash
brew install ffmpeg uv
uv sync
```

The first run downloads the YOLO model (about 40 MB) into `models/`. On an M-series
Mac it uses the GPU (Apple "MPS") automatically, and it falls back to the CPU on other machines.

On a Mac, detection can run on the Neural Engine instead: set
`detection.model: models/yolo11m.mlpackage` in `config.yaml`. It's exported from
`yolo11m.pt` the first time (about a minute), then finds the same objects in about half the
time and power. To re-export (say, after changing `detection.imgsz`), delete the
`.mlpackage`.

## Recording tips

- Record at **1080p, 30 fps**. 4K or 60 fps makes processing much slower and doesn't help counting.
- Keep the phone in exactly the same spot and framing each time, so one calibration
  covers every session. Use a clamp or tripod, and keep the phone plugged in.
- Shoot in daylight only, for now.
- The phone's clock stamps the start time. To check it, film your phone's lock
  screen for a second at the start.

## Calibration

`calibration.yaml` holds the scene geometry, all in image pixels, plus a `site_name` for
report titles. It is gitignored because it describes where the camera is. Start from the
example, then redraw every shape for your camera:

```bash
cp calibration.example.yaml calibration.yaml
```

To view the current shapes on a frame of your video:

```bash
uv run python calibrate.py input/VIDEO.MOV --preview-only
```

Then open `calibration_preview.jpg`. To redraw anything:

```bash
uv run python calibrate.py input/VIDEO.MOV
```

A window steps through each shape. Click the points and press **Enter**. Pressing
Enter without clicking keeps the existing gray shape.

| Shape | What it's for |
|---|---|
| Count line | Every vehicle crossing it is counted, once, with direction. It sits beside the crosswalk on the west leg, across both lanes. |
| Speed trap A and B | Two lines across the road, each starting at one end of the **red curb** on the far (south) side. Speed = distance between them ÷ time taken. |
| Gates (east, west, south, north) | One short line per way in or out of the scene. The gates a vehicle crosses tell us whether it went straight through or turned. |
| Crosswalk | 4 corners: the two near-side (south) corners first, then the two far-side corners. A person who walks across it is a pedestrian crossing. |

### The one measurement that matters for speeds

Tape-measure the **length of the red curb** on the south sidewalk, from end to end.
Photograph the tape in place. Then edit `calibration.yaml`:

```yaml
speed_trap:
  distance_ft: 27.5          # your measurement
  distance_measured: true
  distance_notes: Red curb, south side, tape-measured 2026-10-16, photo IMG_3101
```

Every speed scales directly with this number. The red curb was measured at
25 ft 11 in on 2026-10-15. If you re-measure, update `calibration.yaml` and run
`summarize.py` (below) to rescale an existing run's speeds without re-processing.

### Crosswalk length (for the crossing-wait analysis)

`road.crosswalk_length_ft` in `config.yaml` is the curb-to-curb distance a pedestrian
walks. It is currently an estimate of 36 ft. You can measure it without stepping into the
street: use the measuring tool in Google Maps satellite view, or count the crosswalk
stripes and the gaps between them. Then set `crosswalk_length_measured: true`.

## Running

```bash
# Quick test on the first 5 minutes (check it works before a full run)
uv run python analyze.py input/VIDEO.MOV --minutes 5

# Or a 5-minute slice starting 20 minutes in
uv run python analyze.py input/VIDEO.MOV --start-min 20 --minutes 5

# Full run: about 1 hour of processing per hour of video
uv run python analyze.py input/VIDEO.MOV
```

Outputs go to `out/<video name>/`, or `out/<video name>_<start>-<end>min/` for a partial
run, so a quick test never overwrites a full run. Status lines print every 30 seconds.
Useful options:

- `--no-video`: skip the annotated video (slightly faster).
- `--stride 3`: detect on every 3rd frame, which is faster but a little less accurate.
- `--start-time "2026-10-15T08:29:01"`: override the start time if the phone metadata is wrong.

Settings such as the speed limit, model and overlay toggles are in `config.yaml`.

To rebuild the report after changing `config.yaml` or `calibration.yaml`, without
re-processing the video (this takes seconds):

```bash
uv run python summarize.py out/VIDEO
```

The first step always copies the video without its audio track into `work/`.
Nothing after that reads the original file, and the finished video is checked
automatically to make sure it has no audio.

## Live camera (Tapo C120)

`live.py` watches a Tapo C120 (or any camera with an RTSP stream) instead of a recording.
It saves a clip of each pedestrian drivers don't yield to (one clip with all of those drivers),
and of every pedestrian crossing, as they happen. When you stop it, it writes the usual CSVs and `summary.md` for the
whole session.

### Camera setup (once)

1. In the Tapo app, open the camera, then **Settings → Advanced Settings → Camera Account**,
   and create a user name and password. This is a separate login just for the camera, not
   your TP-Link account. On newer firmware you may also need to turn on
   **Me → Tapo Lab → Third-Party Compatibility**.
2. Find the camera's IP address (in the app under **Settings → Device Info**, or in your
   router), and reserve it in your router so it doesn't change. Put it in `config.yaml`:

   ```yaml
   live:
     host: 192.168.1.50
   ```

3. Save the camera account in the macOS Keychain, so the password is never in a project
   file. Run this once, with your camera user name; it asks for the password without showing
   it:

   ```bash
   security add-generic-password -U -s tapo-camera -a your-camera-user -w
   ```

   Run it again to change the password (`-U` replaces the saved one). The item is named
   `tapo-camera` in the Keychain Access app; to use another name, set `live.keychain_item`
   in `config.yaml`. If `TAPO_USER` and `TAPO_PASSWORD` are both set in the environment,
   they are used instead.

4. Mount the camera where it sees the crosswalk and both approaches, the same way as the
   phone (see Recording tips). Then draw the scene on the camera's own view. The camera's
   framing and resolution differ from the phone's, so it gets its own calibration file,
   `calibration.tapo.yaml`:

   ```bash
   uv run python calibrate.py --camera
   ```

   Enter the same red-curb length (25.92 ft) for the speed trap. Check
   `calibration.tapo_preview.jpg` afterwards. The C120 also sends audio, but it is never
   decoded or saved.

### Running

```bash
uv run python live.py              # until Ctrl+C
uv run python live.py --hours 8    # stop by itself after 8 hours
uv run python live.py --show       # also watch the tracking live in a window
```

`--show` opens a window with the same overlay as `annotated.mp4`: the crosswalk, count
line, speed trap, gates, a box per tracked vehicle and person, and running counts. Press
**q** in the window (or close it) to hide it; the session keeps going. The window is
skipped on frames where detection is behind, so it never costs detections.

Press **Ctrl+C** (or `kill` the process) to stop. Clips of drivers who did not yield go to
**`out/did_not_yield/`**, one folder shared by every session, so they're all in one place.
Each is saved about 20 seconds after it happens: the pedestrian has to finish crossing first,
and every car that could still count has to pass. These are the same clips as for a recording,
with the date in the file name. (If `yielding.clip_outcomes` adds other outcomes, they get
their own shared folders, e.g. `out/yielded/`.) Everything else goes to `out/live_<date>_<time>/`:

- `crossing_clips/`: one clip per pedestrian crossing, so you can check the results by eye.
  Each starts 10 seconds before the pedestrian is first seen waiting at the curb
  (`yielding.crossing_clip_seconds_before`), so you can see whether they had to wait for
  cars, and ends once they're across. Every car is boxed, with judged drivers in green
  (yielded) or red (did not yield) and the rest in gray. People crossing together share one
  clip. The file name has the time, direction, pedestrian ids and driver outcomes, e.g.
  `20261029_150726_northbound_P260_2yielded.mp4`. Turn them off with
  `yielding.clip_every_crossing: false`.
- `yield_log.csv`: one row per pedestrian crossing and per judged driver, added as soon as
  they are judged, with the driver clip's file name. Clips and this log are already on disk if the session never finishes
  (battery dies, a crash); the files below are only written on stop.
- On stop: `vehicles.csv`, `pedestrians.csv`, `yield_events.csv`, `summary.md`, the review
  images and `all_did_not_yield.mp4` (that session's clips joined). Live mode doesn't write an
  annotated video.

To keep clips, the stream is also recorded in 10-second pieces under `work/live/`. Pieces
older than 10 minutes are deleted, and the folder is removed when the session ends. If the
camera drops off Wi-Fi, `live.py` reconnects every 5 seconds and carries on.

**Sleep.** While `live.py` runs, it keeps the Mac from sleeping when idle (and from any
sleep while on power), so there's no need for `caffeinate`. Closing the lid still puts a
MacBook to sleep, unless it's on power with an external display attached. If the Mac does
sleep, nothing is watched while it's asleep, but nothing already seen is lost: on wake,
crossings from before the sleep are judged and their clips saved, and the session carries
on. `summary.md` lists the time with no video, and leaves it out of the per-hour rates.

A status line every minute shows `skipped N frame(s)`. If that keeps growing, detection
can't keep up with the camera. Use the Neural Engine model above if you aren't
already, then raise `live.stride` (it is 2: every other frame, 10 detections a second from
the C120's 20 fps) or use `detection.model: models/yolo11s.pt` in `config.yaml`.

To rehearse without the camera, play a recording back at real speed as if it were live,
using the phone's calibration:

```bash
uv run python live.py input/VIDEO.MOV --calibration calibration.yaml
```

The `yielding` settings in `config.yaml` apply to both modes. `waiting_area_px` is in image
pixels, so it may need adjusting for the camera's view.

## Reading the outputs

**summary.md** is the report: the recording window, vehicles counted (total, per hour,
by direction, in 15-minute bins), movements, speed statistics, pedestrian crossings,
how long pedestrians wait for a safe gap, calibration details and validation status.

Its **How this compares** section puts the numbers in context with published references:

- **Volume:** an estimated daily volume, compared with FHWA's typical ranges for local
  streets, collectors and arterials.
- **FHWA crossing treatments:** which treatments FHWA's guide for uncontrolled crossings
  lists for this road. The guide's table is keyed to lanes, posted speed and daily traffic.
- **Speed and injury risk:** pedestrian risk of death and severe injury at the posted limit
  and at the measured 85th-percentile speed (AAA Foundation).
- **Crossing delay:** the Highway Capacity Manual's A to F grade for the average wait.

**vehicles.csv** has one row per vehicle that crossed a gate or the count line:

| Column | Meaning |
|---|---|
| `timestamp_local` | When it crossed the count line, or when it was first seen if it never crossed. |
| `video_seconds` | The same moment as seconds into the video, so you can jump to it. |
| `counted` | `True` if it crossed the count line. These rows are the headline count. |
| `direction` | eastbound or westbound at the count line. |
| `entry_gate`, `exit_gate`, `movement` | How it entered and left: `through`, `left_turn`, `right_turn` or `unknown`. `movement_inferred` means one gate was deduced from the count line. |
| `speed_mph`, `n_points`, `speed_valid`, `discard_reason` | Speed-trap result. Only `through` vehicles count toward speed statistics, because turning vehicles slow down by design. |

**pedestrians.csv** lists one row per crossing, with its time and direction.

**Driver yielding** (`yield_events.csv`, `yield_clips/`): for each pedestrian who crossed,
every vehicle that reached the crosswalk while they were trying to cross is judged:

- **did not yield**: drove through without slowing below 5 mph, in a lane the pedestrian
  still had to cross. While the pedestrian waits at the curb, both lanes count; set
  `yielding.waiting_counts_both_lanes: false` to count only the lane next to them.
- **yielded**: stopped or nearly stopped first. Only the first vehicle per direction counts,
  so a queue behind it doesn't inflate the rate.
- **too close to stop**: reached the crosswalk less than 3 s after the pedestrian appeared, so
  it is not counted against the driver.

Every pedestrian a driver didn't yield to gets one clip in `yield_clips/` (in
`out/did_not_yield/` for live sessions), from a few seconds before the first such driver reaches
the crosswalk until just after the last, with each of them boxed in red, the pedestrian
highlighted, the time and a caption. All of them are joined in `yield_clips/all_did_not_yield.mp4`. **Watch each clip before sharing it**: the rules are
automatic, and plants hide part of the near (north) end of the crosswalk. The numbers are in
the "Drivers yielding to pedestrians" section of `summary.md`, and `pedestrians.csv` gains each
crossing's wait at the curb and driver counts.

To redo the yielding analysis after changing the `yielding` settings in `config.yaml`, without
re-processing the video (it reuses the tracks saved in `tracks.pkl`):

```bash
uv run python yielding.py out/VIDEO
```

**Sight distance** (`sight_distance` in `config.yaml`): how far back a driver on an approach
can first see a pedestrian at the crosswalk. The report compares it with the AASHTO stopping
sight distance (200 ft at 30 mph, more on a downhill approach) and notes FHWA's provision for
an additional RRFB in advance of the crosswalk where sight distance is short. To measure it:

1. One person stands at the curb at the crosswalk.
2. Another walks back along the sidewalk on that approach, with their eyes about 3.5 ft above
   the road (a driver's eye height, roughly crouching).
3. Stop where you can no longer see the first person's knees. AASHTO assumes a 2 ft high object.
4. Measure the distance with a measuring wheel or Google Maps, and set `feet:`. If the approach
   slopes down toward the crosswalk, set `grade_pct:` negative (for example -4 for a 4% downgrade).

Then run `summarize.py` to update the report.

Yield results are also broken down **by direction**, to compare the approaches.

**pedestrians_reviewed.csv** and **ped_check/** cover everyone whose feet came near the
crosswalk, counted or not, with the reason: `counted`, `near but never inside` (for example,
walking along the sidewalk past the gated entrance), `riding a bike or motorcycle`, `did not
move far enough across`, or `too briefly inside`. Each image shows the person's path, the time,
and where to find them in the video. Flip through these to check that no real crossings were missed.

**movement_check/** holds a few sample images for each movement type, each with the
vehicle's path drawn on it. Flip through them to confirm through and turn are classified correctly.

**Charts:** `vehicles_per_15min.png` and `speed_histogram.png`, which marks the speed
limit and the 85th percentile. The 85th percentile is the number traffic engineers use.

In the video, a vehicle's box is **blue** until its speed is measured. After that it's
**green** at or under the limit, **yellow** up to 10 mph over, and **red** above that.
Turning vehicles are **gray** "TURN", pedestrians are magenta and bicycles are cyan.
"est" after a speed means the trap hasn't been tape-measured yet.

## Validation

The city will trust the numbers more if they've been checked:

1. **Manual count.** Watch 10 minutes of the video, count vehicles crossing the
   count line by hand, then run:
   ```bash
   uv run python validate.py out/VIDEO --count-window 08:40 08:50 --manual-count 37
   ```
   The target is within 5%.
2. **Known speeds.** Drive your own car past at a steady speed on cruise control,
   a few passes in each direction, while recording. Note the time and speed of each pass in a CSV:
   ```
   timestamp,direction,mph
   08:41:10,eastbound,25
   08:43:55,westbound,25
   ```
   Then run:
   ```bash
   uv run python validate.py out/VIDEO --speeds reference_speeds.csv
   ```
   The target is a mean absolute error of 3 mph or less.

Both results are added to `summary.md`.

## How it works

1. **Strip audio** (ffmpeg stream copy) and read the start time from the phone's metadata.
2. **Detect** cars, trucks, buses, motorcycles, bicycles and people with YOLO11 on every
   2nd frame, then **track** them with ByteTrack. People are also looked for in a
   full-resolution crop around the crosswalk (`detection.crosswalk_zoom`), where someone
   waiting at the curb is otherwise too small to detect reliably. Someone who vanishes at the
   crosswalk (behind the plants, say) and reappears within 3 s is kept as one person. Each frame's timing comes from its
   own timestamp, because iPhone video has a variable frame rate. The live camera is the
   opposite: a steady frame rate with timestamps off by up to half a second, so live frames
   are timed one frame period apart, kept in line with the timestamps over the long run.
   (Recordings cut from the camera keep its timestamps, so `analyze.py` misreads their speeds.)
3. Each vehicle's **ground point** (the bottom-center of its box) is smoothed with a rolling
   median. A crossing is recorded when that point passes the count line, a gate or a trap line.
4. A **pedestrian crossing** is a person who spends at least 0.5 s inside the crosswalk
   and moves at least 30% of the way across. People standing at the corner and bike
   riders don't count, and neither do drivers seen through car windows.

Known limitations: daylight only. A vehicle hidden behind a truck the whole way can
be missed. Speeds depend on the trap distance.
