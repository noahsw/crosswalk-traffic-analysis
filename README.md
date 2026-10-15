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
  still had to cross. While the pedestrian waits at the curb, only the lane next to them counts.
- **yielded**: stopped or nearly stopped first. Only the first vehicle per direction counts,
  so a queue behind it doesn't inflate the rate.
- **too close to stop**: reached the crosswalk less than 3 s after the pedestrian appeared, so
  it is not counted against the driver.

Every "did not yield" gets an ~9 second clip in `yield_clips/` with the vehicle boxed in red,
the pedestrian highlighted, the time and a caption. All of them are joined in
`yield_clips/all_did_not_yield.mp4`. **Watch each clip before sharing it**: the rules are
automatic, and plants hide part of the near (north) end of the crosswalk. The numbers are in
the "Drivers yielding to pedestrians" section of `summary.md`, and `pedestrians.csv` gains each
crossing's wait at the curb and driver counts.

To redo the yielding analysis after changing the `yielding` settings in `config.yaml`, without
re-processing the video (it reuses the tracks saved in `tracks.pkl`):

```bash
uv run python yielding.py out/VIDEO
```

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
   2nd frame, then **track** them with ByteTrack. Each frame's timing comes from its
   own timestamp, because iPhone video has a variable frame rate.
3. Each vehicle's **ground point** (the bottom-center of its box) is smoothed with a rolling
   median. A crossing is recorded when that point passes the count line, a gate or a trap line.
4. A **pedestrian crossing** is a person who spends at least 0.5 s inside the crosswalk
   and moves at least 30% of the way across. People standing at the corner and bike
   riders don't count, and neither do drivers seen through car windows.

Known limitations: daylight only. A vehicle hidden behind a truck the whole way can
be missed. Speeds depend on the trap distance.
