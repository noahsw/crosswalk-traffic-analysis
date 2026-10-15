# TODO

## On site (you)

- [x] Tape-measure the red curb on the south sidewalk: 25 ft 11 in (2026-10-15).
- [x] Measure the crosswalk length: 34 ft on Google Maps (2026-10-15).
- [ ] Do a 10-minute manual count and run `validate.py --count-window ... --manual-count ...`.
- [ ] Optional: drive past at a fixed speed (cruise control), 3+ passes each direction, log times,
      and run `validate.py --speeds`.

## Next run

- [ ] Re-run the first full hour. After the first run, the gated-entrance gate was extended
      to cover its exit lane, and the compass orientation was corrected (camera on the north side, facing southeast).
      The first run's CSVs and report were relabeled, but its annotated video still shows the old
      eastbound/westbound labels, and cars leaving the gated entrance show as "not seen".

## Features

- [x] Driver yielding: `yield_events.csv`, `yield_clips/`, and a summary section.
- [ ] Watch every yield clip from the full hour and note any misjudged ones; tune
      `yielding` settings if needed (`yielding.py` re-runs in seconds).
- [ ] Re-apply gate / count-line edits from the saved `tracks.pkl` instead of re-running detection.
- [ ] Combine several recording sessions (different days/times, school hours) into one report.
- [ ] PDF version of the summary for the city.
- [ ] Full homography calibration (top-down plane in feet), so each vehicle's maximum
      instantaneous speed can be reported, not just its average through the trap.
- [ ] Optional blur of people and license plates if the video is ever shared publicly.

## Maintenance

- [ ] `supervision.ByteTrack` is deprecated (pinned `supervision<0.31`, where it is removed).
      Move to Roboflow's `trackers` package before upgrading.
- [ ] Night and low-light footage is untested.
