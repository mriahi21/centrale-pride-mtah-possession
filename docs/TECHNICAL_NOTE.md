# Technical note — Centrale Pride

(Submit this file as `Centrale Pride.md`.) MTAH 2027 Phase 1 — ball possession, BAR–ATH tactical video (199,319 frames).

**Final profile: `v1` — `python -m poss.make_submission --data-dir <DATA> --team-name "Centrale Pride" --profile v1` (no extra flags).** Profiles `v2`/`v3` were built, run on the real bundle and **not selected** (section 2).

## 1. Approach
A transparent, **training-free** temporal state model. spellF1 rewards whole possessions, so the design priorities are: no false splits, prompt but not twitchy hand-overs, and stoppages found from corroborated evidence.

1. **Parse the tactical ATD** (`video_frame = frame_id - 1`), 46 anonymous track slots → positions in metres. Slots are treated as track fragments: only team membership and positions are used, collectively, never as identities. Each half is processed independently.
2. **Ball cleaning**: reject impossible jumps (>45 m/s); interpolate short occlusions (≤2 s) with reduced confidence. A missing ball is *missing*, never "dead".
3. **Control evidence** per frame: Gaussian proximity (σ = 2 m) of the nearest home vs. nearest away player to the ball, discounted when the ball is fast (a pass flying past a defender is not control) and by ball confidence. A missing ball is zero evidence, so ownership carries through occlusions.
4. **Two-state Viterbi** (H/A) with a switch penalty λ (hysteresis). Possession changes hands only when the opposing run of evidence outweighs λ, and the switch lands where that run starts. Possessions are decoded independently between stoppages; the possession after a stoppage is decided afresh from restart evidence.
5. **Dead ball (D)** and **Smart Tagging** — see below (profile differences).
6. `X` is written only outside the graded halves (free). Inside graded halves we always answer.

### Profile `v1` (validated first candidate)
Dead ball from an additive score: out-of-bounds ball, stationary ball, slow players (unsupervised mixture, rank fallback), fast-ball veto, set-piece tag bonus; high/low threshold region growing; min durations; short-gap fill. λ chosen from a generic football prior on mean possession length (12 s) over a grid. Tag kinds' team sign learned without labels from agreement with the ball-only decode.

### Profile `v2` (built and evaluated; NOT selected, see section 2)
* **Stoppages need corroboration**: a stationary ball, quiet players or a set-piece tag alone never enters D; a reliable out-of-bounds, or two weak cues, do. Cues are weighted by observed support; occlusion gaps are bridged only when the ball was last and next seen stationary at the same spot; fast ball / controlled dribbling preserves genuine short live stretches; the stoppage starts at the first reliable out-of-bounds frame and ends when the ball actually leaves its resting spot. The weight of the "quiet players" cue adapts to how bimodal activity is in that half (unsupervised).
* **Control evidence**: interpolated ball frames influence ownership only over ≤1 s gaps with plausible endpoint motion; uncertainty is raised around occlusions; isolated touches are discounted; sustained contact (≥80% of 0.5 s, ≥1 m margin to the other team, slow ball) adds evidence.
* **λ = dwell (0.6 s) × typical strength of confident evidence**, clipped to [4, 30]. It depends only on the evidence, not on possession counts, dead-ball share, or any target duration.
* **Smart Tagging** only as capped soft evidence: a kind's sign must be consistent in both halves; the contribution is attenuated to zero where tracking evidence is strong; unknown team (`N/A`, all throw-ins) is neutral. Restart priors come from the first sustained contact after the stoppage (and a reliable tag, if any).

### Profile `v3` (built and evaluated; NOT selected, see section 2)
`v3 = v2 +` optional streams, applied **only if they pass predeclared verification checks** (otherwise the tracking-only evidence stands alone):
* **Broadcast tracking → ball fusion.** The broadcast recording has its own timeline (cuts, replays). Both trackings report the ball in the same pitch coordinates, so we align in tracking space: every 10 s window of broadcast ball positions is matched, by FFT cross-correlation, against the whole tactical ball track (orientation tested both ways); a window is accepted only with enough co-observed frames, small RMS distance, a clear margin over the runner-up, genuine ball motion (a stationary ball matches anywhere) and a consistent neighbour; every mapped frame is then re-verified against local agreement. The broadcast ball only fills frames where the tactical ball is missing. Accepted if ≥15% of broadcast frames map with median agreement ≤2 m (p90 ≤6 m).
  *Held-out check of the broadcast fill (ball position only, no possession labels):* stretches where the tactical ball was reliably observed are hidden (six gap lengths, 10–500 frames); the alignment is recomputed on the masked track and the hidden positions are recovered from the broadcast and, for comparison, by the short-gap interpolation the model already uses. Coverage and error are measured per gap length and per alignment-confidence tercile. The verify radius and the minimum gap length that broadcast may fill (shorter gaps stay with interpolation unless broadcast is at least as accurate there) are chosen from this held-out ball error alone, and the stream is applied only if the held-out recovery meets predeclared error limits.
* **Tactical video → pitch motion energy.** Frame-differencing inside the pitch (homography-projected mask) at 5 fps; used as a weak "quiet" cue independent of tracking errors. It is kept only if every check passes: frame count = 199,319; enough camera-still, calibrated frames (pan/zoom from the homography; the cue is switched off where the camera pans/zooms or the calibration is poor); per-block clock agreement with the tracking clock (≤12 frames in every 10-minute block); correlation with tracking activity well above its time-shifted copies; and incremental evidence — after regressing out tracking activity and camera motion, the residual must still separate ball-out-of-bounds frames (above its time-shifted controls). The references come from the tracking only; no labels.
* **Tactical audio → whistle events** (if an audio track exists): tonal 2.5–4.5 kHz bursts ≥0.24 s (robust z-score). Kept only if tracking activity drops after whistles and whistles coincide with tracking-derived stoppage starts more often than time-shifted copies of themselves. Used as a weak corroborating cue.
* **Broadcast video → targeted ball detections (optional, `--video-pass`).** Windows are chosen where extra ball evidence matters (long remaining tactical ball gaps, disputed turnovers, uncertain restarts), mapped to broadcast frames through the verified tracking alignment, and rejected at cut/replay boundaries, on ambiguous mappings or where the broadcast homography is unreliable. A deterministic classical detector (small bright round blobs inside the projected pitch, linked in pitch metres by dynamic programming) proposes ball tracks; a track fills a gap only if it is speed-consistent with the *observed* tactical ball at both gap flanks. The detector is first verified on the held-out stretches (hidden ball, alignment from the masked track); the pass is promoted only if the held-out error is small and it recovers frames that broadcast tracking plus interpolation did not. Otherwise the broadcast-tracking candidate stands.
* **Homography** → per-frame reliability (valid, reprojection error) gating the video cue; plus a tracking-vs-image consistency check.
The new cues are weighted 0.85 each (below the entry threshold of 1.65): they can only corroborate another cue, never create a stoppage alone.

## 2. Candidate selection and assets
All candidates were produced from the real bundle by one local script (`poss.fullrun`) and judged only by **predeclared plausibility gates** (dead-ball share per half 5–50 %, 80–500 possessions per half, mean possession 5–60 s, no team above 85 % of live frames, <5 % possessions shorter than 1 s, no `X` on graded frames) plus held-out verification of each extra signal. The gates are generic football sanity limits fixed before any real output was seen; they catch degenerate outputs and say nothing about accuracy.
* `v1` passed the format check and every gate.
* `v2` and every `v3` variant produced valid files but **failed the dead-ball-share gate** (too little stoppage time detected in the real tracking), so they were not submitted. We did not retune them to pass the gate on the real data.
* Broadcast tracking: the alignment and its held-out ball-recovery check passed (ball position only), but the only candidates that use it share the `v2` stoppage logic and therefore failed the gate.
* Tactical video motion cue, whistle cue and the broadcast-video ball detector did not pass their verification checks on the real footage and were not used. The broadcast-video detector, developed on synthetic video only, was rejected by its held-out verification.
* Better ball recovery or a verified cue would support using a signal; it would not establish better full-match `spellF1`, and no independent labels were available.

**Assets used by the final candidate:** tactical ATD (`BAR-ATH_tactical_atd.txt` + `_atd.xml`) and tactical Smart Tagging only. **Assets inspected but not used by the final candidate:** tactical video (+audio), tactical homography, broadcast video, broadcast ATD and broadcast homography (they were used only inside the evaluated, non-selected `v3` experiments). Per-asset results are in a local report (not committed). Public ground truth: used only to check the pipeline format; never for training, thresholds, ablations or candidate choice.

## 3. Training data
**None.** No learned parameters from any labelled data, from this game or others. No hand labels or hand corrections. Constants are football-logic defaults in `poss/*.py`, apart from the two changes made after the first real run (section 7). Tag-kind signs and the activity-bimodality weight are estimated per half from the supplied data by unsupervised procedures (algorithmic pseudo-labels, no human decisions). Development used a synthetic match simulator and feature-level scenarios (`poss/synth.py`, `poss/scenarios.py`; no match data).

## 4. Possession rules implemented
Possession stays with the last team in controlled possession while the ball is loose or in flight; a deflection or brief touch does not flip it; a change needs sustained opposing contact. A possession is a maximal H/A run; a stoppage always separates possessions and the next possession is decided afresh.

## 5. Limitations (honest)
* **No quantitative evaluation on the real match is claimed**: full labels are private; no independent labelled match was available; the public 5-minute window was used only for wiring. Synthetic results (simulator and scenario suite) check regressions and robustness, not real-match performance.
* All thresholds/weights are football-logic priors, untuned on labels; real tracking quirks (ball clipped at the lines, stationary live balls, slow build-up, long occlusions) may cause missed or spurious stoppages.
* The switch penalty λ selected by the v1 prior sits at the lowest value of its grid (3), so a still lower value was never considered; possessions may be split too eagerly.
* 2-D ball only (no height); aerial duels and long balls rely on the speed discount.
* Throw-in restarts carry no team tag; the restart team comes from ball evidence only.
* The tactical-video cue is a coarse motion-energy signal and can only corroborate. The broadcast-video ball detector is classical and was developed on synthetic video only; it is used only if its held-out ball error is acceptable, and every verification here concerns ball position / cue consistency, not possession accuracy. Broadcast fusion only helps where the alignment verifies.
* Synthetic ablations cannot show the real value of the video/audio cues (synthetic stoppages are easy); their inclusion rests on verification, not on measured accuracy.

## 6. Reproduction
```bash
pip install -r requirements-full.txt        # pinned; tested: Python 3.14, numpy 2.5.5, pandas 3.1.1 (v1 needs only requirements-pinned.txt)
# data bundle layout: <DATA>/tactical/BAR-ATH_tactical_{atd.txt,atd.xml,smart_tagging.xml}
python -m poss.make_submission --data-dir <DATA> --team-name "Centrale Pride" --profile v1 --out outputs
python evaluation/validate_submission.py "outputs/Centrale Pride.csv"
python -m pytest -q tests                   # synthetic regression tests, no match data
```
Deterministic: the same bundle and commit produce the same file. Expected SHA-256 of `Centrale Pride.csv`: `e46bd760ec7b3c39e105e7faf3bcd1019a557779bb76487d87cb8b90d7ad0faf`. The `v1` output is pinned by a regression test (`tests/test_frozen_v1.py`) and was regenerated bit-identically by later revisions of the code on the real bundle. The official baseline path remains available: `python starter/baseline.py --window 50 --out outputs`.

## 7. Development tools and changes after the first real run
**AI coding assistant.** The code, tests and documentation in this repository were written with Claude Code (Anthropic; model Claude Sonnet 5.5), working at the team's direction in a local session. The assistant also ran the pipeline and the verification scripts through shell access on the machine that holds the data bundle. No LLM or other pretrained model is part of the pipeline: the labels come from deterministic code over the tactical ATD and Smart Tagging files, and the `poss` package imports no network library. The team's working rule was that the assistant should work from code, synthetic tests and aggregate status output rather than frame-level data. Some aggregate output was nonetheless visible to the assistant, for example console summaries from the first real run (possession count, dead-ball share, tag agreement), the official validator's summary of the final CSV, and the result of the single public-sample wiring check. The team has not verified what else the assistant saw and makes no claim beyond this.

**Two changes after the first real run (commit `e471aec`, both label-free).** They were prompted by aggregate console output of the first full run on the real bundle, not by any label, the public sample or the private ground truth:
1. *Switch-penalty grid.* λ is picked from a grid by the generic 12 s mean-possession prior. The first run picked the lowest grid value, so the grid was extended downward from (12, 20, 30, 45, 65, 90, 130) to (3, 5, 8, 12, 20, 30, 45, 65, 90, 130). The extended grid still picks its lowest value (3).
2. *Rank-based low-activity fallback.* The slow-player cue is weighted by an unsupervised two-mode fit of player speeds. On the real data the speeds were not bimodal, so the cue switched itself off. A fallback was added for that case only: within each half, activity at or below the 5th percentile scores +1, the 35th percentile scores 0, and clearly higher activity scores negative (scaled by 0.75).

Neither change refers to any frame, time or label. The public-sample wiring check was run after these changes. The v1 output has not changed since: later revisions of the code regenerated it bit-identically, and it is pinned by `tests/test_frozen_v1.py` and by the SHA-256 given with the submission.
