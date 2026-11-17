# Centrale Pride — MTAH 2027 Phase 1 (ball possession)

Per-frame labels `H/A/D/X` for the BAR–ATH tactical video. Training-free temporal state model: soft ball-control
evidence + Viterbi hysteresis, corroborated stoppage detection, Smart Tagging as capped soft evidence; profile `v3`
adds optional signals (broadcast tracking, tactical video/audio, broadcast-video ball detections) that are each applied
only if they pass predeclared verification checks. No model is trained on any labels.
Technical note: `docs/TECHNICAL_NOTE.md`. Challenge: https://github.com/metrica-sports/more-than-a-hack-2027

The match data is **not** included and must not be committed (see `.gitignore`).

## Reproduce (needs the gated data bundle)
Layout: `<DATA>/tactical/BAR-ATH_tactical{.mp4,_atd.txt,_atd.xml,_homography.csv,_smart_tagging.xml}` and
`<DATA>/broadcast/BAR-ATH_broadcast{.mp4,_atd.txt,_atd.xml,_homography.csv}` (profiles v1/v2 need only the tactical ATD files).
```bash
pip install -r requirements-full.txt      # pinned; Python 3.14 tested
python -m poss.make_submission --data-dir <DATA> --team-name "Centrale Pride" --profile v1  --out outputs
python evaluation/validate_submission.py "outputs/Centrale Pride.csv"
python -m pytest -q tests                 # synthetic wiring tests, no match data
```
Deterministic: the same bundle and commit give the same file (compare SHA-256). Intermediate arrays are cached under
`<DATA>/cache` (never commit it); deleting the cache forces recomputation from the raw inputs.
`games.py`, `evaluation/`, `starter/` are vendored unmodified from the official repository (MIT).
