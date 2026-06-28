# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **dataset only** — no source code yet. It holds raw sign-language gesture
recordings captured from **two Myo armbands** (one per arm), each streaming
8-channel surface EMG plus a 9-axis IMU. Intended for training/eval of
sign-language recognition models. Any analysis/training code added later will
consume the CSVs described below.

## Data layout

Everything lives inside `dataset/wgswcr8z24-2.zip` (Mendeley dataset
`wgswcr8z24`), which contains two nested zips:

- `10_ASL_Signs.zip` → `all/User{1..10}/` (User5 absent → **9 users**), **550
  CSVs**, **20 distinct ASL word signs** (bird, blue, cat, cost, dollar, gold,
  goodnight, happy, horse, hot, hurt, large, mom, pizza, please, shirt, wash,
  home, day, orange). Each user recorded only a **subset** of the 20 signs,
  ~3–4 repetitions each — counts are **not uniform** across users or signs.
- `15health_signs_5singers.zip` → `health_signs/HealthSigns_User{1..5}/`, **299
  CSVs**, **15 health-phrase signs** (headache, soreness, swelling, tired,
  cantsleep, coldrunnynose, notfeelgood, upsetstomach, everymorning,
  everynight, allmorning, monthly, continuouslyforanhour, takeliquidmedicine,
  thatsterrible).

Extract with: `cd dataset && unzip -o wgswcr8z24-2.zip && unzip -o 10_ASL_Signs.zip && unzip -o 15health_signs_5singers.zip`

### Label = filename
Each CSV is one recording. The class label is the filename prefix before the
underscore-id: `headache_145367335.csv` → label `headache`. The trailing number
is a capture timestamp/id, not meaningful for training.

## CSV schema (one recording per file)

- Header row + **50 timesteps** (`Counter` 1..50). 35 columns total.
- Layout: `Counter` then 17 channels per arm, suffixed `L` (left arm) then `R`
  (right arm):
  - `EMG0..EMG7` — 8 surface-EMG channels (integer ADC counts)
  - `AX,AY,AZ` — accelerometer (g)
  - `GX,GY,GZ` — gyroscope (deg/s)
  - `OR,OP,OY` — orientation roll / pitch / yaw

So a sample is a `50 × 34`-feature time series (or `50 × 17 × 2` arms).

## Known data quirks — handle these

- **EMG warm-up**: the first 1–2 rows of most files have all EMG channels = 0
  (IMU is already valid). Drop or mask them.
- **4 files have 101 rows** instead of 50 (apparent double-captures): two in
  ASL (`User8/mom_143923240`, `User6/home_143923418`) and two in health
  (`User5/headache_145395028`, `User2/continuouslyforanhour_145392845`).
- **Misspelled label**: `HealthSigns_User4` uses `continuoslyforanhour`
  (missing the second "u") instead of `continuouslyforanhour`. Normalize labels
  before grouping.
- **Per-arm scale differs**: EMG is integer counts; IMU is floating point in
  different units — scale/normalize per channel-group, not globally.

## Gotcha: extracting the zips

The directories inside the zips were archived **read-only** (`dr-xr-xr-x`,
no write bit). To delete an extracted tree you must restore write first:
`find <dir> -type d -exec chmod u+rwx {} + && rm -rf <dir>`.
Extract to a scratch/ignored path — extracted CSVs are not tracked in git.
