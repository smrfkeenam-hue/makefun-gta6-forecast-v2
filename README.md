# GTA VI Twitch CCU forecaster — v2

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/smrfkeenam-hue/makefun-gta6-forecast-v2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/smrfkeenam-hue/makefun-gta6-forecast-v2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for [make.fun `gta6-twitch-ccu-forecast-01`](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): forecast, for hours 2–24 of
the UTC release day (2026-11-19), five quantiles (5/25/50/75/95%) of Twitch
concurrent viewers in the Grand Theft Auto VI category, from hour 1's print
and the past launches bundled in each case as `comparables`. Score is pinball
loss on the log scale (lower is better). Python 3.13, standard library only,
no network, deterministic.

## Model (v2 — ensemble, fitted at run time from the case's comparables)

An equal-weight ensemble, in log-quantile space, of three components:

1. **v1 blend.** Per-hour level regression of the log-ratio r(h) =
   ln(viewers(h)/viewers(1)) on log(hour-1) across filtered donors (bigger
   launches are flatter, corr −0.82), blended 50/50 with the median curve of
   the 6 nearest donors in log(hour-1). Spread 0.7 × residual std, floor
   0.10. The target's effective log(hour-1) is capped at the largest
   rolling-like donor's (The Last of Us Part II, 365k).
2. **Partial-alignment blend.** Same blend, except morning hours (2–9) are
   aligned by hours since the US-East unlock instead of UTC hour: donors are
   pooled in u = h − east_h coordinates (rolling launches only) and read off
   at the target's own unlock hour. This puts the US unlock jump at the
   right hour for the target's DST regime (h5 in EDT, h6 in EST) instead of
   smearing it across both. The evening keeps UTC hours, where the diurnal
   pattern dominates.
3. **Heteroskedastic blend.** Component 2 with a size-conditional spread:
   the level-regression residual variance falls steeply with launch size in
   the midday/evening (log-variance slope ≈ −0.7/−1.0), so small targets get
   a wider spread and large targets a narrower one (multiplier clipped to
   [0.5, 2]).

**Donor filter** (as v1): drop tier `target-history`,
`early-access-contaminated`, and `weak`; drop launches whose hour-1 print is
under 2% of their release-day peak; require a usable hour-1 print and ≥12
release-day hours. Keeps 19 of 38 comparables on the practice cases.

**Safety net:** each ensemble member runs in isolation; a member that raises
or returns a non-finite/non-positive quantile is dropped, and if all three
fail the forecast falls back to the house baseline (persistence). An invalid
case takes no prize, so a degraded forecast always beats a crash.

## Scores (mean pinball loss, lower is better)

| | p00 | p01 | p02 | p03 | mean |
|---|---|---|---|---|---|
| v2 (this entry) | 0.240 | 0.118 | 0.095 | 0.182 | **0.159** |
| v1 | 0.237 | 0.126 | 0.104 | 0.190 | 0.164 |
| house baseline | | | | | 0.387 |

Coverage (90% interval): p00 0.96, p01 0.96, p02 0.91, p03 0.83.
Leave-one-donor-out (19 pseudo-targets): v2 0.278 vs v1 0.280.

Honest note: the gain over v1 is modest (~0.005) and within the noise of 4
practice cases; v2 does not reach the ~0.139 top-20 practice bar. The build
tested ~25 fresh angles (unlock-wave regression, rolling-only donors,
channels decomposition, empirical quantiles, pre-release-peak features,
functional PCA, asymmetric spreads, platform features, pre-peak anchoring)
and found the remaining loss is almost all median error, while the launch
amplitude (peak/hour-1) varies 3.4×–12.1× within a size band driven by the
US-audience share — which is unobservable at forecast time, because the US
has not unlocked yet at hour 1. That part of the error is largely
irreducible from the case data.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `*.out` per case.
Runs in ~0.1 s per case; output is byte-identical across runs.
