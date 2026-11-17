"""GTA VI Twitch CCU forecaster, v2, for make.fun gta6-twitch-ccu-forecast-01.

Forecasts, for hours 2..24 of the UTC release day, five quantiles (5/25/50/
75/95%) of Twitch concurrent viewers in the Grand Theft Auto VI category,
from hour 1's print and the past launches bundled in the case as
`comparables`. Python 3.13, standard library only, no network, deterministic.

The model (my own work, fitted at run time from the case's comparables) is
an equal-weight ensemble, in log-quantile space, of three components:

  1. v1 blend -- per-hour level regression of the log-ratio r(h) =
     ln(viewers(h)/viewers(1)) on log(hour-1) across filtered donors
     (bigger launches are flatter, corr -0.82), blended 50/50 with the
     median curve of the 6 nearest donors in log(hour-1). Spread 0.7 x
     residual std, floor 0.10. The target's effective log(hour-1) is capped
     at the largest rolling-like donor's (The Last of Us Part II, 365k), so
     the slope is never extrapolated past the biggest rolling launch.
  2. Partial-alignment blend -- the same blend, except the morning hours
     (2..9) are aligned by hours since the US-East unlock instead of UTC
     hour: donors are pooled in u = h - east_h coordinates (rolling
     launches only) and read off at the target's own unlock hour. This puts
     the US unlock jump at the right hour for the target's DST regime
     (h5 in EDT, h6 in EST) instead of smearing it across both. The evening
     keeps UTC hours, where the diurnal pattern dominates and does not shift
     with US DST.
  3. Heteroskedastic blend -- component 2 with a size-conditional spread:
     the level-regression residual variance falls steeply with launch size
     in the midday/evening (log-variance slope about -0.7/-1.0), so small
     targets get a wider spread and large targets a narrower one
     (multiplier exp(0.5 * gamma_band * (lp1 - mean log v1)), clipped to
     [0.5, 2]).

Donor filter (as v1): drop tier `target-history`, `early-access-
contaminated` and `weak`; drop launches whose hour-1 print is under 2% of
their release-day peak (late simultaneous unlocks); require a usable hour-1
print and at least 12 release-day hours. On the practice cases this keeps
19 of 38 comparables.

Practice score (mean pinball loss on the log scale, lower is better),
scored end to end through the grader's file interface (with 2-decimal
rounding): v2 0.159 (p00 0.240, p01 0.118, p02 0.095, p03 0.182) vs v1
0.164 and the house baseline 0.387. Leave-one-donor-out (19
pseudo-targets) 0.278 vs v1 0.280. The gain is modest and within the
noise of 4 practice cases; the honest finding of this build is that the
remaining loss is almost all median error, and the launch amplitude
(peak/hour-1) varies 3.4x-12.1x within a size band driven by the
US-audience share, which is unobservable at forecast time (the US has
not unlocked yet at hour 1), so it is largely irreducible from the
case data.

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out
to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
HOURS = list(range(2, 25))
SPREAD_K = 0.7
SPREAD_FLOOR = 0.10
KNN_K = 6
MORNING_END = 9
BLEND_W = 0.5


def round2(x):
    """Round half up to 2 decimals; whole numbers come back as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 100.0
    return int(v) if v == int(v) else v


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def us_dst(day_start):
    """True if US daylight saving time is in effect on the release date.

    DST runs from the second Sunday of March to the first Sunday of
    November (2:00 local). Computed from the UTC date of day_start.
    """
    d = parse_time(day_start).date()
    y = d.year

    def nth_sunday(month, n):
        first = datetime(y, month, 1, tzinfo=timezone.utc).date()
        # days to the first Sunday (weekday(): Mon=0 .. Sun=6)
        off = (6 - first.weekday()) % 7
        return first + timedelta(days=off + 7 * (n - 1))

    start = nth_sunday(3, 2)
    end = nth_sunday(11, 1)
    return start <= d < end


def day_curve(comp, field="viewers"):
    """{hour 1..24: value} of the comparable's release day (UTC)."""
    ds = parse_time(comp["day_start"])
    out = {}
    for p in comp["hourly"]:
        t = parse_time(p["from"])
        h = int((t - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and p.get(field) is not None:
            out[h] = p[field]
    return out


def tier_of(comp):
    m = re.search(r"Tier:\s*([^.]+)", comp.get("unlock", ""))
    return m.group(1).strip().lower() if m else ""


def donor_curves(case):
    """Filtered donor log-ratio curves {name: {hour: ln(v(h)/v(1))}} and
    their hour-1 prints {name: v1}. (The v1 donor filter.)"""
    out, v1s = {}, {}
    for m in case["comparables"]:
        t = tier_of(m)
        if t in ("target-history", "early-access-contaminated", "weak"):
            continue
        cv = day_curve(m)
        v1 = cv.get(1)
        if not v1 or v1 < 1:
            continue
        peak = max(v for v in cv.values() if v is not None)
        if v1 < 0.02 * peak:
            continue
        if len([v for v in cv.values() if v is not None]) < 12:
            continue
        out[m["name"]] = {h: math.log(max(1, v) / v1)
                          for h, v in cv.items() if v is not None}
        v1s[m["name"]] = v1
    return out, v1s


def _ols1(pairs):
    """Least squares y ~ a + b*x. Returns (a, b)."""
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    b = sxy / sxx if sxx > 0 else 0.0
    return my - b * mx, b


def _rolling_cap(case, donors, v1s):
    """Cap on the target's effective log(hour-1): the largest rolling-like
    donor's log(hour-1) (release-day peak at hour 15 or later)."""
    by_name = {m["name"]: m for m in case["comparables"]}
    caps = []
    for name in donors:
        cv = day_curve(by_name[name])
        peak_h = max(cv, key=lambda h: cv[h] or -1)
        if peak_h >= 15:
            caps.append(math.log(max(1, v1s[name])))
    return max(caps) if caps else None


def _level_reg_component(donors, v1s, lp1_eff, k=SPREAD_K, floor=SPREAD_FLOOR):
    """Per-hour level regression. Returns {hour: (median log-ratio, spread)}."""
    slopes = {}
    for h in HOURS:
        pairs = [(math.log(max(1, v1s[n])), c[h])
                 for n, c in donors.items() if h in c]
        if len(pairs) < 5:
            continue
        slopes[h] = _ols1(pairs)[1]
    out = {}
    for h in HOURS:
        pairs = [(math.log(max(1, v1s[n])), c[h])
                 for n, c in donors.items() if h in c]
        if len(pairs) < 5:
            out[h] = (0.0, floor)
            continue
        nb = [slopes[hh] for hh in (h - 1, h, h + 1) if hh in slopes]
        b = sum(nb) / len(nb) if nb else 0.0
        a = median([y - b * x for x, y in pairs])
        med = a + b * lp1_eff
        res = [y - (a + b * x) for x, y in pairs]
        sd = (sum(r * r for r in res) / max(1, len(res) - 1)) ** 0.5
        out[h] = (med, max(floor, k * sd))
    return out


def _knn_component(donors, v1s, lp1_eff, k=KNN_K, spread_k=SPREAD_K,
                   floor=SPREAD_FLOOR):
    """Median curve of the k nearest donors in log(hour-1).
    Returns {hour: (median log-ratio, spread)}."""
    near = sorted(donors, key=lambda n: abs(math.log(max(1, v1s[n])) - lp1_eff))
    near = near[:k]
    out = {}
    for h in HOURS:
        rs = [donors[n][h] for n in near if h in donors[n]]
        if not rs:
            out[h] = (0.0, floor)
            continue
        med = median(rs)
        sd = (sum((r - med) ** 2 for r in rs) / max(1, len(rs) - 1)) ** 0.5
        out[h] = (med, max(floor, spread_k * sd))
    return out


def _blend_quantiles(reg_qs, knn, p1, w=BLEND_W):
    """Blend a regression forecast ({hour: [q...]}) with a kNN component
    ({hour: (med, spread)}) in log space, weight w on the regression."""
    out = {}
    for h in HOURS:
        mk, sk = knn.get(h, (0.0, SPREAD_FLOOR))
        out[h] = [math.exp(w * math.log(reg_qs[h][i]) +
                           (1 - w) * math.log(p1 * math.exp(mk + Z[i] * sk)))
                  for i in range(5)]
    return out


def _reg_forecast(reg, p1):
    return {h: [p1 * math.exp(reg[h][0] + z * reg[h][1]) for z in Z]
            for h in HOURS}


def v1_blend(case, donors, v1s):
    """Component 1: level regression + kNN, UTC hours."""
    p1 = max(1, case["known"][0]["viewers"])
    lp1 = math.log(p1)
    cap = _rolling_cap(case, donors, v1s)
    lp1_eff = min(lp1, cap) if cap is not None else lp1
    reg = _level_reg_component(donors, v1s, lp1_eff)
    knn = _knn_component(donors, v1s, lp1_eff)
    return _blend_quantiles(_reg_forecast(reg, p1), knn, p1)


def _donor_unlock_map(case, donors):
    """{donor: (US-East unlock hour, US-West unlock hour)} from each
    donor's DST regime: EDT -> (5, 8), EST -> (6, 9)."""
    by_name = {m["name"]: m for m in case["comparables"]}
    out = {}
    for name in donors:
        out[name] = (5, 8) if us_dst(by_name[name]["day_start"]) else (6, 9)
    return out


def _target_unlock(case):
    return (5, 8) if us_dst(case["day_start"]) else (6, 9)


def rolling_only_donors(case, donors, v1s):
    """Donors whose release-day peak falls at hour 15 or later (rolling
    midnight-local shape, not simultaneous global)."""
    by_name = {m["name"]: m for m in case["comparables"]}
    rd, rv = {}, {}
    for name, c in donors.items():
        cv = day_curve(by_name[name])
        if max(cv, key=lambda h: cv[h] or -1) >= 15:
            rd[name] = c
            rv[name] = v1s[name]
    return rd, rv


def partial_align(case, donors, v1s, morning_end=MORNING_END,
                  k=SPREAD_K, floor=SPREAD_FLOOR):
    """Partial unlock alignment: morning hours aligned by hours since the
    US-East unlock (rolling donors only), evening in UTC hours."""
    p1 = max(1, case["known"][0]["viewers"])
    lp1 = math.log(p1)
    cap = _rolling_cap(case, donors, v1s)
    lp1_eff = min(lp1, cap) if cap is not None else lp1
    te, _ = _target_unlock(case)
    rd, rv = rolling_only_donors(case, donors, v1s)
    umap = _donor_unlock_map(case, rd)
    u_data = {}
    for name, c in rd.items():
        eh, _ = umap[name]
        for h, r in c.items():
            if h < 2:
                continue
            u_data.setdefault(h - eh, []).append(
                (math.log(max(1, rv[name])), r))
    u_pred = {}
    for u, pairs in u_data.items():
        if len(pairs) < 3:
            continue
        _, b = _ols1(pairs)
        a_med = median([y - b * x for x, y in pairs])
        med = a_med + b * lp1_eff
        res = [y - (a_med + b * x) for x, y in pairs]
        sd = (sum(r * r for r in res) / max(1, len(res) - 1)) ** 0.5
        u_pred[u] = (med, max(floor, k * sd))
    reg = _level_reg_component(donors, v1s, lp1_eff, k, floor)
    out = {}
    for h in HOURS:
        if h <= morning_end:
            u = h - te
            med, s = u_pred.get(u, reg.get(h, (0.0, floor)))
        else:
            med, s = reg.get(h, (0.0, floor))
        out[h] = [p1 * math.exp(med + z * s) for z in Z]
    return out


def partial_align_blend(case, donors, v1s, morning_end=MORNING_END, w=BLEND_W):
    """Component 2: partial alignment blended 50/50 with the kNN."""
    p1 = max(1, case["known"][0]["viewers"])
    lp1 = math.log(p1)
    cap = _rolling_cap(case, donors, v1s)
    lp1_eff = min(lp1, cap) if cap is not None else lp1
    reg = partial_align(case, donors, v1s, morning_end=morning_end)
    knn = _knn_component(donors, v1s, lp1_eff)
    return _blend_quantiles(reg, knn, p1, w)


def _band_of(h):
    return "morning" if h < 10 else ("midday" if h < 15 else "evening")


def _band_gamma(donors, v1s):
    """Per-band size-variance slope: regress log(residual^2) on log(v1)
    using the level-regression residuals pooled within each band.
    Returns {band: (gamma, mean_log_v1)}."""
    bands = {"morning": range(2, 10), "midday": range(10, 15),
             "evening": range(15, 25)}
    # Fallback mean log(hour-1), computed from this case's donors rather
    # than hardcoded: the final case's donor pool differs from practice.
    mean_lv1 = sum(math.log(max(1, v)) for v in v1s.values()) / max(1, len(v1s))
    out = {}
    for band, hours in bands.items():
        pts = []
        for h in hours:
            pairs = [(math.log(max(1, v1s[n])), c[h])
                     for n, c in donors.items() if h in c]
            if len(pairs) < 5:
                continue
            _, b = _ols1(pairs)
            a_med = median([y - b * x for x, y in pairs])
            for x, y in pairs:
                r = y - (a_med + b * x)
                pts.append((x, math.log(max(1e-6, r * r))))
        if len(pts) < 20:
            out[band] = (0.0, mean_lv1)
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        out[band] = (sxy / sxx if sxx > 0 else 0.0, mx)
    return out


def hetero_blend(case, donors, v1s, morning_end=MORNING_END, w=BLEND_W,
                 clip=(0.5, 2.0)):
    """Component 3: partial-alignment blend + size-conditional spread."""
    p1 = max(1, case["known"][0]["viewers"])
    lp1 = math.log(p1)
    cap = _rolling_cap(case, donors, v1s)
    lp1_eff = min(lp1, cap) if cap is not None else lp1
    base = partial_align_blend(case, donors, v1s, morning_end=morning_end, w=w)
    gammas = _band_gamma(donors, v1s)
    out = {}
    for h in HOURS:
        gamma, mlv = gammas[_band_of(h)]
        mult = math.exp(0.5 * gamma * (lp1_eff - mlv))
        mult = min(clip[1], max(clip[0], mult))
        med = math.log(base[h][2])
        out[h] = [math.exp(med + (math.log(base[h][i]) - med) * mult)
                  for i in range(5)]
    return out


def forecast(case):
    """The v2 ensemble: equal-weight log-quantile average of the three
    components. Returns {hour: [q05, q25, q50, q75, q95]} in viewers.

    Safety net: each member is run in isolation, and a member that raises
    or returns a non-finite/non-positive quantile is dropped; if all three
    fail, the forecast falls back to the house baseline (persistence).
    An invalid case takes no prize, so a degraded forecast always beats
    a crash.
    """
    p1 = max(1, case["known"][0]["viewers"])

    def baseline():
        return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
                for h in HOURS}

    donors, v1s = donor_curves(case)
    if not donors:
        return baseline()
    members = []
    for fn in (v1_blend, partial_align_blend, hetero_blend):
        try:
            m = fn(case, donors, v1s)
            ok = all(
                len(m[h]) == 5 and
                all(math.isfinite(q) and q > 0 for q in m[h])
                for h in HOURS)
            if ok:
                members.append(m)
        except Exception:
            continue
    if not members:
        return baseline()
    n = len(members)
    out = {}
    for h in HOURS:
        out[h] = [math.exp(sum(math.log(m[h][i]) for m in members) / n)
                  for i in range(5)]
    return out


def clean_quantiles(qs):
    """Clamp each quantile to (0, 1e9], round half-up to 2 decimals, and
    force non-decreasing order."""
    out = []
    prev = 0.01
    for q in qs:
        q = min(1e9, max(0.01, q))
        q = round2(q)
        q = max(q, prev)
        out.append(q)
        prev = q
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text())
        fc = forecast(case)
        hours = [{"hour": h, "q": clean_quantiles(fc[h])} for h in HOURS]
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps({"hours": hours}))


if __name__ == "__main__":
    main()
