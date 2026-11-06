"""
Step 15 — Compare the detected panel area with existing installation data (Leuven vs Gent).

Your TA's second point: compare with data that already exists. Official statistics (e.g.
Fluvius open data 'lokale productie-installaties per gemeente') give, per municipality and
technology, the NUMBER of installations and the installed POWER (in kVA: inverter power).
We measure AREA. To compare, convert power to an expected area:

    expected m² = power (kVA) x kWp-per-kVA x m²-per-kWp

  kWp per kVA : panels (kWp) are often sized somewhat above the inverter (kVA).
                Default 1.0; try 1.0-1.3 and show the range (sensitivity).
  m² per kWp  : depends on panel age/efficiency. Low end ~4.0 (a modern 445 Wp panel of
                1.0 x 1.8 m, as in the Zonnekaart's example); high end = median of the
                BDAPPV metadata (older French installations, step 02).

Two comparisons, from weak to strong claims:
  1. RELATIVE (robust): is the ratio Leuven/Gent of detected m² per km² similar to the ratio
     of installed power per km² in the statistics? Unit conversions cancel out.
  2. ABSOLUTE (fragile): is the detected m² inside the expected m² range? If not, say why
     (photo year vs statistics date, threshold, tilt, missed small panels, kVA vs kWp...).

The installation file is YOURS: tell the script its column names, e.g. for a Fluvius export:
  python 15_compare_installations.py --run unet_final --cities leuven gent \
      --installations data/flanders/installations.csv --sep ";" \
      --muni-col Hoofdgemeente --count-col "Aantal installaties" --power-col "Vermogen (kVA)" \
      --tech-col Technologie --tech-value "Zonne-energie" --muni-names Leuven Gent
(These column names are EXAMPLES. Open your CSV and use the real ones.)

Requires: 14_area_statistics.py for every city
Output:   outputs/installation_comparison.csv, plots/installation_comparison.png
"""

import argparse
import json

import numpy as np
import pandas as pd

from common import OUTPUTS_DIR, PLOTS_DIR, PROCESSED_DIR, ensure_dirs

MODERN_M2_PER_KWP = 4.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--cities", nargs="+", required=True)
    p.add_argument("--installations", required=True)
    p.add_argument("--sep", default=",")
    p.add_argument("--muni-col", required=True)
    p.add_argument("--count-col", required=True)
    p.add_argument("--power-col", required=True)
    p.add_argument("--tech-col", default=None)
    p.add_argument("--tech-value", default=None)
    p.add_argument("--filter-col", default=None, help="e.g. a date column, to keep one reference date")
    p.add_argument("--filter-value", default=None)
    p.add_argument("--muni-names", nargs="+", default=None, help="names in the CSV, same order as --cities")
    p.add_argument("--kwp-per-kva", type=float, nargs=2, default=[1.0, 1.0], metavar=("LOW", "HIGH"))
    a = p.parse_args()
    ensure_dirs()

    inst = pd.read_csv(a.installations, sep=a.sep, dtype=str)
    for col in (a.muni_col, a.count_col, a.power_col, a.tech_col, a.filter_col):
        if col and col not in inst.columns:
            raise SystemExit(f"Column {col!r} not in {a.installations}. Columns: {list(inst.columns)}")
    if a.tech_col:
        inst = inst[inst[a.tech_col].str.strip().str.lower() == a.tech_value.lower()]
    if a.filter_col:
        inst = inst[inst[a.filter_col].astype(str).str.strip() == a.filter_value]
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                                  if s.astype(str).str.contains(",").any() else s, errors="coerce")
    inst[a.count_col], inst[a.power_col] = num(inst[a.count_col]), num(inst[a.power_col])

    factors = json.loads((PROCESSED_DIR / "area_factors.json").read_text()) \
        if (PROCESSED_DIR / "area_factors.json").exists() else {}
    m2_high = factors.get("m2_per_kwp_bdappv_median", 7.0)
    m2_low = MODERN_M2_PER_KWP
    print(f"m² per kWp range: {m2_low:.1f} (modern panels) - {m2_high:.1f} (BDAPPV median); "
          f"kWp per kVA: {a.kwp_per_kva[0]}-{a.kwp_per_kva[1]}")

    names = a.muni_names or [c.capitalize() for c in a.cities]
    rows = []
    for city, name in zip(a.cities, names):
        s = json.loads((OUTPUTS_DIR / "flanders" / city / f"summary_{a.run}.json").read_text())
        sel = inst[inst[a.muni_col].str.strip().str.lower() == name.lower()]
        if sel.empty:
            raise SystemExit(f"No rows for {name!r} in column {a.muni_col}. Check --muni-names.")
        count, power = float(sel[a.count_col].sum()), float(sel[a.power_col].sum())
        km2 = s["municipality_km2"] or np.nan
        rows.append({
            "city": city, "municipality_km2": km2,
            "detected_projected_m2": s["projected_m2_roof"], "detected_real_m2": s["real_m2_est"],
            "detected_bias_corrected_m2": s["bias_corrected_m2"],
            "installations": count, "power_kVA": power,
            "expected_m2_low": power * a.kwp_per_kva[0] * m2_low,
            "expected_m2_high": power * a.kwp_per_kva[1] * m2_high,
            "detected_m2_per_km2": s["real_m2_est"] / km2, "installations_per_km2": count / km2,
            "kVA_per_km2": power / km2, "detected_m2_per_installation": s["real_m2_est"] / max(count, 1)})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS_DIR / "installation_comparison.csv", index=False)
    with pd.option_context("display.float_format", "{:,.1f}".format, "display.width", 200):
        print(df.T.to_string())

    if len(df) >= 2:
        c1, c2 = df.iloc[0], df.iloc[1]
        print(f"\nRelative comparison {c1.city}/{c2.city}:")
        print(f"  detected panel m² per km²    : {c1.detected_m2_per_km2 / c2.detected_m2_per_km2:.2f}")
        print(f"  installations per km² (data) : {c1.installations_per_km2 / c2.installations_per_km2:.2f}")
        print(f"  installed kVA per km² (data) : {c1.kVA_per_km2 / c2.kVA_per_km2:.2f}")
        print("  Similar ratios = the map captures the real difference between the cities.")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.detected_real_m2 / 1e3, 0.4, label="detected (÷cos tilt)")
    ax.errorbar(x - 0.2, df.detected_real_m2 / 1e3,
                yerr=[(df.detected_real_m2 - df.detected_projected_m2) / 1e3, np.zeros(len(df))],
                fmt="none", color="k", capsize=4)
    ax.bar(x + 0.2, (df.expected_m2_low + df.expected_m2_high) / 2e3, 0.4, color="lightgrey",
           label="expected from installed power")
    ax.errorbar(x + 0.2, (df.expected_m2_low + df.expected_m2_high) / 2e3,
                yerr=(df.expected_m2_high - df.expected_m2_low) / 2e3, fmt="none", color="k", capsize=4)
    ax.set_xticks(x, df.city)
    ax.set_ylabel("panel area (1000 m²)")
    ax.set_title("Detected rooftop panel area vs installation statistics")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "installation_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {OUTPUTS_DIR / 'installation_comparison.csv'} and {PLOTS_DIR / 'installation_comparison.png'}")


if __name__ == "__main__":
    main()
