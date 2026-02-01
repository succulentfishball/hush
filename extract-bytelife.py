"""
ByteLife — Data Extraction & Preprocessing
============================================
Works with the UKHLS Calendar Year datasets:
  SN 9333 — Calendar Year 2022 (file prefix: lmn_)
  SN 9471 — Calendar Year 2023 (file prefix: mno_)

These are cross-sectional snapshots, so we treat each year independently,
then stack and deduplicate (keeping each person's most recent observation).

Usage:
    python extract_bytelife.py --data_dir /path/to/unzipped/tab/files

Expected folder layout (tab-delimited downloads, unzipped):
    data_dir/
        lmn_indresp.tab      # 2022 individual responses
        lmn_hhresp.tab       # 2022 household responses
        mno_indresp.tab      # 2023 individual responses
        mno_hhresp.tab       # 2023 household responses

Output:
    bytelife_clean.csv       — final preprocessed dataset, ready for model training
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os


# ---------------------------------------------------------------------------
# 1. CONFIGURATION — all variable names in one place
# ---------------------------------------------------------------------------

# Each calendar year dataset uses a different wave-combo prefix for its variables.
# The key table (from the user guides):
#   SN 9333 (2022): prefix = lmn_   (data from Waves 12, 13, 14)
#   SN 9471 (2023): prefix = mno_   (data from Waves 13, 14, 15)

DATASETS = {
    2022: {
        "prefix": "lmn",
        "indresp_file": "lmn_indresp.tab",
        "hhresp_file":  "lmn_hhresp.tab",
    },
    2023: {
        "prefix": "mno",
        "indresp_file": "mno_indresp.tab",
        "hhresp_file":  "mno_hhresp.tab",
    },
}

# Variables we want, expressed as {prefix}_varname.
# We build the actual column names dynamically below.
# "root" names (no prefix needed) are listed separately.
PREFIXED_VARS = [
    "age_dv",           # Age at interview
    "sex_dv",           # Sex (derived)
    "hiqual_dv",        # Highest qualification
    "jbstat",           # Labour force status
    "jbnssec3_dv",      # Occupation: NS-SEC 3-class
    "mastat_dv",        # Marital status (de-facto)
    "nchild_dv",        # Number of own children <16 in household
    "fimnnet_dv",       # Total net monthly income (TARGET)
    "sclfsato",         # Life satisfaction 1–7 (TARGET)
    "hidp",             # Household ID (for merging hhresp)
]

UNPREFIXED_VARS = [
    "pidp",             # Unique person identifier (no prefix in calendar year files)
]

# From hhresp only:
HHRESP_PREFIXED_VARS = [
    "hhsize",           # Household size
    "hidp",             # Household ID (merge key)
]

# Sentinel codes used by UKHLS for "missing" — these are NOT real values.
SENTINEL_CODES = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14]


# ---------------------------------------------------------------------------
# 2. LOAD ONE DATASET
# ---------------------------------------------------------------------------

def load_one_year(year: int, data_dir: str) -> pd.DataFrame:
    """Load indresp + hhresp for a single calendar year, merge, return raw df."""

    cfg = DATASETS[year]
    pfx = cfg["prefix"]

    # --- indresp ---
    indresp_path = os.path.join(data_dir, cfg["indresp_file"])
    if not os.path.exists(indresp_path):
        sys.exit(f"ERROR: Cannot find {indresp_path}")

    # Build expected column names
    prefixed_cols = [f"{pfx}_{v}" for v in PREFIXED_VARS]
    all_ind_cols  = UNPREFIXED_VARS + prefixed_cols

    print(f"  Loading {cfg['indresp_file']} ...")
    indresp = pd.read_csv(indresp_path, sep="\t", usecols=all_ind_cols, low_memory=False)

    # Rename: strip prefix so downstream code is year-agnostic
    rename_map = {f"{pfx}_{v}": v for v in PREFIXED_VARS}
    indresp.rename(columns=rename_map, inplace=True)

    # --- hhresp ---
    hhresp_path = os.path.join(data_dir, cfg["hhresp_file"])
    if not os.path.exists(hhresp_path):
        sys.exit(f"ERROR: Cannot find {hhresp_path}")

    hh_cols = [f"{pfx}_{v}" for v in HHRESP_PREFIXED_VARS]
    print(f"  Loading {cfg['hhresp_file']} ...")
    hhresp = pd.read_csv(hhresp_path, sep="\t", usecols=hh_cols, low_memory=False)
    hhresp.rename(columns={f"{pfx}_{v}": v for v in HHRESP_PREFIXED_VARS}, inplace=True)

    # Merge on hidp (many individuals : one household)
    df = indresp.merge(hhresp[["hidp", "hhsize"]], on="hidp", how="left")
    df["calendar_year"] = year

    print(f"  → {len(df):,} adult respondents loaded for {year}")
    return df


# ---------------------------------------------------------------------------
# 3. STACK YEARS & DEDUPLICATE
# ---------------------------------------------------------------------------

def stack_and_deduplicate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Stack all years. A small number of people appear in both 2022 and 2023
    (the calendar year files have ~3–4% overlap at boundaries). Keep only
    each person's LATEST observation so no individual leaks into both train
    and test.
    """
    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined rows before dedup: {len(df):,}")

    # Sort by year ascending, then take last (= most recent)
    df = (
        df.sort_values("calendar_year")
        .groupby("pidp")
        .last()
        .reset_index()
    )
    print(f"  Unique individuals after dedup: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# 4. CLEAN & PREPROCESS
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinels, derive features, filter, encode."""

    # --- 4a. Replace sentinel codes with NaN ---
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace(SENTINEL_CODES, np.nan)

    # --- 4b. Derive "ever_married" (binary) ---
    # mastat_dv values (after sentinel removal):
    #   1 = married, 2 = cohabiting, 3 = single, 4 = separated,
    #   5 = divorced, 6 = widowed
    # "Ever married" = currently or previously married (1, 4, 5, 6)
    df["ever_married"] = df["mastat_dv"].isin([1, 4, 5, 6]).astype("Int8")

    # --- 4c. Log-transform income ---
    # Clip negatives first (rare: debts > income), then log1p for zero-safety
    df["log_income"] = np.log1p(df["fimnnet_dv"].clip(lower=0))

    # --- 4d. Filter to working-age adults with both targets present ---
    before = len(df)
    df = df.dropna(subset=["fimnnet_dv", "sclfsato"])  # need both targets
    df = df[df["age_dv"].between(25, 65)]              # working-age
    # jbstat: 1=employed FT, 2=employed PT, 3=self-employed
    # Keep only people currently working
    df = df[df["jbstat"].isin([1, 2, 3])]
    print(f"\n  Rows after filtering (working-age, employed, targets present): "
          f"{len(df):,}  (dropped {before - len(df):,})")

    # --- 4e. Select and rename final columns before encoding ---
    # Drop columns we no longer need
    df = df.drop(columns=["hidp", "mastat_dv", "jbstat", "calendar_year"],
                 errors="ignore")

    # --- 4f. One-hot encode categoricals ---
    # sex_dv:      1 = male, 2 = female  → single binary column
    df["sex_female"] = (df["sex_dv"] == 2).astype("Int8")
    df.drop(columns=["sex_dv"], inplace=True)

    # hiqual_dv:   ordinal (1=degree … 5=no qualifications) — keep as-is
    # jbnssec3_dv: 1=managerial/professional, 2=intermediate, 3=routine/manual
    #              → one-hot (drop first to avoid multicollinearity)
    df = pd.get_dummies(df, columns=["jbnssec3_dv"], prefix="occ", drop_first=True)

    # --- 4g. Final column audit ---
    # Expected final columns:
    #   pidp (ID, not a feature)
    #   age_dv, sex_female, hiqual_dv, ever_married, nchild_dv, hhsize   ← inputs
    #   occ_2.0, occ_3.0                                                  ← occupation dummies
    #   fimnnet_dv, log_income, sclfsato                                  ← targets / derived target

    return df


# ---------------------------------------------------------------------------
# 5. SUMMARY STATS
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  BYTELIFE — CLEANED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total individuals:        {len(df):,}")
    print(f"  Missing values per column:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("      None")
    else:
        for col, n in missing.items():
            print(f"      {col}: {n:,} ({100*n/len(df):.1f}%)")

    print(f"\n  --- Input Features ---")
    print(f"  Age:                 {df['age_dv'].min():.0f}–{df['age_dv'].max():.0f} "
          f"(median {df['age_dv'].median():.0f})")
    print(f"  Female:              {df['sex_female'].mean()*100:.1f}%")
    print(f"  Ever married:        {df['ever_married'].mean()*100:.1f}%")
    print(f"  Children (<16):      {df['nchild_dv'].median():.0f} median "
          f"(max {df['nchild_dv'].max():.0f})")
    print(f"  Household size:      {df['hhsize'].median():.0f} median")
    print(f"  Highest qual (1–5):  {df['hiqual_dv'].median():.0f} median")

    print(f"\n  --- Targets ---")
    print(f"  Net monthly income:  £{df['fimnnet_dv'].median():,.0f} median "
          f"(£{df['fimnnet_dv'].quantile(0.1):,.0f} P10 – £{df['fimnnet_dv'].quantile(0.9):,.0f} P90)")
    print(f"  Log income:          {df['log_income'].median():.2f} median")
    print(f"  Life satisfaction:   {df['sclfsato'].median():.0f} median (1–7 scale)")
    print(f"  Satisfaction dist:")
    for score in sorted(df["sclfsato"].dropna().unique()):
        pct = (df["sclfsato"] == score).mean() * 100
        print(f"      {int(score)}: {pct:5.1f}%  {'█' * int(pct)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ByteLife data extraction")
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to folder containing the .tab files"
    )
    parser.add_argument(
        "--output", default="bytelife_clean.csv",
        help="Output CSV path (default: bytelife_clean.csv)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: {args.data_dir} is not a valid directory")

    print("=" * 60)
    print("  BYTELIFE — Extracting & Preprocessing UKHLS Data")
    print("=" * 60)

    # Load each year
    frames = []
    for year in sorted(DATASETS.keys()):
        print(f"\n[{year}]")
        df = load_one_year(year, args.data_dir)
        frames.append(df)

    # Stack & deduplicate
    print("\n[Deduplication]")
    df = stack_and_deduplicate(frames)

    # Clean & preprocess
    print("\n[Preprocessing]")
    df = clean(df)

    # Summary
    print_summary(df)

    # Save
    df.to_csv(args.output, index=False)
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()