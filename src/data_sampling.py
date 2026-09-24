import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = os.path.expanduser(
    "~/Downloads/US_Accidents_March23.csv"
)

OUTPUT_FILE = "data/US_Accidents_1M.csv"

TARGET_ROWS = 1_000_000
CHUNK_SIZE = 100_000
RANDOM_SEED = 42


# ============================================================
# Step 1: Count records by Year × Severity
# ============================================================

print("=" * 70)
print("US ACCIDENTS - STRATIFIED SAMPLING")
print("=" * 70)

print("\n[1/2] Calculating Year × Severity distribution...")

group_counts = {}

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["Start_Time", "Severity"],
    chunksize=CHUNK_SIZE
):

    chunk["Year"] = pd.to_datetime(
        chunk["Start_Time"],
        errors="coerce"
    ).dt.year

    counts = chunk.groupby(
        ["Year", "Severity"]
    ).size()

    for group, count in counts.items():
        group_counts[group] = (
            group_counts.get(group, 0) + count
        )


group_counts = pd.Series(group_counts).sort_index()

total_rows = int(group_counts.sum())

sampling_fraction = TARGET_ROWS / total_rows

print(f"\nOriginal records: {total_rows:,}")

print("\nYear × Severity distribution:")
print(group_counts)

print(
    f"\nSampling fraction: "
    f"{sampling_fraction:.6f}"
)


# ============================================================
# Step 2: Create stratified sample
# ============================================================

print("\n[2/2] Creating sample...")

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

rng = np.random.default_rng(RANDOM_SEED)

first_write = True
sampled_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    chunk["Year"] = pd.to_datetime(
        chunk["Start_Time"],
        errors="coerce"
    ).dt.year

    selected = np.zeros(
        len(chunk),
        dtype=bool
    )

    for _, group in chunk.groupby(
        ["Year", "Severity"],
        sort=False
    ):

        random_values = rng.random(len(group))

        mask = random_values < sampling_fraction

        selected[group.index - chunk.index[0]] = mask

    sampled_chunk = chunk.loc[selected].copy()

    sampled_chunk.drop(
        columns=["Year"],
        inplace=True
    )

    if len(sampled_chunk) > 0:

        sampled_chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False

        sampled_rows += len(sampled_chunk)

    if chunk_number % 10 == 0:

        print(
            f"Processed {chunk_number} chunks | "
            f"Sampled: {sampled_rows:,}"
        )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("SAMPLING COMPLETE")
print("=" * 70)

print(f"Original records : {total_rows:,}")
print(f"Sampled records  : {sampled_rows:,}")
print(f"Output file      : {OUTPUT_FILE}")
print(f"Random seed      : {RANDOM_SEED}")

print("=" * 70)