"""
Preprocess MTA Subway Hourly Ridership Data
Aggregates duplicate entries by timestamp and station complex ID.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def clean_ridership_value(value):
    """Convert ridership string to integer, handling commas."""
    if isinstance(value, int):
        return value
    return int(str(value).replace(',', ''))

def preprocess_ridership_data(
    input_file="MTA_Subway_Hourly_Ridership__2020-2024_20260131.csv",
    output_prefix="mta_ridership_clean"
):
    """
    Aggregate ridership data by timestamp and station_complex_id.
    
    Args:
        input_file: Path to raw MTA CSV file
        output_prefix: Prefix for output files
    """
    
    print("=" * 70)
    print("MTA Ridership Data Preprocessing")
    print("=" * 70)
    
    # Load data
    print(f"\n1. Loading data from {input_file}...")
    df = pd.read_csv(input_file, parse_dates=["transit_timestamp"])
    print(f"   ✓ Loaded {len(df):,} rows")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    # Show sample of raw data
    print("\n2. Sample of raw data (first 3 rows):")
    print(df.head(3))
    
    # Check for duplicates
    print("\n3. Checking for duplicate entries...")
    duplicate_keys = df.groupby(['transit_timestamp', 'station_complex_id']).size()
    num_duplicates = (duplicate_keys > 1).sum()
    print(f"   ✓ Found {num_duplicates:,} timestamp+station pairs with multiple entries")
    
    if num_duplicates > 0:
        print("\n   Example duplicates:")
        example_dups = duplicate_keys[duplicate_keys > 1].head(3)
        for (ts, station_id), count in example_dups.items():
            print(f"     - {ts} | Station {station_id}: {count} entries")
            sample = df[(df['transit_timestamp'] == ts) & (df['station_complex_id'] == station_id)]
            print(f"       Ridership values: {sample['ridership'].tolist()}")
    
    # Clean ridership values
    print("\n4. Cleaning ridership values (removing commas)...")
    df['ridership'] = df['ridership'].apply(clean_ridership_value)
    print(f"   ✓ Converted to integers")
    
    # Aggregate by timestamp and station_complex_id
    print("\n5. Aggregating data...")
    print("   - Grouping by: transit_timestamp + station_complex_id")
    print("   - Ridership: SUM")
    print("   - Dropping: Georeference (station-level, not entry-level)")
    
    # Keep only relevant columns for aggregation
    agg_df = df[['transit_timestamp', 'station_complex_id', 'ridership']].copy()
    
    # Group and sum ridership
    aggregated = (
        agg_df
        .groupby(['transit_timestamp', 'station_complex_id'], as_index=False)
        .agg({'ridership': 'sum'})
    )
    
    print(f"   ✓ Aggregated to {len(aggregated):,} unique entries")
    print(f"   ✓ Reduction: {len(df) - len(aggregated):,} duplicate rows removed")
    
    # Sort by timestamp and station
    print("\n6. Sorting data...")
    aggregated = aggregated.sort_values(['transit_timestamp', 'station_complex_id'])
    
    # Add time features
    print("\n7. Adding time features...")
    aggregated['hour'] = aggregated['transit_timestamp'].dt.hour
    aggregated['day_of_week'] = aggregated['transit_timestamp'].dt.dayofweek
    aggregated['is_weekend'] = aggregated['day_of_week'].isin([5, 6]).astype(int)
    aggregated['date'] = aggregated['transit_timestamp'].dt.date
    
    # Calculate statistics
    print("\n8. Data statistics:")
    print(f"   - Date range: {aggregated['transit_timestamp'].min()} to {aggregated['transit_timestamp'].max()}")
    print(f"   - Unique stations: {aggregated['station_complex_id'].nunique()}")
    print(f"   - Unique timestamps: {aggregated['transit_timestamp'].nunique()}")
    print(f"   - Total ridership: {aggregated['ridership'].sum():,}")
    print(f"   - Average ridership per entry: {aggregated['ridership'].mean():.1f}")
    print(f"   - Min ridership: {aggregated['ridership'].min()}")
    print(f"   - Max ridership: {aggregated['ridership'].max():,}")
    
    # Save full aggregated dataset
    output_full = f"{output_prefix}_full.csv"
    print(f"\n9. Saving full aggregated dataset to {output_full}...")
    aggregated.to_csv(output_full, index=False)
    print(f"   ✓ Saved {len(aggregated):,} rows")
    
    # Split by year for easier handling
    print("\n10. Splitting by year...")
    aggregated['year'] = aggregated['transit_timestamp'].dt.year
    
    for year in sorted(aggregated['year'].unique()):
        year_data = aggregated[aggregated['year'] == year].copy()
        year_data = year_data.drop('year', axis=1)
        
        output_year = f"{output_prefix}_{year}.csv"
        year_data.to_csv(output_year, index=False)
        print(f"   ✓ {year}: {len(year_data):,} rows → {output_year}")
    
    # Create train/test split (80/20 by time)
    print("\n11. Creating train/test split (80/20 by time)...")
    unique_timestamps = sorted(aggregated['transit_timestamp'].unique())
    split_idx = int(len(unique_timestamps) * 0.8)
    split_time = unique_timestamps[split_idx]
    
    train_df = aggregated[aggregated['transit_timestamp'] < split_time].copy()
    test_df = aggregated[aggregated['transit_timestamp'] >= split_time].copy()
    
    output_train = f"{output_prefix}_train.csv"
    output_test = f"{output_prefix}_test.csv"
    
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    
    print(f"   ✓ Train: {len(train_df):,} rows ({len(train_df)/len(aggregated)*100:.1f}%) → {output_train}")
    print(f"   ✓ Test:  {len(test_df):,} rows ({len(test_df)/len(aggregated)*100:.1f}%) → {output_test}")
    print(f"   ✓ Split at: {split_time}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Input:  {len(df):,} rows (with duplicates)")
    print(f"Output: {len(aggregated):,} rows (deduplicated)")
    print(f"\nFiles created:")
    print(f"  1. {output_full} (full dataset)")
    print(f"  2. {output_train} (training set)")
    print(f"  3. {output_test} (test set)")
    for year in sorted(aggregated['year'].unique()):
        print(f"  4. {output_prefix}_{year}.csv (year {year})")
    
    print("\n✅ Preprocessing complete!")
    print("=" * 70)
    
    return aggregated


if __name__ == "__main__":
    # Run preprocessing
    df = preprocess_ridership_data()
    
    # Show sample of output
    print("\nSample of cleaned data:")
    print(df.head(10))
