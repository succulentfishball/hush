"""
Comprehensive GNN Model Evaluation
Evaluates the trained GNN model on MTA ridership data.
Computes regression metrics: MSE, RMSE, MAE, R², MAPE
Creates visualizations and analysis.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import os


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class GNN(nn.Module):
    """GNN model architecture (must match training)."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mlp = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        return self.mlp(h).squeeze()


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_data(csv_path, edges_path, stats_path, cmplx_to_node_path):
    """
    Load and prepare ridership data for evaluation.
    
    Returns:
        snapshots: List of temporal graph snapshots
        edge_tensor: Edge connectivity
        num_nodes: Number of nodes in graph
        stats_df: Normalization statistics
        cmplx_to_node: Complex ID to node ID mapping
    """
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Load ridership data (already cleaned and aggregated)
    print(f"\n1. Loading ridership data from {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["transit_timestamp"])
    print(f"   ✓ Loaded {len(df):,} rows")
    print(f"   ✓ Date range: {df['transit_timestamp'].min()} to {df['transit_timestamp'].max()}")
    print(f"   ✓ Unique stations: {df['station_complex_id'].nunique()}")
    print(f"   ✓ Unique timestamps: {df['transit_timestamp'].nunique()}")
    
    # Load normalization stats
    print(f"\n2. Loading normalization stats from {stats_path}")
    stats_df = pd.read_csv(stats_path)
    print(f"   ✓ Loaded stats for {len(stats_df)} stations")
    
    # Load complex ID -> node ID mapping
    print(f"\n3. Loading node mapping from {cmplx_to_node_path}")
    mapping_df = pd.read_csv(cmplx_to_node_path)
    cmplx_to_node = dict(zip(mapping_df['complex_id'], mapping_df['node_id']))
    node_to_cmplx = dict(zip(mapping_df['node_id'], mapping_df['complex_id']))
    num_nodes = len(cmplx_to_node)
    print(f"   ✓ Loaded mapping for {num_nodes} nodes")
    
    # Map station complex IDs to node IDs
    df['node_id'] = df['station_complex_id'].map(cmplx_to_node)
    
    # Filter out stations not in the mapping
    before = len(df)
    df = df[df['node_id'].notna()].copy()
    df['node_id'] = df['node_id'].astype(int)
    after = len(df)
    if before != after:
        print(f"   ⚠ Filtered {before - after} rows with unmapped stations")
    
    # Merge with normalization stats
    df = df.merge(stats_df, left_on='station_complex_id', right_on='complex_id', how='left')
    
    # Normalize ridership
    print("\n4. Normalizing ridership values...")
    df['ridership_norm'] = (df['ridership'] - df['mean']) / (df['std'] + 1e-6)
    
    # Add temporal features (sin/cos encoding of hour)
    print("5. Adding temporal features (sin/cos hour encoding)...")
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Load edges
    print(f"\n6. Loading edges from {edges_path}")
    edges = pd.read_csv(edges_path)
    
    edge_list = []
    for _, row in edges.iterrows():
        start_cmplx = row['from_complex_id']
        end_cmplx = row['to_complex_id']
        if start_cmplx in cmplx_to_node and end_cmplx in cmplx_to_node:
            start_node = cmplx_to_node[start_cmplx]
            end_node = cmplx_to_node[end_cmplx]
            edge_list.append([start_node, end_node])
            edge_list.append([end_node, start_node])  # Undirected
    
    edge_tensor = torch.tensor(edge_list, dtype=torch.long).T
    print(f"   ✓ Built edge tensor with {edge_tensor.shape[1]:,} edges")
    
    # Build temporal snapshots
    print("\n7. Building temporal graph snapshots...")
    snapshots = []
    
    timestamps = sorted(df['transit_timestamp'].unique())
    print(f"   ✓ Found {len(timestamps)} unique timestamps")
    print(f"   ✓ Building {len(timestamps) - 1} temporal snapshots (t -> t+1)...")
    
    for i in range(len(timestamps) - 1):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        
        # Current state (t0) - features
        df_t0 = df[df['transit_timestamp'] == t0].copy()
        # Next state (t1) - targets
        df_t1 = df[df['transit_timestamp'] == t1].copy()
        
        # Build feature matrix (num_nodes x 3: ridership_norm, sin_hour, cos_hour)
        X = torch.zeros(num_nodes, 3, dtype=torch.float32)
        y_actual = torch.zeros(num_nodes, dtype=torch.float32)
        y_norm_actual = torch.zeros(num_nodes, dtype=torch.float32)
        
        # Fill in features for stations with data at t0
        for _, row in df_t0.iterrows():
            node_id = int(row['node_id'])
            X[node_id, 0] = row['ridership_norm']
            X[node_id, 1] = row['sin_hour']
            X[node_id, 2] = row['cos_hour']
        
        # Fill in targets for stations with data at t1
        for _, row in df_t1.iterrows():
            node_id = int(row['node_id'])
            y_actual[node_id] = row['ridership']  # Raw ridership
            y_norm_actual[node_id] = row['ridership_norm']  # Normalized ridership
        
        # Store station metadata for denormalization
        station_stats = {}
        for _, row in stats_df.iterrows():
            complex_id = row['complex_id']
            if complex_id in cmplx_to_node:
                node_id = cmplx_to_node[complex_id]
                station_stats[node_id] = {
                    'mean': row['mean'],
                    'std': row['std']
                }
        
        snapshots.append({
            'features': X,
            'targets_raw': y_actual,
            'targets_norm': y_norm_actual,
            'timestamp_t0': t0,
            'timestamp_t1': t1,
            'station_stats': station_stats
        })
    
    print(f"   ✓ Created {len(snapshots)} temporal snapshots")
    
    return snapshots, edge_tensor, num_nodes, stats_df, cmplx_to_node, node_to_cmplx


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, snapshots, edge_tensor):
    """
    Evaluate model on snapshots and compute metrics.
    
    Returns:
        results: Dict with predictions, actuals, and per-snapshot metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    
    model.eval()
    
    all_preds_raw = []
    all_actuals_raw = []
    all_preds_norm = []
    all_actuals_norm = []
    per_snapshot_metrics = []
    
    with torch.no_grad():
        for i, snap in enumerate(snapshots):
            X = snap['features']
            y_actual_raw = snap['targets_raw'].numpy()
            y_actual_norm = snap['targets_norm'].numpy()
            station_stats = snap['station_stats']
            
            # Get predictions (normalized)
            y_pred_norm = model(X, edge_tensor).numpy()
            
            # Denormalize predictions to get raw ridership
            y_pred_raw = np.zeros_like(y_pred_norm)
            for node_id in range(len(y_pred_norm)):
                if node_id in station_stats:
                    mean = station_stats[node_id]['mean']
                    std = station_stats[node_id]['std']
                    y_pred_raw[node_id] = y_pred_norm[node_id] * std + mean
                else:
                    y_pred_raw[node_id] = 0
            
            # Only evaluate on stations with actual ridership data
            mask = y_actual_raw > 0
            
            if mask.sum() > 0:
                # Store predictions and actuals
                all_preds_raw.extend(y_pred_raw[mask])
                all_actuals_raw.extend(y_actual_raw[mask])
                all_preds_norm.extend(y_pred_norm[mask])
                all_actuals_norm.extend(y_actual_norm[mask])
                
                # Compute per-snapshot metrics
                snap_mse = mean_squared_error(y_actual_raw[mask], y_pred_raw[mask])
                snap_mae = mean_absolute_error(y_actual_raw[mask], y_pred_raw[mask])
                
                per_snapshot_metrics.append({
                    'snapshot': i,
                    'timestamp': snap['timestamp_t1'],
                    'num_stations': mask.sum(),
                    'mse': snap_mse,
                    'mae': snap_mae
                })
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(snapshots)} snapshots...")
    
    print(f"   ✓ Evaluated all {len(snapshots)} snapshots")
    
    # Convert to arrays
    all_preds_raw = np.array(all_preds_raw)
    all_actuals_raw = np.array(all_actuals_raw)
    all_preds_norm = np.array(all_preds_norm)
    all_actuals_norm = np.array(all_actuals_norm)
    
    # Compute overall metrics on raw ridership
    print("\n" + "=" * 80)
    print("COMPUTING METRICS (Raw Ridership)")
    print("=" * 80)
    
    mse = mean_squared_error(all_actuals_raw, all_preds_raw)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_actuals_raw, all_preds_raw)
    r2 = r2_score(all_actuals_raw, all_preds_raw)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((all_actuals_raw - all_preds_raw) / (all_actuals_raw + 1e-6))) * 100
    
    # Additional metrics
    median_ae = np.median(np.abs(all_actuals_raw - all_preds_raw))
    max_error = np.max(np.abs(all_actuals_raw - all_preds_raw))
    
    # Percentage within tolerance
    tolerance_10 = np.mean(np.abs(all_actuals_raw - all_preds_raw) <= 10) * 100
    tolerance_20 = np.mean(np.abs(all_actuals_raw - all_preds_raw) <= 20) * 100
    tolerance_50 = np.mean(np.abs(all_actuals_raw - all_preds_raw) <= 50) * 100
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'median_ae': median_ae,
        'max_error': max_error,
        'tolerance_10': tolerance_10,
        'tolerance_20': tolerance_20,
        'tolerance_50': tolerance_50,
        'num_predictions': len(all_preds_raw)
    }
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Number of predictions: {metrics['num_predictions']:,}")
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   Median AE: {median_ae:.4f}")
    print(f"   Max Error: {max_error:.2f}")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"\n📈 Prediction Accuracy:")
    print(f"   Within ±10 riders: {tolerance_10:.2f}%")
    print(f"   Within ±20 riders: {tolerance_20:.2f}%")
    print(f"   Within ±50 riders: {tolerance_50:.2f}%")
    
    results = {
        'predictions_raw': all_preds_raw,
        'actuals_raw': all_actuals_raw,
        'predictions_norm': all_preds_norm,
        'actuals_norm': all_actuals_norm,
        'metrics': metrics,
        'per_snapshot_metrics': per_snapshot_metrics
    }
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results, output_dir='evaluation_results'):
    """Create comprehensive visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"CREATING VISUALIZATIONS")
    print("=" * 80)
    print(f"\nSaving plots to: {output_dir}/")
    
    predictions = results['predictions_raw']
    actuals = results['actuals_raw']
    residuals = actuals - predictions
    
    # 1. Scatter plot: Predictions vs Actuals
    plt.figure(figsize=(10, 8))
    plt.scatter(actuals, predictions, alpha=0.3, s=20, edgecolors='none')
    
    # Perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    plt.xlabel('Actual Ridership (riders/hour)', fontsize=12)
    plt.ylabel('Predicted Ridership (riders/hour)', fontsize=12)
    plt.title('GNN Predictions vs Actual Ridership', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predictions_vs_actuals.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ predictions_vs_actuals.png")
    plt.close()
    
    # 2. Residual plot
    plt.figure(figsize=(12, 6))
    plt.scatter(predictions, residuals, alpha=0.3, s=20, edgecolors='none')
    plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero error')
    plt.xlabel('Predicted Ridership (riders/hour)', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residuals.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ residuals.png")
    plt.close()
    
    # 3. Error distribution
    plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
    plt.axvline(x=np.median(residuals), color='orange', linestyle='--', lw=2, 
                label=f'Median error: {np.median(residuals):.2f}')
    plt.xlabel('Prediction Error (riders)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ error_distribution.png")
    plt.close()
    
    # 4. Error by actual ridership (binned)
    plt.figure(figsize=(12, 6))
    bins = np.percentile(actuals, [0, 25, 50, 75, 90, 100])
    bin_indices = np.digitize(actuals, bins)
    
    error_by_bin = []
    bin_labels = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            error_by_bin.append(np.abs(residuals[mask]))
            bin_labels.append(f'{bins[i-1]:.0f}-{bins[i]:.0f}')
    
    plt.boxplot(error_by_bin, labels=bin_labels)
    plt.xlabel('Actual Ridership Range (riders/hour)', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Prediction Error vs Ridership Level', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_by_ridership_level.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ error_by_ridership_level.png")
    plt.close()
    
    # 5. Cumulative error distribution
    plt.figure(figsize=(10, 6))
    abs_errors = np.abs(residuals)
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
    plt.plot(sorted_errors, cumulative, linewidth=2)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50th percentile')
    plt.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    plt.xlabel('Absolute Error (riders)', fontsize=12)
    plt.ylabel('Cumulative Percentage (%)', fontsize=12)
    plt.title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cumulative_error.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ cumulative_error.png")
    plt.close()
    
    # 6. Per-snapshot MAE over time
    if results['per_snapshot_metrics']:
        snap_df = pd.DataFrame(results['per_snapshot_metrics'])
        
        plt.figure(figsize=(14, 6))
        plt.plot(snap_df['snapshot'], snap_df['mae'], linewidth=1.5, alpha=0.7)
        plt.xlabel('Snapshot Index (time)', fontsize=12)
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
        plt.title('Model Performance Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mae_over_time.png', dpi=150, bbox_inches='tight')
        print(f"   ✓ mae_over_time.png")
        plt.close()


def save_metrics_report(results, output_dir='evaluation_results'):
    """Save detailed metrics report to text file."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f'{output_dir}/evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GNN MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Predictions: {results['metrics']['num_predictions']:,}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("REGRESSION METRICS (Raw Ridership)\n")
        f.write("=" * 80 + "\n\n")
        
        metrics = results['metrics']
        f.write(f"Mean Squared Error (MSE):           {metrics['mse']:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE):     {metrics['rmse']:.4f}\n")
        f.write(f"Mean Absolute Error (MAE):          {metrics['mae']:.4f}\n")
        f.write(f"Median Absolute Error:              {metrics['median_ae']:.4f}\n")
        f.write(f"Maximum Error:                      {metrics['max_error']:.2f}\n")
        f.write(f"R² Score:                           {metrics['r2']:.4f}\n")
        f.write(f"Mean Absolute Percentage Error:     {metrics['mape']:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PREDICTION ACCURACY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Predictions within ±10 riders:      {metrics['tolerance_10']:.2f}%\n")
        f.write(f"Predictions within ±20 riders:      {metrics['tolerance_20']:.2f}%\n")
        f.write(f"Predictions within ±50 riders:      {metrics['tolerance_50']:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        actuals = results['actuals_raw']
        predictions = results['predictions_raw']
        
        f.write("Actual Ridership:\n")
        f.write(f"  Mean:     {np.mean(actuals):.2f}\n")
        f.write(f"  Median:   {np.median(actuals):.2f}\n")
        f.write(f"  Std Dev:  {np.std(actuals):.2f}\n")
        f.write(f"  Min:      {np.min(actuals):.2f}\n")
        f.write(f"  Max:      {np.max(actuals):.2f}\n\n")
        
        f.write("Predicted Ridership:\n")
        f.write(f"  Mean:     {np.mean(predictions):.2f}\n")
        f.write(f"  Median:   {np.median(predictions):.2f}\n")
        f.write(f"  Std Dev:  {np.std(predictions):.2f}\n")
        f.write(f"  Min:      {np.min(predictions):.2f}\n")
        f.write(f"  Max:      {np.max(predictions):.2f}\n\n")
    
    print(f"\n   ✓ evaluation_report.txt")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("GNN MODEL EVALUATION")
    print("=" * 80)
    print(f"\nEvaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # File paths
    data_csv = "mta_ridership_clean_full.csv"
    edges_csv = "complex_edges.csv"
    stats_csv = "stats.csv"
    cmplx_to_node_csv = "cmplx_to_node.csv"
    model_path = "model.pt"
    output_dir = "evaluation_results"
    
    # Check if files exist
    required_files = [data_csv, edges_csv, stats_csv, cmplx_to_node_csv, model_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Error: Missing required files: {missing_files}")
        return
    
    # Load data
    snapshots, edge_tensor, num_nodes, stats_df, cmplx_to_node, node_to_cmplx = load_data(
        data_csv, edges_csv, stats_csv, cmplx_to_node_csv
    )
    
    # Load model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    print(f"\nLoading model from {model_path}...")
    
    model = GNN(in_dim=3, hidden_dim=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Architecture: 3 input features → 64 hidden → 1 output")
    print(f"   ✓ Input features: [ridership_norm, sin_hour, cos_hour]")
    print(f"   ✓ Graph: {num_nodes} nodes, {edge_tensor.shape[1]} edges")
    
    # Evaluate
    results = evaluate_model(model, snapshots, edge_tensor)
    
    # Create visualizations
    plot_results(results, output_dir)
    
    # Save metrics report
    save_metrics_report(results, output_dir)
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - evaluation_report.txt")
    print(f"  - predictions_vs_actuals.png")
    print(f"  - residuals.png")
    print(f"  - error_distribution.png")
    print(f"  - error_by_ridership_level.png")
    print(f"  - cumulative_error.png")
    print(f"  - mae_over_time.png")
    print(f"\n📊 Quick Summary:")
    print(f"  - Total predictions: {results['metrics']['num_predictions']:,}")
    print(f"  - R² Score: {results['metrics']['r2']:.4f}")
    print(f"  - RMSE: {results['metrics']['rmse']:.2f} riders")
    print(f"  - MAE: {results['metrics']['mae']:.2f} riders")
    print(f"  - Predictions within ±20 riders: {results['metrics']['tolerance_20']:.1f}%")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
