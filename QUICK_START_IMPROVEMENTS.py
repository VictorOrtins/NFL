"""
Quick Start Improvements for NFL Big Data Bowl 2026
Copy these functions directly into your notebook for immediate use.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================================
# PRIORITY 1: MULTI-PLAYER CONTEXT FEATURES
# ============================================================================

def add_multi_player_features(df):
    """
    Add features capturing player interactions.
    
    New features:
    - dist_to_target_receiver: Distance to the targeted receiver
    - nearest_opponent_dist: Distance to nearest defender/receiver
    - nearby_players_count: Number of players within 5 yards
    - avg_opponent_dist: Average distance to all opponents
    """
    print("Adding multi-player context features...")
    
    # Initialize new columns
    df['dist_to_target_receiver'] = np.nan
    df['nearest_opponent_dist'] = np.nan
    df['nearby_players_count'] = 0
    df['avg_opponent_dist'] = np.nan
    
    # Group by play and frame
    grouped = df.groupby(['game_id', 'play_id', 'frame_id'])
    
    for (game_id, play_id, frame_id), group in tqdm(grouped, desc="Processing frames"):
        # Get targeted receiver position
        target_receiver = group[group['player_role'] == 'Targeted Receiver']
        
        if len(target_receiver) == 0:
            continue
            
        target_x = target_receiver.iloc[0]['x']
        target_y = target_receiver.iloc[0]['y']
        
        # Process each player in this frame
        for idx in group.index:
            row = df.loc[idx]
            
            # Distance to targeted receiver
            dist_to_target = np.sqrt((row['x'] - target_x)**2 + (row['y'] - target_y)**2)
            df.at[idx, 'dist_to_target_receiver'] = dist_to_target
            
            # Get opponents
            if row['player_side'] == 'Offense':
                opponents = group[group['player_side'] == 'Defense']
            else:
                opponents = group[group['player_side'] == 'Offense']
            
            if len(opponents) > 0:
                # Calculate distances to all opponents
                opp_dists = np.sqrt(
                    (opponents['x'] - row['x'])**2 + 
                    (opponents['y'] - row['y'])**2
                )
                
                # Nearest opponent
                df.at[idx, 'nearest_opponent_dist'] = opp_dists.min()
                
                # Average opponent distance
                df.at[idx, 'avg_opponent_dist'] = opp_dists.mean()
            else:
                df.at[idx, 'nearest_opponent_dist'] = 999.0
                df.at[idx, 'avg_opponent_dist'] = 999.0
            
            # Count nearby players (within 5 yards)
            all_others = group[group.index != idx]
            if len(all_others) > 0:
                all_dists = np.sqrt(
                    (all_others['x'] - row['x'])**2 + 
                    (all_others['y'] - row['y'])**2
                )
                nearby_count = (all_dists < 5.0).sum()
                df.at[idx, 'nearby_players_count'] = nearby_count
    
    print(f"✓ Added 4 new features")
    return df


# ============================================================================
# PRIORITY 2: COMPLETE PREDICTION PIPELINE
# ============================================================================

def predict_and_submit(model, config, preprocessor):
    """
    Complete prediction pipeline for test data.
    
    Args:
        model: Trained Keras model
        config: Configuration dictionary
        preprocessor: Fitted sklearn ColumnTransformer
    
    Returns:
        submission_df: DataFrame ready for submission
    """
    print("\n" + "="*60)
    print("PREDICTION PIPELINE")
    print("="*60)
    
    # Load test data
    print("\n1. Loading test data...")
    test_input_path = config['BASE_PATH'].replace('/train', '/test_input.csv')
    test_input_df = pd.read_csv(test_input_path)
    print(f"   Loaded {len(test_input_df)} rows")
    
    # Feature engineering
    print("\n2. Feature engineering...")
    from datetime import datetime
    
    # Height conversion
    def convert_height_to_inches(height_str):
        try:
            feet, inches = map(int, str(height_str).split('-'))
            return feet * 12 + inches
        except:
            return np.nan
    
    # Age calculation
    def calculate_age(birth_date_str, current_date=datetime(2024, 1, 1)):
        try:
            birth_date = datetime.strptime(str(birth_date_str), '%Y-%m-%d')
            return (current_date - birth_date).days / 365.25
        except:
            return np.nan
    
    test_input_df['player_height_inches'] = test_input_df['player_height'].apply(convert_height_to_inches)
    test_input_df['player_age'] = test_input_df['player_birth_date'].apply(calculate_age)
    test_input_df['dist_to_land_spot'] = np.sqrt(
        (test_input_df['x'] - test_input_df['ball_land_x'])**2 + 
        (test_input_df['y'] - test_input_df['ball_land_y'])**2
    )
    test_input_df['delta_x_to_land'] = test_input_df['ball_land_x'] - test_input_df['x']
    test_input_df['delta_y_to_land'] = test_input_df['ball_land_y'] - test_input_df['y']
    
    dir_rad = np.deg2rad(test_input_df['dir'])
    test_input_df['vx'] = test_input_df['s'] * np.cos(dir_rad)
    test_input_df['vy'] = test_input_df['s'] * np.sin(dir_rad)
    
    # Add multi-player features if you implemented them
    # test_input_df = add_multi_player_features(test_input_df)
    
    # Preprocessing
    print("\n3. Preprocessing...")
    numeric_features = [
        'x', 'y', 's', 'a', 'player_height_inches', 'player_age',
        'dist_to_land_spot', 'delta_x_to_land', 'delta_y_to_land', 'vx', 'vy'
    ]
    categorical_features = ['play_direction', 'player_position', 'player_side', 'player_role']
    
    processed_test_data = preprocessor.transform(test_input_df[numeric_features + categorical_features])
    feature_names = preprocessor.get_feature_names_out()
    processed_test_df = pd.DataFrame(processed_test_data, columns=feature_names, index=test_input_df.index)
    
    id_cols = ['game_id', 'play_id', 'nfl_id', 'num_frames_output']
    final_test_df = pd.concat([test_input_df[id_cols], processed_test_df], axis=1)
    
    # Create sequences
    print("\n4. Creating sequences...")
    unique_plays = final_test_df[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    encoder_input_data = []
    play_identifiers = []
    
    for _, row in tqdm(unique_plays.iterrows(), total=len(unique_plays), desc="   Creating sequences"):
        game_id, play_id, nfl_id = row['game_id'], row['play_id'], row['nfl_id']
        
        input_seq = final_test_df[
            (final_test_df['game_id'] == game_id) & 
            (final_test_df['play_id'] == play_id) & 
            (final_test_df['nfl_id'] == nfl_id)
        ]
        
        input_features = input_seq[feature_names].values
        
        # Pad input
        padded_input = np.zeros((config['MAX_INPUT_LEN'], len(feature_names)))
        seq_len = min(len(input_features), config['MAX_INPUT_LEN'])
        padded_input[-seq_len:] = input_features[-seq_len:]
        
        encoder_input_data.append(padded_input)
        play_identifiers.append((game_id, play_id, nfl_id, input_seq['num_frames_output'].iloc[0]))
    
    X_enc_test = np.array(encoder_input_data)
    print(f"   Created {len(X_enc_test)} sequences")
    
    # Generate predictions
    print("\n5. Generating predictions...")
    predictions = []
    
    for i in tqdm(range(len(X_enc_test)), desc="   Predicting"):
        input_seq = X_enc_test[i:i+1]
        decoder_input = np.zeros((1, 1, 2))  # Start token
        
        num_frames = int(play_identifiers[i][3])
        
        # Autoregressive prediction
        for _ in range(num_frames):
            pred = model.predict([input_seq, decoder_input], verbose=0)
            next_coord = pred[:, -1:, :]  # Last timestep
            decoder_input = np.concatenate([decoder_input, next_coord], axis=1)
        
        # Remove start token
        output_sequence = decoder_input[0, 1:, :]
        predictions.append(output_sequence)
    
    # Format submission
    print("\n6. Formatting submission...")
    submission_rows = []
    
    for i, play_info in enumerate(play_identifiers):
        game_id, play_id, nfl_id, num_frames = play_info
        predicted_trajectory = predictions[i]
        
        for frame_idx in range(len(predicted_trajectory)):
            frame_id = frame_idx + 1
            row_id = f"{game_id}_{play_id}_{nfl_id}_{frame_id}"
            x_pred, y_pred = predicted_trajectory[frame_idx]
            submission_rows.append({'id': row_id, 'x': x_pred, 'y': y_pred})
    
    submission_df = pd.DataFrame(submission_rows)
    
    # Sanity checks
    print("\n7. Running sanity checks...")
    issues = []
    
    if submission_df['x'].isna().any() or submission_df['y'].isna().any():
        issues.append("Contains NaN values")
    
    if (submission_df['x'] < 0).any() or (submission_df['x'] > 120).any():
        issues.append(f"X out of bounds: {(submission_df['x'] < 0).sum() + (submission_df['x'] > 120).sum()} rows")
    
    if (submission_df['y'] < 0).any() or (submission_df['y'] > 53.3).any():
        issues.append(f"Y out of bounds: {(submission_df['y'] < 0).sum() + (submission_df['y'] > 53.3).sum()} rows")
    
    if issues:
        print("   ⚠️ Issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✓ All checks passed")
    
    # Save
    submission_df.to_csv('submission.csv', index=False)
    print(f"\n✓ Submission saved: {len(submission_df)} predictions")
    print(f"  File: submission.csv")
    
    return submission_df


# ============================================================================
# PRIORITY 3: COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_model_comprehensive(model, X_enc_val, y_dec_val, dec_input_val):
    """
    Comprehensive evaluation with multiple metrics.
    
    Args:
        model: Trained model
        X_enc_val: Validation encoder input
        y_dec_val: Validation ground truth
        dec_input_val: Validation decoder input
    
    Returns:
        results: Dictionary of metrics
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Get predictions
    print("\n1. Generating predictions...")
    y_pred = model.predict([X_enc_val, dec_input_val], verbose=1)
    
    # Overall metrics
    print("\n2. Calculating metrics...")
    results = {}
    
    # RMSE
    mse = np.mean((y_dec_val - y_pred)**2)
    results['overall_rmse'] = np.sqrt(mse)
    results['overall_mae'] = np.mean(np.abs(y_dec_val - y_pred))
    
    # Per-coordinate
    results['x_rmse'] = np.sqrt(np.mean((y_dec_val[:, :, 0] - y_pred[:, :, 0])**2))
    results['y_rmse'] = np.sqrt(np.mean((y_dec_val[:, :, 1] - y_pred[:, :, 1])**2))
    results['x_mae'] = np.mean(np.abs(y_dec_val[:, :, 0] - y_pred[:, :, 0]))
    results['y_mae'] = np.mean(np.abs(y_dec_val[:, :, 1] - y_pred[:, :, 1]))
    
    # Euclidean distance
    distances = np.sqrt(np.sum((y_dec_val - y_pred)**2, axis=2))
    results['mean_distance_error'] = np.mean(distances)
    results['median_distance_error'] = np.median(distances)
    results['90th_percentile_error'] = np.percentile(distances, 90)
    results['95th_percentile_error'] = np.percentile(distances, 95)
    results['max_error'] = np.max(distances)
    
    # Temporal analysis (error by frame)
    max_frames = y_dec_val.shape[1]
    frame_errors = []
    for frame_idx in range(max_frames):
        valid_mask = ~np.isnan(y_dec_val[:, frame_idx, 0])
        if valid_mask.sum() > 0:
            frame_error = np.sqrt(np.mean(
                (y_dec_val[valid_mask, frame_idx, :] - y_pred[valid_mask, frame_idx, :])**2
            ))
            frame_errors.append(frame_error)
        else:
            frame_errors.append(np.nan)
    
    results['frame_errors'] = frame_errors
    
    # Print results
    print("\n3. Results:")
    print(f"\n   Overall Metrics:")
    print(f"   - RMSE:              {results['overall_rmse']:.4f} yards")
    print(f"   - MAE:               {results['overall_mae']:.4f} yards")
    print(f"   - Mean Distance Err: {results['mean_distance_error']:.4f} yards")
    print(f"   - Median Distance:   {results['median_distance_error']:.4f} yards")
    print(f"   - 90th Percentile:   {results['90th_percentile_error']:.4f} yards")
    print(f"   - Max Error:         {results['max_error']:.4f} yards")
    
    print(f"\n   Per-Coordinate:")
    print(f"   - X RMSE: {results['x_rmse']:.4f} yards")
    print(f"   - Y RMSE: {results['y_rmse']:.4f} yards")
    
    # Visualization
    print("\n4. Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall metrics
    ax = axes[0, 0]
    metrics = ['overall_rmse', 'overall_mae', 'mean_distance_error', 'median_distance_error']
    values = [results[m] for m in metrics]
    bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('Overall Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Yards', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Per-coordinate comparison
    ax = axes[0, 1]
    coord_metrics = ['x_rmse', 'y_rmse', 'x_mae', 'y_mae']
    coord_values = [results[m] for m in coord_metrics]
    bars = ax.bar(coord_metrics, coord_values, color=['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
    ax.set_title('Per-Coordinate Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Yards', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Temporal error
    ax = axes[1, 0]
    valid_frames = [i for i, e in enumerate(frame_errors) if not np.isnan(e)]
    valid_errors = [e for e in frame_errors if not np.isnan(e)]
    ax.plot(valid_frames, valid_errors, marker='o', linewidth=2, markersize=4)
    ax.set_title('Error by Frame (Temporal Analysis)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('RMSE (yards)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[1, 1]
    ax.hist(distances.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(results['mean_distance_error'], color='r', linestyle='--', 
               linewidth=2, label=f'Mean: {results["mean_distance_error"]:.2f}')
    ax.axvline(results['median_distance_error'], color='g', linestyle='--', 
               linewidth=2, label=f'Median: {results["median_distance_error"]:.2f}')
    ax.set_title('Distance Error Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance Error (yards)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_evaluation.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: comprehensive_evaluation.png")
    
    return results


# ============================================================================
# PRIORITY 4: DATA AUGMENTATION
# ============================================================================

def augment_training_data(input_df, output_df):
    """
    Augment training data by flipping plays horizontally.
    Doubles the training data size.
    
    Args:
        input_df: Input training data
        output_df: Output training data
    
    Returns:
        augmented_input_df, augmented_output_df
    """
    print("\nAugmenting training data...")
    
    # Create copies
    aug_input = input_df.copy()
    aug_output = output_df.copy()
    
    # Flip y-coordinates (field width is 53.3 yards)
    aug_input['y'] = 53.3 - aug_input['y']
    aug_input['ball_land_y'] = 53.3 - aug_input['ball_land_y']
    
    # Flip direction angles
    aug_input['dir'] = (360 - aug_input['dir']) % 360
    aug_input['o'] = (360 - aug_input['o']) % 360
    
    # Flip velocity components
    if 'vy' in aug_input.columns:
        aug_input['vy'] = -aug_input['vy']
    
    if 'delta_y_to_land' in aug_input.columns:
        aug_input['delta_y_to_land'] = -aug_input['delta_y_to_land']
    
    # Flip output
    aug_output['y'] = 53.3 - aug_output['y']
    
    # Modify IDs to avoid conflicts (add large offset)
    aug_input['play_id'] = aug_input['play_id'] + 100000
    aug_output['play_id'] = aug_output['play_id'] + 100000
    
    # Concatenate
    combined_input = pd.concat([input_df, aug_input], ignore_index=True)
    combined_output = pd.concat([output_df, aug_output], ignore_index=True)
    
    print(f"✓ Original size: {len(input_df)} → Augmented size: {len(combined_input)}")
    
    return combined_input, combined_output


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("""
    USAGE EXAMPLES:
    
    1. Add multi-player features:
       input_df = add_multi_player_features(input_df)
    
    2. Augment training data:
       input_df, output_df = augment_training_data(input_df, output_df)
    
    3. Evaluate model:
       results = evaluate_model_comprehensive(model, X_enc_val, y_dec_val, dec_input_val)
    
    4. Generate submission:
       submission = predict_and_submit(model, config, preprocessor)
    """)
