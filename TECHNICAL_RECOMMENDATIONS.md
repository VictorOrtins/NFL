# Technical Recommendations & Action Plan
## NFL Big Data Bowl 2026 - Player Trajectory Prediction

---

## Executive Summary

**Current Status**: You have a working Transformer encoder-decoder model with:
- ✅ **Validation RMSE**: ~3.54 yards (best epoch 18)
- ✅ **Training RMSE**: ~3.46 yards  
- ✅ **Model size**: 1.86M parameters (d_model=64)
- ✅ **Good feature engineering**: Distance to landing spot, velocity components, etc.
- ✅ **Proper data pipeline**: Caching, preprocessing, train/val split

**Key Issues Identified**:
1. ⚠️ **Single-player modeling**: No inter-player interactions
2. ⚠️ **Teacher forcing**: Inference differs from training
3. ⚠️ **Limited context**: Only individual player features
4. ⚠️ **Incomplete prediction function**: Test inference not fully implemented

---

## Priority 1: Fix Critical Issues (Quick Wins)

### 1.1 Complete the Prediction Pipeline ⚡

**Problem**: The `predict_with_transformer()` function is incomplete (line 1051-1081 in notebook).

**Solution**: Implement proper autoregressive inference.

```python
def predict_with_transformer_complete(config, model, preprocessor):
    """Complete prediction pipeline for Transformer model."""
    
    # Load test data
    test_input_df = pd.read_csv(os.path.join(config['BASE_PATH'], '../test_input.csv'))
    test_ids_df = pd.read_csv(os.path.join(config['BASE_PATH'], '../test.csv'))
    
    # Feature engineering
    test_input_df = feature_engineering(test_input_df)
    
    # Preprocessing
    numeric_features = ['x', 'y', 's', 'a', 'player_height_inches', 'player_age', 
                       'dist_to_land_spot', 'delta_x_to_land', 'delta_y_to_land', 'vx', 'vy']
    categorical_features = ['play_direction', 'player_position', 'player_side', 'player_role']
    
    processed_test_data = preprocessor.transform(test_input_df[numeric_features + categorical_features])
    feature_names = preprocessor.get_feature_names_out()
    processed_test_df = pd.DataFrame(processed_test_data, columns=feature_names, index=test_input_df.index)
    
    id_cols = ['game_id', 'play_id', 'nfl_id', 'num_frames_output']
    final_test_df = pd.concat([test_input_df[id_cols], processed_test_df], axis=1)
    
    # Create sequences
    X_enc_test, play_identifiers = create_sequences(
        final_test_df, None, feature_names, 
        config['MAX_INPUT_LEN'], config['MAX_OUTPUT_LEN'], is_test=True
    )
    
    print("Generating predictions...")
    predictions = []
    
    for i in tqdm(range(len(X_enc_test))):
        input_seq = X_enc_test[i:i+1]
        decoder_input = np.zeros((1, 1, 2))  # Start token
        
        num_frames = play_identifiers[i][3]
        
        for _ in range(num_frames):
            pred = model.predict([input_seq, decoder_input], verbose=0)
            next_coord = pred[:, -1:, :]  # Last timestep
            decoder_input = np.concatenate([decoder_input, next_coord], axis=1)
        
        # Remove start token
        output_sequence = decoder_input[0, 1:, :]
        predictions.append(output_sequence)
    
    # Format submission
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
    submission_df.to_csv('submission.csv', index=False)
    print(f"Submission saved: {len(submission_df)} predictions")
    
    return submission_df
```

**Action**: Add this to your notebook and test it.

---

### 1.2 Add Multi-Player Context Features 🎯

**Problem**: Each player is predicted independently, ignoring interactions.

**Solution**: Engineer features that capture player relationships.

```python
def add_multi_player_features(df):
    """Add features capturing player interactions."""
    
    # Group by play
    play_groups = df.groupby(['game_id', 'play_id', 'frame_id'])
    
    new_features = []
    
    for (game_id, play_id, frame_id), group in play_groups:
        # Get targeted receiver position
        target_receiver = group[group['player_role'] == 'Targeted Receiver']
        
        if len(target_receiver) > 0:
            target_x, target_y = target_receiver.iloc[0][['x', 'y']]
            
            # For each player in this frame
            for idx, row in group.iterrows():
                # Distance to targeted receiver
                dist_to_target = np.sqrt((row['x'] - target_x)**2 + (row['y'] - target_y)**2)
                
                # Get nearest defender/receiver
                if row['player_side'] == 'Offense':
                    defenders = group[group['player_side'] == 'Defense']
                    if len(defenders) > 0:
                        dists = np.sqrt((defenders['x'] - row['x'])**2 + (defenders['y'] - row['y'])**2)
                        nearest_defender_dist = dists.min()
                    else:
                        nearest_defender_dist = 999
                else:
                    receivers = group[group['player_side'] == 'Offense']
                    if len(receivers) > 0:
                        dists = np.sqrt((receivers['x'] - row['x'])**2 + (receivers['y'] - row['y'])**2)
                        nearest_receiver_dist = dists.min()
                    else:
                        nearest_receiver_dist = 999
                
                # Count nearby players (within 5 yards)
                all_others = group[group.index != idx]
                dists_all = np.sqrt((all_others['x'] - row['x'])**2 + (all_others['y'] - row['y'])**2)
                nearby_count = (dists_all < 5).sum()
                
                new_features.append({
                    'index': idx,
                    'dist_to_target_receiver': dist_to_target,
                    'nearest_defender_dist': nearest_defender_dist if row['player_side'] == 'Offense' else nearest_receiver_dist,
                    'nearby_players_count': nearby_count
                })
    
    # Merge back to original dataframe
    features_df = pd.DataFrame(new_features).set_index('index')
    df = df.join(features_df)
    
    return df
```

**Action**: 
1. Add this function to your feature engineering pipeline
2. Update `numeric_features` list to include new features
3. Retrain model

**Expected Impact**: 5-10% RMSE improvement

---

### 1.3 Implement Proper Evaluation Metrics 📊

**Problem**: Only tracking overall RMSE, no detailed analysis.

**Solution**: Create comprehensive evaluation.

```python
def evaluate_predictions(y_true, y_pred, play_identifiers, input_df):
    """Comprehensive evaluation of predictions."""
    
    results = {
        'overall_rmse': np.sqrt(np.mean((y_true - y_pred)**2)),
        'overall_mae': np.mean(np.abs(y_true - y_pred)),
        'x_rmse': np.sqrt(np.mean((y_true[:, :, 0] - y_pred[:, :, 0])**2)),
        'y_rmse': np.sqrt(np.mean((y_true[:, :, 1] - y_pred[:, :, 1])**2)),
    }
    
    # Per-role analysis
    roles = ['Targeted Receiver', 'Defensive Coverage', 'Other Route Runner']
    for role in roles:
        role_mask = [info[4] == role for info in play_identifiers]  # Assuming role is stored
        if sum(role_mask) > 0:
            role_rmse = np.sqrt(np.mean((y_true[role_mask] - y_pred[role_mask])**2))
            results[f'{role}_rmse'] = role_rmse
    
    # Temporal analysis (error by frame)
    max_frames = y_true.shape[1]
    frame_errors = []
    for frame_idx in range(max_frames):
        frame_error = np.sqrt(np.mean((y_true[:, frame_idx, :] - y_pred[:, frame_idx, :])**2))
        frame_errors.append(frame_error)
    
    results['frame_errors'] = frame_errors
    
    # Distance-based analysis
    distances = np.sqrt(np.sum((y_true - y_pred)**2, axis=2))  # Euclidean distance per frame
    results['mean_distance_error'] = np.mean(distances)
    results['median_distance_error'] = np.median(distances)
    results['90th_percentile_error'] = np.percentile(distances, 90)
    
    return results

def plot_evaluation_results(results):
    """Visualize evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall metrics
    ax = axes[0, 0]
    metrics = ['overall_rmse', 'overall_mae', 'x_rmse', 'y_rmse']
    values = [results[m] for m in metrics]
    ax.bar(metrics, values)
    ax.set_title('Overall Metrics')
    ax.set_ylabel('Yards')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Per-role RMSE
    ax = axes[0, 1]
    role_metrics = [k for k in results.keys() if 'Receiver' in k or 'Coverage' in k or 'Runner' in k]
    if role_metrics:
        role_values = [results[m] for m in role_metrics]
        ax.bar(role_metrics, role_values, color='orange')
        ax.set_title('RMSE by Player Role')
        ax.set_ylabel('RMSE (yards)')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Temporal error
    ax = axes[1, 0]
    frame_errors = results['frame_errors']
    ax.plot(range(1, len(frame_errors)+1), frame_errors, marker='o')
    ax.set_title('Error by Frame (Temporal Analysis)')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('RMSE (yards)')
    ax.grid(True)
    
    # Error distribution
    ax = axes[1, 1]
    error_stats = ['mean_distance_error', 'median_distance_error', '90th_percentile_error']
    error_values = [results[m] for m in error_stats]
    ax.bar(error_stats, error_values, color='green')
    ax.set_title('Distance Error Distribution')
    ax.set_ylabel('Yards')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('detailed_evaluation.png', dpi=150)
    print("Evaluation plot saved as 'detailed_evaluation.png'")
```

**Action**: Add to notebook and run after training.

---

## Priority 2: Model Architecture Improvements

### 2.1 Increase Model Capacity 🚀

**Current**: d_model=64, 1.86M parameters  
**Observation**: Training loss still decreasing at epoch 28 → model can learn more

**Recommendation**:
```python
config_transformer_large = {
    'BASE_PATH': './nfl-big-data-bowl-2026-prediction/train',
    'TRAIN_WEEKS': range(1, 16),
    'VALIDATION_WEEKS': range(16, 19),
    'MAX_INPUT_LEN': 30,
    'MAX_OUTPUT_LEN': 40,
    'BATCH_SIZE': 8,
    'EPOCHS': 50,
    'LEARNING_RATE': 0.0002,
    'D_MODEL': 128,        # Increased from 64
    'NUM_HEADS': 8,
    'DFF': 512,            # Increased from 256
    'MODEL_NAME': 'transformer_model_D128'
}
```

**Expected Impact**: 3-5% RMSE improvement  
**Trade-off**: Longer training time (~2x)

---

### 2.2 Add Scheduled Sampling 🎲

**Problem**: Teacher forcing creates train/test mismatch.

**Solution**: Gradually use predicted values during training.

```python
class ScheduledSamplingCallback(tf.keras.callbacks.Callback):
    """Gradually transition from teacher forcing to autoregressive."""
    
    def __init__(self, initial_prob=1.0, final_prob=0.0, decay_epochs=20):
        super().__init__()
        self.initial_prob = initial_prob
        self.final_prob = final_prob
        self.decay_epochs = decay_epochs
        self.teacher_forcing_prob = initial_prob
    
    def on_epoch_end(self, epoch, logs=None):
        # Linear decay
        if epoch < self.decay_epochs:
            self.teacher_forcing_prob = self.initial_prob - (
                (self.initial_prob - self.final_prob) * epoch / self.decay_epochs
            )
        else:
            self.teacher_forcing_prob = self.final_prob
        
        print(f"\nTeacher forcing probability: {self.teacher_forcing_prob:.3f}")
```

**Action**: Implement custom training loop with scheduled sampling (advanced).

---

### 2.3 Try Alternative Architectures 🏗️

**Option A: Simpler LSTM Baseline**
- Faster to train
- Good for comparison
- You already have `train_gru.ipynb` - analyze those results!

**Option B: Graph Neural Network (GNN)**
- Explicitly models player interactions
- Better for multi-agent scenarios
- More complex to implement

**Recommendation**: 
1. First, improve current Transformer with multi-player features
2. Then try GNN if time permits

---

## Priority 3: Data & Training Improvements

### 3.1 Data Augmentation 📈

```python
def augment_play_data(input_df, output_df):
    """Augment training data by flipping plays horizontally."""
    
    augmented_input = input_df.copy()
    augmented_output = output_df.copy()
    
    # Flip y-coordinates (field width is 53.3 yards)
    augmented_input['y'] = 53.3 - augmented_input['y']
    augmented_input['ball_land_y'] = 53.3 - augmented_input['ball_land_y']
    
    # Flip direction angles
    augmented_input['dir'] = (360 - augmented_input['dir']) % 360
    augmented_input['o'] = (360 - augmented_input['o']) % 360
    
    # Flip output
    augmented_output['y'] = 53.3 - augmented_output['y']
    
    # Concatenate with original
    combined_input = pd.concat([input_df, augmented_input], ignore_index=True)
    combined_output = pd.concat([output_df, augmented_output], ignore_index=True)
    
    return combined_input, combined_output
```

**Expected Impact**: 2-3% RMSE improvement

---

### 3.2 Better Hyperparameter Tuning 🎛️

**Current**: Fixed hyperparameters  
**Recommendation**: Try these variations

| Parameter | Current | Try |
|-----------|---------|-----|
| Learning Rate | 0.0002 | 0.0001, 0.0003 |
| Batch Size | 8 | 16, 32 |
| DFF | 256 | 512, 1024 |
| Dropout | 0.1 | 0.1, 0.2, 0.3 |
| Num Layers | 4/4 | 6/6, 8/8 |

**Action**: Use grid search or Optuna for automated tuning.

---

### 3.3 Ensemble Methods 🎭

**Strategy**: Combine multiple models for better predictions.

```python
def ensemble_predictions(models, X_enc_test, play_identifiers, weights=None):
    """Ensemble multiple model predictions."""
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    all_predictions = []
    
    for model in models:
        model_preds = []
        for i in range(len(X_enc_test)):
            # ... [prediction logic] ...
            model_preds.append(pred)
        all_predictions.append(model_preds)
    
    # Weighted average
    ensemble_preds = []
    for i in range(len(X_enc_test)):
        weighted_pred = sum(w * pred[i] for w, pred in zip(weights, all_predictions))
        ensemble_preds.append(weighted_pred)
    
    return ensemble_preds
```

**Models to Ensemble**:
1. Transformer (d_model=64)
2. Transformer (d_model=128)
3. GRU/LSTM baseline
4. Simple physics-based model

**Expected Impact**: 5-8% RMSE improvement

---

## Priority 4: Physics-Based Constraints

### 4.1 Add Physical Realism 🏃

**Problem**: Model can predict impossible movements.

**Solution**: Post-process predictions with constraints.

```python
def apply_physics_constraints(predictions, max_speed=10.0, max_accel=5.0, dt=0.1):
    """Apply physical constraints to predictions."""
    
    constrained = predictions.copy()
    
    for i in range(1, len(constrained)):
        # Calculate velocity
        dx = constrained[i, 0] - constrained[i-1, 0]
        dy = constrained[i, 1] - constrained[i-1, 1]
        speed = np.sqrt(dx**2 + dy**2) / dt
        
        # Limit speed
        if speed > max_speed:
            scale = max_speed / speed
            constrained[i, 0] = constrained[i-1, 0] + dx * scale
            constrained[i, 1] = constrained[i-1, 1] + dy * scale
        
        # Calculate acceleration
        if i > 1:
            prev_dx = constrained[i-1, 0] - constrained[i-2, 0]
            prev_dy = constrained[i-1, 1] - constrained[i-2, 1]
            ax = (dx - prev_dx) / (dt**2)
            ay = (dy - prev_dy) / (dt**2)
            accel = np.sqrt(ax**2 + ay**2)
            
            # Limit acceleration
            if accel > max_accel:
                # Smooth acceleration
                scale = max_accel / accel
                constrained[i, 0] = constrained[i-1, 0] + prev_dx + (dx - prev_dx) * scale
                constrained[i, 1] = constrained[i-1, 1] + prev_dy + (dy - prev_dy) * scale
    
    return constrained
```

**Expected Impact**: 1-2% RMSE improvement + more realistic trajectories

---

## Implementation Roadmap

### Week 1: Quick Wins ⚡
- [ ] Complete prediction pipeline (1.1)
- [ ] Add multi-player features (1.2)
- [ ] Implement evaluation metrics (1.3)
- [ ] Retrain model with new features
- [ ] Generate first submission

### Week 2: Model Improvements 🚀
- [ ] Train larger model (d_model=128)
- [ ] Implement data augmentation (3.1)
- [ ] Try different hyperparameters (3.2)
- [ ] Apply physics constraints (4.1)

### Week 3: Advanced Techniques 🎯
- [ ] Implement scheduled sampling (2.2)
- [ ] Build ensemble (3.3)
- [ ] Analyze errors in detail
- [ ] Optimize based on leaderboard feedback

### Week 4: Final Polish ✨
- [ ] Fine-tune best models
- [ ] Create final ensemble
- [ ] Validate on multiple folds
- [ ] Generate final submission

---

## Expected Performance Trajectory

| Stage | Validation RMSE | Improvement |
|-------|----------------|-------------|
| **Current Baseline** | 3.54 yards | - |
| + Multi-player features | 3.35 yards | -5.4% |
| + Larger model | 3.20 yards | -4.5% |
| + Data augmentation | 3.12 yards | -2.5% |
| + Physics constraints | 3.08 yards | -1.3% |
| + Ensemble (3 models) | 2.90 yards | -5.8% |
| **Target** | **< 3.0 yards** | **-15%** |

---

## Key Insights from Your Current Results

### ✅ What's Working Well
1. **Model is learning**: Training loss decreasing consistently
2. **No severe overfitting**: Val loss close to train loss (gap ~0.5 yards)
3. **Good convergence**: Learning rate reduction helping
4. **Reasonable performance**: 3.54 yards RMSE is decent for baseline

### ⚠️ Areas for Improvement
1. **Validation loss plateaued**: After epoch 18, not much improvement
2. **Could use more capacity**: Training loss still decreasing
3. **Missing context**: Single-player modeling limiting performance
4. **Incomplete pipeline**: Can't generate submissions yet

---

## Debugging & Monitoring

### Add These Checks

```python
def sanity_check_predictions(predictions, play_identifiers):
    """Verify predictions are reasonable."""
    
    issues = []
    
    for i, pred in enumerate(predictions):
        # Check for NaN
        if np.isnan(pred).any():
            issues.append(f"Play {i}: Contains NaN")
        
        # Check for extreme values
        if (pred[:, 0] < 0).any() or (pred[:, 0] > 120).any():
            issues.append(f"Play {i}: X out of bounds [0, 120]")
        
        if (pred[:, 1] < 0).any() or (pred[:, 1] > 53.3).any():
            issues.append(f"Play {i}: Y out of bounds [0, 53.3]")
        
        # Check for impossible speeds
        if len(pred) > 1:
            dists = np.sqrt(np.sum(np.diff(pred, axis=0)**2, axis=1))
            speeds = dists / 0.1  # 10 fps = 0.1s per frame
            if (speeds > 15).any():  # 15 yards/sec is very fast
                issues.append(f"Play {i}: Unrealistic speed detected")
    
    if issues:
        print("⚠️ Prediction Issues Found:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
    else:
        print("✅ All predictions passed sanity checks")
    
    return issues
```

---

## Resources & References

### Code Examples
- [Kaggle NFL Big Data Bowl 2024 Winners](https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/discussion)
- [Transformer for Time Series](https://keras.io/examples/timeseries/timeseries_transformer_classification/)
- [Multi-Agent Trajectory Prediction](https://github.com/agrimgupta92/sgan)

### Papers to Read
1. "Attention Is All You Need" - Original Transformer
2. "Social LSTM" - Multi-agent trajectory prediction
3. "TrafficPredict" - Heterogeneous agent modeling

### Tools
- **Weights & Biases**: Track experiments
- **Optuna**: Hyperparameter optimization
- **TensorBoard**: Visualize training

---

## Final Checklist Before Submission

- [ ] Predictions for all required (game_id, play_id, nfl_id, frame_id)
- [ ] No NaN or invalid values
- [ ] Submission format matches exactly: `id,x,y`
- [ ] File size reasonable (<100MB)
- [ ] Sanity checks passed
- [ ] Validated on local test set
- [ ] Model saved for reproducibility

---

## Questions to Answer

1. **What's the competition metric?** (Likely RMSE, but confirm)
2. **What's the leaderboard baseline?** (Check public submissions)
3. **Are there any special rules?** (E.g., only score certain players)
4. **What's the submission deadline?** (Plan accordingly)

---

## Summary

**Your current model is a solid foundation.** The main improvements needed are:

1. **Complete the prediction pipeline** (critical)
2. **Add multi-player context** (high impact)
3. **Increase model capacity** (moderate impact)
4. **Ensemble multiple models** (high impact)

**Estimated timeline**: 2-3 weeks to implement all recommendations.

**Expected final performance**: Sub-3.0 yards RMSE (competitive for leaderboard).

Good luck! 🏈🚀
