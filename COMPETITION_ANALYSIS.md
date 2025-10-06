# NFL Big Data Bowl 2026 - Competition Analysis

## Competition Overview

### Objective
Predict player movement trajectories during the time when a football pass is in the air, from the moment the quarterback releases the ball until it's caught or ruled incomplete.

### Key Challenge
Given:
- Pre-pass tracking data (player positions, velocities, orientations before the ball is thrown)
- Target receiver identification
- Ball landing location (x, y coordinates)

Predict:
- Frame-by-frame (x, y) positions for all players while the ball is in the air
- 10 frames per second (e.g., 2.5 second pass = 25 frames to predict)

### Evaluation
- **Training data**: Historical 2023 NFL season data (weeks 1-18)
- **Leaderboard**: Live evaluation on last 5 weeks of 2025 NFL season (future data)
- **Metric**: Not explicitly stated, but likely Mean Squared Error or similar distance-based metric

---

## Data Structure

### Input Files (train/input_2023_w[01-18].csv)
**Shape**: ~22 columns per frame, multiple frames per player per play

**Key Features**:
1. **Identifiers**:
   - `game_id`: Unique game identifier
   - `play_id`: Play identifier (unique within game)
   - `nfl_id`: Player identifier
   - `frame_id`: Frame number in sequence (10 fps)
   - `player_to_predict`: Boolean flag indicating if this player's trajectory will be scored

2. **Spatial Features**:
   - `x`: Position along field length (0-120 yards)
   - `y`: Position along field width (0-53.3 yards)
   - `s`: Speed (yards/second)
   - `a`: Acceleration (yards/second²)
   - `dir`: Direction of motion (degrees)
   - `o`: Player orientation (degrees)

3. **Context Features**:
   - `play_direction`: Offense moving left or right
   - `absolute_yardline_number`: Distance from end zone
   - `ball_land_x`, `ball_land_y`: Where the ball will land
   - `num_frames_output`: Number of frames to predict for this player

4. **Player Attributes**:
   - `player_name`, `player_height`, `player_weight`, `player_birth_date`
   - `player_position`: Position (QB, WR, CB, etc.)
   - `player_side`: Offense or Defense
   - `player_role`: Targeted Receiver, Defensive Coverage, Passer, Other Route Runner

### Output Files (train/output_2023_w[01-18].csv)
**Shape**: Simple format with predictions

**Columns**:
- `game_id`, `play_id`, `nfl_id`, `frame_id`
- `x`, `y`: **TARGET POSITIONS TO PREDICT**

### Test Files
- `test_input.csv`: Same structure as training input (no ground truth)
- `test.csv`: Mock test set showing required submission format (game_id, play_id, nfl_id, frame_id)

---

## Current Implementation Analysis

### Your Existing Approach (train_gpt.ipynb)

#### Architecture: Transformer Encoder-Decoder
You've implemented a sequence-to-sequence transformer model:

**Strengths**:
1. ✅ **Attention mechanism**: Can capture relationships between players
2. ✅ **Positional encoding**: Handles temporal sequences well
3. ✅ **Feature engineering**: Good features (distance to landing spot, velocity components, etc.)
4. ✅ **Proper preprocessing**: StandardScaler for numeric, OneHotEncoder for categorical

**Current Configuration**:
```python
D_MODEL = 64 (or 128)
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DFF = 512
MAX_INPUT_LEN = variable (sequence of pre-pass frames)
MAX_OUTPUT_LEN = variable (frames while ball is in air)
```

#### Potential Issues & Improvements

1. **Single-Player Modeling**:
   - Current approach: Predicts each player independently
   - **Problem**: Ignores inter-player interactions (defenders reacting to receivers, etc.)
   - **Solution**: Consider multi-agent modeling or graph neural networks

2. **Teacher Forcing in Decoder**:
   - Training uses ground truth previous positions
   - Inference uses predicted positions (exposure bias)
   - **Solution**: Scheduled sampling or autoregressive training

3. **Fixed Sequence Length**:
   - Padding to MAX_OUTPUT_LEN may be inefficient
   - Different plays have different pass durations
   - **Current handling**: `num_frames_output` tracks actual length (good!)

4. **Missing Context**:
   - Not explicitly modeling other players' positions
   - Ball trajectory not modeled (only landing spot given)
   - **Solution**: Add relative positions to other players as features

5. **Evaluation Metric**:
   - Using MSE loss
   - Should verify if competition uses same metric
   - Consider adding physics-based constraints (speed limits, acceleration limits)

---

## Recommended Improvements

### Priority 1: Multi-Player Context
**Add features capturing relationships between players**:
```python
# For each player, add:
- Distance to nearest defender/receiver
- Relative velocity to target receiver
- Number of players within 5-yard radius
- Average position of defensive/offensive players
```

### Priority 2: Trajectory Modeling
**Instead of frame-by-frame prediction, model trajectory parameters**:
- Predict velocity and acceleration curves
- Integrate to get positions (more physically consistent)
- Add constraints: max speed ~10 yards/sec, max acceleration ~5 yards/sec²

### Priority 3: Graph Neural Network (GNN)
**Model player interactions explicitly**:
```python
# Graph structure:
- Nodes: Players
- Edges: Proximity-based connections
- Node features: Current state (x, y, s, a, dir, o)
- Edge features: Relative positions, velocities
- Message passing: Players influence each other's predictions
```

### Priority 4: Ensemble Methods
**Combine multiple models**:
- Transformer (current)
- LSTM/GRU baseline
- Physics-based model
- Weighted average or stacking

### Priority 5: Data Augmentation
**Increase training data**:
- Flip plays horizontally (left ↔ right)
- Add noise to positions (simulate tracking errors)
- Temporal augmentation (different starting frames)

---

## Evaluation Strategy

### Cross-Validation
Current: Week-based split (good!)
- Train: Weeks 1-16
- Validation: Weeks 17-18

**Recommendation**: 
- Use multiple folds (e.g., 3-fold by weeks)
- Validate on different game scenarios (long passes, short passes, different field positions)

### Metrics to Track
1. **Overall MSE/RMSE**: Primary metric
2. **Per-role accuracy**: Separate metrics for:
   - Targeted receivers (most important)
   - Defensive coverage
   - Other route runners
3. **Temporal accuracy**: Error by frame (early frames vs. late frames)
4. **Spatial accuracy**: Error by field region

### Error Analysis
Look for patterns in errors:
- Do errors increase over time (accumulation)?
- Are certain positions harder to predict (DBs vs. WRs)?
- Are long passes harder than short passes?

---

## Next Steps

### Immediate Actions
1. ✅ **Verify current model performance**:
   - Check validation RMSE
   - Visualize predictions vs. ground truth
   - Identify systematic errors

2. **Quick Wins**:
   - Add multi-player context features
   - Tune hyperparameters (d_model, num_layers)
   - Implement better inference strategy (no teacher forcing)

3. **Medium-term**:
   - Implement GNN architecture
   - Create ensemble with current transformer
   - Add physics constraints

4. **Long-term**:
   - Analyze competition leaderboard patterns
   - Study top solutions from previous Big Data Bowls
   - Consider domain-specific knowledge (football tactics)

---

## Code Quality Improvements

### Current Code Structure
✅ Good modular design with utility functions
✅ Data caching (processed_training_data.npz)
✅ Proper train/val split

### Suggestions
1. **Add logging**: Track experiments systematically
2. **Config management**: Use YAML/JSON for hyperparameters
3. **Visualization**: Add trajectory plotting functions
4. **Metrics**: Implement competition-specific evaluation
5. **Documentation**: Add docstrings to all functions

---

## Resources & References

### Similar Competitions
- NFL Big Data Bowl 2024, 2023, 2022 (check winning solutions on Kaggle)
- NBA Player Movement Prediction
- Soccer Player Tracking competitions

### Relevant Papers
- "Attention Is All You Need" (Transformer architecture)
- "Graph Networks for Multi-Agent Trajectory Prediction"
- "Social LSTM: Human Trajectory Prediction in Crowded Spaces"
- "TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents"

### Key Concepts
- **Sequence-to-sequence modeling**
- **Multi-agent trajectory prediction**
- **Graph neural networks**
- **Physics-informed neural networks**
- **Attention mechanisms**

---

## Competition Timeline Considerations

### Live Leaderboard (2025 Season)
- Predictions on future games (weeks TBD)
- Need robust model that generalizes
- Avoid overfitting to 2023 data
- Consider year-over-year changes (rule changes, team strategies)

### Submission Format
```csv
id,x,y
{game_id}_{play_id}_{nfl_id}_{frame_id},predicted_x,predicted_y
```

---

## Summary

**Your current approach is solid** with a transformer-based sequence model. The main areas for improvement are:

1. **Inter-player modeling**: Add features or architecture to capture player interactions
2. **Inference strategy**: Fix teacher forcing issue in prediction
3. **Evaluation**: Implement comprehensive metrics and error analysis
4. **Experimentation**: Try GNN, ensemble methods, physics constraints

**Recommended Priority**:
1. First, validate current model thoroughly
2. Add multi-player context features (quick win)
3. Implement GNN or attention-based multi-agent model
4. Create ensemble for final submission

Good luck with the competition! 🏈
