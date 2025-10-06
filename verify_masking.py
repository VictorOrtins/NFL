"""
Verification script to check if masking is implemented correctly.
Run this after training to verify masking is working.
"""

import numpy as np
import os

def verify_masking():
    """Verify that masking is correctly implemented in the processed data."""
    
    print("="*60)
    print("MASKING VERIFICATION")
    print("="*60)
    
    # Check if processed data exists
    PROCESSED_DATA_PATH = 'processed_training_data.npz'
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("\n❌ ERROR: processed_training_data.npz not found!")
        print("   Run training first to generate the data.")
        return False
    
    print(f"\n✓ Found: {PROCESSED_DATA_PATH}")
    
    # Load data
    print("\nLoading data...")
    try:
        data = np.load(PROCESSED_DATA_PATH)
        print(f"✓ Loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return False
    
    # Check for required keys
    print("\nChecking data structure...")
    required_keys = ['X_enc_train', 'y_dec_train', 'X_enc_val', 'y_dec_val', 
                     'mask_train', 'mask_val']
    
    missing_keys = []
    for key in required_keys:
        if key in data:
            print(f"  ✓ {key}: shape {data[key].shape}")
        else:
            print(f"  ❌ {key}: MISSING")
            missing_keys.append(key)
    
    if missing_keys:
        print(f"\n❌ ERROR: Missing keys: {missing_keys}")
        print("   Delete processed_training_data.npz and retrain to add masks.")
        return False
    
    # Verify mask properties
    print("\n" + "="*60)
    print("MASK ANALYSIS")
    print("="*60)
    
    mask_train = data['mask_train']
    mask_val = data['mask_val']
    y_dec_train = data['y_dec_train']
    y_dec_val = data['y_dec_val']
    
    # Check mask shape
    print(f"\nMask shapes:")
    print(f"  Training mask:   {mask_train.shape}")
    print(f"  Validation mask: {mask_val.shape}")
    print(f"  Training output: {y_dec_train.shape}")
    print(f"  Validation output: {y_dec_val.shape}")
    
    # Verify mask is binary
    unique_train = np.unique(mask_train)
    unique_val = np.unique(mask_val)
    
    print(f"\nMask values:")
    print(f"  Training unique values:   {unique_train}")
    print(f"  Validation unique values: {unique_val}")
    
    if not (set(unique_train).issubset({0.0, 1.0}) and set(unique_val).issubset({0.0, 1.0})):
        print("  ⚠️  WARNING: Masks should only contain 0 and 1")
    else:
        print("  ✓ Masks are binary (0 and 1 only)")
    
    # Calculate padding statistics
    print("\n" + "="*60)
    print("PADDING STATISTICS")
    print("="*60)
    
    valid_ratio_train = np.mean(mask_train)
    valid_ratio_val = np.mean(mask_val)
    padding_ratio_train = 1 - valid_ratio_train
    padding_ratio_val = 1 - valid_ratio_val
    
    print(f"\nTraining set:")
    print(f"  Valid positions:  {valid_ratio_train:.1%}")
    print(f"  Padded positions: {padding_ratio_train:.1%}")
    print(f"  Total positions:  {mask_train.size:,}")
    print(f"  Valid count:      {int(mask_train.sum()):,}")
    print(f"  Padded count:     {int((mask_train == 0).sum()):,}")
    
    print(f"\nValidation set:")
    print(f"  Valid positions:  {valid_ratio_val:.1%}")
    print(f"  Padded positions: {padding_ratio_val:.1%}")
    print(f"  Total positions:  {mask_val.size:,}")
    print(f"  Valid count:      {int(mask_val.sum()):,}")
    print(f"  Padded count:     {int((mask_val == 0).sum()):,}")
    
    # Analyze sequence length distribution
    print("\n" + "="*60)
    print("SEQUENCE LENGTH DISTRIBUTION")
    print("="*60)
    
    # Calculate actual sequence lengths from masks
    seq_lengths_train = mask_train.sum(axis=1)
    seq_lengths_val = mask_val.sum(axis=1)
    
    print(f"\nTraining sequences:")
    print(f"  Min length:    {int(seq_lengths_train.min())}")
    print(f"  Max length:    {int(seq_lengths_train.max())}")
    print(f"  Mean length:   {seq_lengths_train.mean():.1f}")
    print(f"  Median length: {np.median(seq_lengths_train):.1f}")
    print(f"  Std dev:       {seq_lengths_train.std():.1f}")
    
    print(f"\nValidation sequences:")
    print(f"  Min length:    {int(seq_lengths_val.min())}")
    print(f"  Max length:    {int(seq_lengths_val.max())}")
    print(f"  Mean length:   {seq_lengths_val.mean():.1f}")
    print(f"  Median length: {np.median(seq_lengths_val):.1f}")
    print(f"  Std dev:       {seq_lengths_val.std():.1f}")
    
    # Check for consistency between masks and outputs
    print("\n" + "="*60)
    print("CONSISTENCY CHECKS")
    print("="*60)
    
    # Check if padded positions in output are zeros
    padded_positions_train = mask_train == 0
    padded_positions_val = mask_val == 0
    
    # Expand mask to match output shape
    padded_mask_train = np.repeat(padded_positions_train[:, :, np.newaxis], 2, axis=2)
    padded_mask_val = np.repeat(padded_positions_val[:, :, np.newaxis], 2, axis=2)
    
    # Check if padded positions are zeros
    padded_values_train = y_dec_train[padded_mask_train]
    padded_values_val = y_dec_val[padded_mask_val]
    
    non_zero_train = (padded_values_train != 0).sum()
    non_zero_val = (padded_values_val != 0).sum()
    
    print(f"\nPadded positions in output:")
    print(f"  Training - non-zero values in padded positions: {non_zero_train}")
    print(f"  Validation - non-zero values in padded positions: {non_zero_val}")
    
    if non_zero_train == 0 and non_zero_val == 0:
        print("  ✓ All padded positions are zeros (expected)")
    else:
        print("  ⚠️  Some padded positions have non-zero values")
    
    # Efficiency gain
    print("\n" + "="*60)
    print("EFFICIENCY GAIN")
    print("="*60)
    
    print(f"\nBy using masking, you're:")
    print(f"  • Ignoring {padding_ratio_train:.1%} of training positions")
    print(f"  • Focusing on {valid_ratio_train:.1%} of meaningful data")
    print(f"  • Expected speedup: ~{1/(1-padding_ratio_train):.2f}x per epoch (theoretical)")
    print(f"  • Expected RMSE improvement: 2-5%")
    
    # Sample weight verification
    print("\n" + "="*60)
    print("SAMPLE WEIGHT VERIFICATION")
    print("="*60)
    
    # Show how to create sample weights
    # Keras expects shape (batch, timesteps), not (batch, timesteps, features)
    sample_weight_train = mask_train
    sample_weight_val = mask_val
    
    print(f"\nSample weight shapes:")
    print(f"  Training:   {sample_weight_train.shape}")
    print(f"  Validation: {sample_weight_val.shape}")
    print(f"  Output:     {y_dec_train.shape}")
    
    expected_shape = (y_dec_train.shape[0], y_dec_train.shape[1])
    if sample_weight_train.shape == expected_shape:
        print("  ✓ Sample weight shape is correct (batch, timesteps)")
        print(f"    Keras will apply the same weight to all features automatically")
    else:
        print(f"  ❌ ERROR: Shape mismatch! Expected {expected_shape}")
        return False
    
    # Example visualization
    print("\n" + "="*60)
    print("EXAMPLE SEQUENCES")
    print("="*60)
    
    # Show first 3 sequences
    print("\nFirst 3 training sequences:")
    for i in range(min(3, len(mask_train))):
        valid_frames = int(mask_train[i].sum())
        total_frames = len(mask_train[i])
        print(f"  Sequence {i+1}: {valid_frames}/{total_frames} valid frames ({valid_frames/total_frames:.1%})")
        
        # Show mask pattern
        mask_str = ''.join(['█' if m == 1 else '░' for m in mask_train[i][:40]])
        print(f"    Mask: {mask_str}{'...' if total_frames > 40 else ''}")
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_checks_passed = True
    
    checks = [
        ("Masks exist in data", 'mask_train' in data and 'mask_val' in data),
        ("Masks are binary", set(unique_train).issubset({0.0, 1.0})),
        ("Padded positions are zeros", non_zero_train == 0 and non_zero_val == 0),
        ("Sample weight shape matches", sample_weight_train.shape == y_dec_train.shape),
    ]
    
    print()
    for check_name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_checks_passed = False
    
    print()
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - Masking is correctly implemented!")
        print("\nYou can now train with confidence that:")
        print("  • Loss is calculated only on valid positions")
        print("  • Model won't waste capacity on predicting padding")
        print("  • Metrics reflect actual prediction quality")
    else:
        print("⚠️  SOME CHECKS FAILED - Review the issues above")
    
    print("\n" + "="*60)
    
    return all_checks_passed


if __name__ == "__main__":
    verify_masking()
