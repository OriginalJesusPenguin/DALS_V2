#!/usr/bin/env python3
"""
Test script for dice score computation.
"""

import torch
import numpy as np
from model.dals_segmenter import compute_dice_score

def test_dice_score():
    """Test the dice score computation function."""
    print("Testing dice score computation...")
    
    # Test case 1: Perfect prediction
    print("\n1. Testing perfect prediction:")
    batch_size = 2
    height, width, depth = 32, 32, 32
    
    # Create perfect prediction (all foreground)
    pred_perfect = torch.zeros(batch_size, 2, height, width, depth)
    pred_perfect[:, 1, :, :, :] = 10.0  # High probability for foreground
    
    # Create matching ground truth
    target_perfect = torch.zeros(batch_size, 2, height, width, depth)
    target_perfect[:, 1, :, :, :] = 1.0  # All foreground
    
    dice_perfect = compute_dice_score(pred_perfect, target_perfect)
    print(f"Perfect prediction dice score: {dice_perfect:.4f} (expected: ~1.0)")
    
    # Test case 2: No overlap
    print("\n2. Testing no overlap:")
    pred_no_overlap = torch.zeros(batch_size, 2, height, width, depth)
    pred_no_overlap[:, 1, :16, :, :] = 10.0  # First half is foreground
    
    target_no_overlap = torch.zeros(batch_size, 2, height, width, depth)
    target_no_overlap[:, 1, 16:, :, :] = 1.0  # Second half is foreground
    
    dice_no_overlap = compute_dice_score(pred_no_overlap, target_no_overlap)
    print(f"No overlap dice score: {dice_no_overlap:.4f} (expected: ~0.0)")
    
    # Test case 3: Partial overlap
    print("\n3. Testing partial overlap:")
    pred_partial = torch.zeros(batch_size, 2, height, width, depth)
    pred_partial[:, 1, :24, :, :] = 10.0  # First 3/4 is foreground
    
    target_partial = torch.zeros(batch_size, 2, height, width, depth)
    target_partial[:, 1, 8:32, :, :] = 1.0  # Last 3/4 is foreground
    
    dice_partial = compute_dice_score(pred_partial, target_partial)
    print(f"Partial overlap dice score: {dice_partial:.4f} (expected: ~0.5)")
    
    # Test case 4: Random prediction
    print("\n4. Testing random prediction:")
    torch.manual_seed(42)
    pred_random = torch.randn(batch_size, 2, height, width, depth)
    target_random = torch.zeros(batch_size, 2, height, width, depth)
    target_random[:, 1, :, :, :] = (torch.rand(height, width, depth) > 0.5).float()
    
    dice_random = compute_dice_score(pred_random, target_random)
    print(f"Random prediction dice score: {dice_random:.4f}")
    
    # Test case 5: Edge case - empty ground truth
    print("\n5. Testing empty ground truth:")
    pred_empty = torch.zeros(batch_size, 2, height, width, depth)
    pred_empty[:, 1, :, :, :] = 10.0
    
    target_empty = torch.zeros(batch_size, 2, height, width, depth)
    # No foreground in ground truth
    
    dice_empty = compute_dice_score(pred_empty, target_empty)
    print(f"Empty ground truth dice score: {dice_empty:.4f} (expected: 0.0)")
    
    # Test case 6: Edge case - empty prediction
    print("\n6. Testing empty prediction:")
    pred_empty_pred = torch.zeros(batch_size, 2, height, width, depth)
    # No foreground in prediction
    
    target_empty_pred = torch.zeros(batch_size, 2, height, width, depth)
    target_empty_pred[:, 1, :, :, :] = 1.0
    
    dice_empty_pred = compute_dice_score(pred_empty_pred, target_empty_pred)
    print(f"Empty prediction dice score: {dice_empty_pred:.4f} (expected: 0.0)")
    
    print("\nDice score computation test completed!")

if __name__ == '__main__':
    test_dice_score()
