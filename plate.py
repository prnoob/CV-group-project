"""
Improved Segmentation - Better Parameters for Your Dataset
This will fix the remaining issues (too much background, edge noise)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

def segment_depth_based_improved(depth_raw, threshold_percentile=80):
    """
    Improved depth-based segmentation with better threshold
    
    CHANGED: Lower threshold_percentile (85 → 80) to be more selective
    """
    valid_depth = depth_raw > 0
    
    if valid_depth.sum() == 0:
        return np.zeros_like(depth_raw, dtype=bool)
    
    # IMPROVEMENT 1: Use lower percentile to focus on closer objects only
    threshold = np.percentile(depth_raw[valid_depth], threshold_percentile)
    
    # Create initial mask
    mask = (depth_raw > 0) & (depth_raw < threshold)
    
    # IMPROVEMENT 2: Larger kernel for better cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # Was (7,7)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    
    if num_labels > 1:
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_component)
    
    return mask.astype(bool)


def clean_mask_aggressive(mask, min_area=15000):
    """
    More aggressive cleaning to remove edge noise
    
    CHANGED: 
    - Increased min_area (10000 → 15000) to remove more noise
    - Larger morphological kernels
    - Additional erosion step to remove edge artifacts
    """
    mask = mask.astype(np.uint8)
    
    # IMPROVEMENT 1: Larger kernel for more aggressive cleanup
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))  # Was (15,15)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_large)
    
    # IMPROVEMENT 2: Additional erosion to remove edge noise
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        # IMPROVEMENT 3: Higher threshold for minimum area
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1
    
    # Keep only largest
    if clean.sum() > 0:
        num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        if num_labels2 > 1:
            largest = 1 + np.argmax(stats2[1:, cv2.CC_STAT_AREA])
            clean = (labels2 == largest).astype(np.uint8)
    
    # IMPROVEMENT 4: Final dilation to restore some edge (but less than we eroded)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.dilate(clean, kernel_dilate, iterations=1)
    
    return clean.astype(bool)


def segment_with_center_bias(depth_raw, threshold_percentile=80, center_weight=1.5):
    """
    NEW METHOD: Bias toward the center of the image (where plate usually is)
    
    This helps when background objects are at similar depth
    """
    h, w = depth_raw.shape
    
    # Get initial mask
    mask = segment_depth_based_improved(depth_raw, threshold_percentile)
    
    # Create center bias weight map
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    
    # Distance from center (normalized)
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    dist_from_center = dist_from_center / dist_from_center.max()
    
    # Create weight map (higher weight in center)
    weight_map = 1.0 - (dist_from_center * 0.5)  # Center: 1.0, Edges: 0.5
    
    # Apply weights to mask
    # Keep only regions that are both in mask AND reasonably centered
    mask_weighted = mask.astype(float) * weight_map
    
    # Threshold the weighted mask
    mask_centered = mask_weighted > 0.3  # Keep only well-centered regions
    
    # Clean up
    mask_centered = clean_mask_aggressive(mask_centered.astype(np.uint8), min_area=15000)
    
    return mask_centered


def try_all_methods(rgb_image, depth_raw):
    """
    Compare different parameter settings
    """
    print("Testing different segmentation parameters...")
    
    # Method 1: Your current approach (baseline)
    mask1 = segment_depth_based_improved(depth_raw, threshold_percentile=50)
    mask1 = clean_mask_aggressive(mask1, min_area=15000)
    
    # Method 2: More selective (lower percentile)
    mask2 = segment_depth_based_improved(depth_raw, threshold_percentile=65)
    mask2 = clean_mask_aggressive(mask2, min_area=15000)
    
    # Method 3: Very selective (even lower)
    mask3 = segment_depth_based_improved(depth_raw, threshold_percentile=70)
    mask3 = clean_mask_aggressive(mask3, min_area=15000)
    
    # Method 4: Center-biased
    mask4 = segment_with_center_bias(depth_raw, threshold_percentile=50)
    
    # Visualize all
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    masks = [mask1, mask2, mask3, mask4]
    titles = [
        f'Current (85%, 10k)\n{mask1.sum():,} px ({100*mask1.sum()/mask1.size:.1f}%)',
        f'Method 2 (80%, 15k)\n{mask2.sum():,} px ({100*mask2.sum()/mask2.size:.1f}%)',
        f'Method 3 (75%, 15k)\n{mask3.sum():,} px ({100*mask3.sum()/mask3.size:.1f}%)',
        f'Center-Biased\n{mask4.sum():,} px ({100*mask4.sum()/mask4.size:.1f}%)'
    ]
    
    for i, (mask, title) in enumerate(zip(masks, titles)):
        # Top row: masks
        axes[0, i].imshow(mask, cmap='gray')
        axes[0, i].set_title(title, fontsize=10)
        axes[0, i].axis('off')
        
        # Bottom row: overlays
        overlay = rgb_image.copy()
        overlay[~mask] = (overlay[~mask] * 0.3).astype(np.uint8)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'Result {i+1}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle('Comparison of Different Parameters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return masks


# ==========================================
# RECOMMENDED SETTINGS FOR YOUR DATASET
# ==========================================

def segment_recommended(depth_raw):
    """
    RECOMMENDED: Use this for your dataset
    
    Based on your images showing ~70% coverage,
    we need to be more selective
    """
    # Step 1: More selective depth threshold
    mask = segment_depth_based_improved(depth_raw, threshold_percentile=78)  # Lower!
    
    # Step 2: Aggressive cleaning
    mask = clean_mask_aggressive(mask, min_area=15000)  # Higher!
    
    return mask


# ==========================================
# USAGE EXAMPLE
# ==========================================
def run_all_image():
    # --- Minimal config ---
    csv_path  = 'train_index.csv'     # columns: dish_id,label,rgb_path,depth_color_path,depth_raw_path
    root_dir  = '.'                   # base dir for relative paths

    df = pd.read_csv(csv_path)

    for i, row in df.iterrows():
        dish_id   = str(row['dish_id'])
        rgb_path  = Path(root_dir) / Path(str(row['rgb_path']).replace('\\', '/'))
        depth_path= Path(root_dir) / Path(str(row['depth_raw_path']).replace('\\', '/'))

        rgb  = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"[skip] RGB not found: {rgb_path}")
            continue
        rgb  = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth is None:
            print(f"[skip] DEPTH not found: {depth_path}")
            continue

        print(f"[{i+1}/{len(df)}] {dish_id} -> run try_all_methods")
        # masks = try_all_methods(rgb, depth)  # dict[name -> mask ndarray]

        # out_dir = Path(output_dir) / dish_id
        # out_dir.mkdir(parents=True, exist_ok=True)
        try_best_methods(rgb, depth, dish_id, out_dir=r"C:\Users\Admin\Documents\GitHub\CV-group-project\outputs")
        # for name, m in masks.items():
        #     if m is None:
        #         continue
        #     # Minimal dtype handling: bool -> uint8 0/255
        #     if m.dtype == bool:
        #         m = (m.astype(np.uint8) * 255)
        #     # Resize to match RGB HxW if needed (keep NEAREST for binary masks)
        #     if m.shape[:2] != rgb.shape[:2]:
        #         m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        #     cv2.imwrite(str(out_dir / f"{name}.png"), m if m.ndim == 2 else cv2.cvtColor(m, cv2.COLOR_RGB2BGR))

def try_best_methods(rgb_image, depth_raw, dish_id, out_dir="outputs"):
    """
    Run only the current (baseline) segmentation and save results.
    - Saves overlay image: out_dir / f"dish{dish_id}_masked.png"
    - Saves binary mask:   out_dir / f"dish{dish_id}_mask.png"
    Returns:
        mask (np.bool_ array)
    """

    # --- ensure output directory exists ---
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # --- Current baseline segmentation ---
    # (keep your original parameters for minimal change)
    mask = segment_depth_based_improved(depth_raw, threshold_percentile=50)
    mask = clean_mask_aggressive(mask, min_area=15000)

    # --- Build overlay (dim background) ---
    overlay = rgb_image.copy()
    overlay[~mask] = (overlay[~mask] * 0.3).astype(np.uint8)

    # --- Save results ---
    base = Path(out_dir) / f"dish_{dish_id}"
    # Save overlay (convert RGB -> BGR for cv2.imwrite)
    cv2.imwrite(str(base.with_name(base.name + "_masked").with_suffix(".png")),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    # Save binary mask as 0/255 PNG
    cv2.imwrite(str(base.with_name(base.name + "_unmask").with_suffix(".png")),
                (mask.astype(np.uint8) * 255))

    return mask

if __name__ == "__main__":
    run_all_image()
#     import pandas as pd
#     from pathlib import Path
    
#     # Configuration
#     csv_path = 'train_index.csv'
#     root_dir = '.'
#     dish_index = 0
    
#     # Load data
#     df = pd.read_csv(csv_path)
#     row = df.iloc[dish_index]
    
#     # Read images
#     rgb_path = row['rgb_path'].replace('\\', '/')
#     depth_path = row['depth_raw_path'].replace('\\', '/')
    
#     rgb_image = cv2.imread(str(Path(root_dir) / rgb_path))
#     rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
#     depth_raw = cv2.imread(str(Path(root_dir) / depth_path), cv2.IMREAD_ANYDEPTH)
    
#     print(f"\n{'='*60}")
#     print(f"Testing improved segmentation on {row['dish_id']}")
#     print(f"{'='*60}\n")
    
#     # Try all methods
#     masks = try_all_methods(rgb_image, depth_raw)



