"""
1.py - Food Segmentation with Plate/Food Separation
Separates food from empty plate for better calorie estimation
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def segment_plate_region(depth_raw, threshold_percentile=70):
    """Find the plate region"""
    valid_depth = depth_raw > 0
    if valid_depth.sum() == 0:
        return np.zeros_like(depth_raw, dtype=bool)
    
    threshold = np.percentile(depth_raw[valid_depth], threshold_percentile)
    mask = (depth_raw > 0) & (depth_raw < threshold)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest)
    
    return mask.astype(bool)


def separate_food_from_plate(rgb_image, depth_raw, plate_mask):
    """
    Separate FOOD from PLATE using depth + color
    
    Key insight:
    - Food has DIFFERENT depth than plate surface
    - Food is more COLORFUL than white plate
    """
    if plate_mask.sum() == 0:
        return np.zeros_like(plate_mask, dtype=bool)
    
    # Get depth within plate
    plate_depth = depth_raw[plate_mask]
    valid = plate_depth > 0
    
    if valid.sum() == 0:
        return plate_mask
    
    valid_depths = plate_depth[valid]
    depth_std = np.std(valid_depths)
    
    # METHOD 1: Depth-based (food has different depth than plate)
    if depth_std < 50:
        # Low variation = flat surface, use color
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        food_mask = plate_mask & ((s > 30) | (v < 150))
    else:
        # High variation = has food with different depths
        depth_20 = np.percentile(valid_depths, 20)
        depth_80 = np.percentile(valid_depths, 80)
        food_mask = plate_mask & ((depth_raw < depth_20) | (depth_raw > depth_80))
    
    # METHOD 2: Color enhancement (food != white)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    colorful = (s > 30) | (v < 150)
    food_mask = food_mask | (plate_mask & colorful)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    food_mask = cv2.morphologyEx(food_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small regions (noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        food_mask.astype(np.uint8), connectivity=8)
    
    clean_food = np.zeros_like(food_mask, dtype=bool)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 500:  # Min 500 pixels
            clean_food[labels == i] = True
    
    return clean_food


def visualize(rgb_image, plate_mask, food_mask, title="Segmentation"):
    """Visualize plate and food separation"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original, Plate mask, Food mask
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('Original RGB', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(plate_mask, cmap='gray')
    plate_pct = 100 * plate_mask.sum() / plate_mask.size
    axes[0, 1].set_title(f'Plate Region\n{plate_mask.sum():,} px ({plate_pct:.1f}%)', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(food_mask, cmap='gray')
    food_pct = 100 * food_mask.sum() / food_mask.size if food_mask.sum() > 0 else 0
    axes[0, 2].set_title(f'Food Only\n{food_mask.sum():,} px ({food_pct:.1f}%)', 
                        fontsize=12, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    # Row 2: Original, Plate overlay, Food overlay
    axes[1, 0].imshow(rgb_image)
    axes[1, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    plate_overlay = rgb_image.copy()
    plate_overlay[~plate_mask] = (plate_overlay[~plate_mask] * 0.3).astype(np.uint8)
    axes[1, 1].imshow(plate_overlay)
    axes[1, 1].set_title('Plate Region', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    food_overlay = rgb_image.copy()
    if food_mask.sum() > 0:
        food_overlay[~food_mask] = (food_overlay[~food_mask] * 0.3).astype(np.uint8)
        axes[1, 2].imshow(food_overlay)
        food_of_plate = 100 * food_mask.sum() / plate_mask.sum() if plate_mask.sum() > 0 else 0
        axes[1, 2].set_title(f'Food Only ✨\n{food_of_plate:.1f}% of plate', 
                           fontsize=12, fontweight='bold', color='green')
    else:
        axes[1, 2].imshow(rgb_image)
        axes[1, 2].text(0.5, 0.5, 'No Food', ha='center', va='center',
                       transform=axes[1, 2].transAxes, fontsize=14, color='red')
        axes[1, 2].set_title('No Food Detected', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def extract_features(rgb_image, depth_image, plate_mask, food_mask):
    """
    Extract features from BOTH plate region AND food-only region
    """
    features = {}
    
    # === PLATE FEATURES ===
    if plate_mask.sum() > 0:
        plate_depth = depth_image[plate_mask]
        valid = plate_depth > 0
        if valid.sum() > 0:
            features['plate_volume'] = (plate_depth[valid].max() - plate_depth[valid]).sum()
            features['plate_area'] = plate_mask.sum()
            features['plate_mean_depth'] = plate_depth[valid].mean()
        else:
            features['plate_volume'] = 0
            features['plate_area'] = plate_mask.sum()
            features['plate_mean_depth'] = 0
        
        plate_rgb = rgb_image[plate_mask]
        features['plate_mean_r'], features['plate_mean_g'], features['plate_mean_b'] = plate_rgb.mean(axis=0)
    else:
        features['plate_volume'] = 0
        features['plate_area'] = 0
        features['plate_mean_depth'] = 0
        features['plate_mean_r'] = 0
        features['plate_mean_g'] = 0
        features['plate_mean_b'] = 0
    
    # === FOOD FEATURES (MOST IMPORTANT!) ===
    if food_mask.sum() > 0:
        food_depth = depth_image[food_mask]
        valid = food_depth > 0
        if valid.sum() > 0:
            features['food_volume'] = (food_depth[valid].max() - food_depth[valid]).sum()
            features['food_area'] = food_mask.sum()
            features['food_mean_depth'] = food_depth[valid].mean()
            features['food_depth_std'] = food_depth[valid].std()
        else:
            features['food_volume'] = 0
            features['food_area'] = food_mask.sum()
            features['food_mean_depth'] = 0
            features['food_depth_std'] = 0
        
        food_rgb = rgb_image[food_mask]
        features['food_mean_r'], features['food_mean_g'], features['food_mean_b'] = food_rgb.mean(axis=0)
        features['food_std_r'], features['food_std_g'], features['food_std_b'] = food_rgb.std(axis=0)
        
        # Texture
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        features['food_texture'] = cv2.Laplacian(gray, cv2.CV_64F)[food_mask].var()
        
        # Food coverage
        features['food_coverage'] = food_mask.sum() / plate_mask.sum() if plate_mask.sum() > 0 else 0
    else:
        features['food_volume'] = 0
        features['food_area'] = 0
        features['food_mean_depth'] = 0
        features['food_depth_std'] = 0
        features['food_mean_r'] = 0
        features['food_mean_g'] = 0
        features['food_mean_b'] = 0
        features['food_std_r'] = 0
        features['food_std_g'] = 0
        features['food_std_b'] = 0
        features['food_texture'] = 0
        features['food_coverage'] = 0
    
    return features


def process_dish(csv_path, root_dir, dish_index):
    """Process single dish with food separation"""
    df = pd.read_csv(csv_path)
    row = df.iloc[dish_index]
    
    # Load images
    rgb_path = row['rgb_path'].replace('\\', '/')
    rgb = cv2.imread(str(Path(root_dir) / rgb_path))
    if rgb is None:
        print(f"❌ Cannot load {rgb_path}")
        return None
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    depth_path = row['depth_raw_path'].replace('\\', '/')
    depth = cv2.imread(str(Path(root_dir) / depth_path), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        print(f"❌ Cannot load {depth_path}")
        return None
    
    # Step 1: Segment plate
    plate_mask = segment_plate_region(depth, threshold_percentile=70)
    
    # Step 2: Separate food from plate
    food_mask = separate_food_from_plate(rgb, depth, plate_mask)
    
    # Extract features
    features = extract_features(rgb, depth, plate_mask, food_mask)
    features['dish_id'] = row['dish_id']
    features['calories'] = row['label']
    
    # Info
    print(f"\n✅ {row['dish_id']}: {row['label']:.1f} cal")
    print(f"   Plate: {plate_mask.sum():,} px ({100*plate_mask.sum()/plate_mask.size:.1f}%)")
    print(f"   Food: {food_mask.sum():,} px ({100*food_mask.sum()/food_mask.size:.1f}%)")
    if plate_mask.sum() > 0:
        print(f"   Food covers {100*food_mask.sum()/plate_mask.sum():.1f}% of plate")
    print(f"   Food volume: {features['food_volume']:.0f}")
    
    # Visualize
    visualize(rgb, plate_mask, food_mask, f"{row['dish_id']}: {row['label']:.1f} cal")
    plt.show()
    
    return features


def process_all(csv_path, root_dir, max_dishes=3301):
    """Process all dishes"""
    df = pd.read_csv(csv_path)
    features_list = []
    
    print(f"Processing {min(max_dishes, len(df))} dishes...")
    
    for idx in range(min(max_dishes, len(df))):
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{max_dishes}...")
        
        try:
            row = df.iloc[idx]
            
            rgb_path = row['rgb_path'].replace('\\', '/')
            rgb = cv2.imread(str(Path(root_dir) / rgb_path))
            if rgb is None:
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            depth_path = row['depth_raw_path'].replace('\\', '/')
            depth = cv2.imread(str(Path(root_dir) / depth_path), cv2.IMREAD_ANYDEPTH)
            if depth is None:
                continue
            
            plate_mask = segment_plate_region(depth, 70)
            food_mask = separate_food_from_plate(rgb, depth, plate_mask)
            
            features = extract_features(rgb, depth, plate_mask, food_mask)
            features['dish_id'] = row['dish_id']
            features['calories'] = row['label']
            features_list.append(features)
            
        except Exception as e:
            print(f"  Error {idx}: {e}")
    
    return pd.DataFrame(features_list)


if __name__ == "__main__":
    print("="*60)
    print("🍽️  FOOD SEGMENTATION WITH PLATE/FOOD SEPARATION")
    print("="*60)
    
    csv_path = 'train_index.csv'
    root_dir = '.'
    
    # Test single dish
    print("\n📋 Testing with food separation...")
    process_dish(csv_path, root_dir, 1122)
    