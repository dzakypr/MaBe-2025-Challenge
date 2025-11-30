import os
import pandas as pd
import numpy as np
import ast

def tracked_body_parts():
    tracked_body_parts = ['body_center',
 'ear_left',
 'ear_right',
 'headpiece_bottombackleft',
 'headpiece_bottombackright',
 'headpiece_bottomfrontleft',
 'headpiece_bottomfrontright',
 'headpiece_topbackleft',
 'headpiece_topbackright',
 'headpiece_topfrontleft',
 'headpiece_topfrontright',
 'lateral_left',
 'lateral_right',
 'neck',
 'nose',
 'tail_base',
 'tail_midpoint',
 'tail_tip',
 'hip_left',
 'hip_right',
 'head',
 'forepaw_left',
 'forepaw_right',
 'hindpaw_left',
 'hindpaw_right',
 'spine_1',
 'spine_2',
 'tail_middle_1',
 'tail_middle_2']
    return tracked_body_parts

def load_and_process_video(data_path, lab_id, video_id, pixel_per_cm, add_bp=None, drop_bp=None):
    tracking_path = os.path.join(data_path, 'train_tracking', lab_id, f'{video_id}.parquet')
    if not os.path.exists(tracking_path):
        return None

    df_long = pd.read_parquet(tracking_path)
    df_wide = df_long.pivot(index="video_frame", columns=["mouse_id", "bodypart"], values=["x","y"])
    df_wide.columns = df_wide.columns.swaplevel(0, 1)
    df_wide.columns = df_wide.columns.swaplevel(1, 2)
    df_wide = df_wide.sort_index(axis=1)

    if add_bp is not None:
        new_nan_parts_list = []
        for m_id in range(1, 5):
            for part in add_bp:
                if (m_id,part,'x') in df_wide.columns:
                    continue
                nan_cols = pd.MultiIndex.from_product(
                    [[m_id], [part], ['x', 'y']],
                    names=df_wide.columns.names
                )

                part_df = pd.DataFrame(
                    np.nan,
                    index=df_wide.index,
                    columns=nan_cols
                )
                new_nan_parts_list.append(part_df)
        if new_nan_parts_list:
            df_wide = pd.concat([df_wide] + new_nan_parts_list, axis=1)

    if drop_bp is not None:
        df_wide = df_wide.drop(columns=drop_bp, level='bodypart', errors='ignore')
    df_wide = df_wide.sort_index(axis=1)
    df_wide = df_wide / pixel_per_cm
    return df_wide

def load_and_process_annotate(lab_id, video_id, path, inclusive=True):
    tracking_path = os.path.join(path, 'train_annotation', lab_id, f'{video_id}.parquet')
    if not os.path.exists(tracking_path):
        return None

    df = pd.read_parquet(tracking_path)
    starts = df['start_frame'].to_numpy()
    stops  = df['stop_frame'].to_numpy()

    if inclusive:
        lengths = (stops - starts + 1).astype(np.int64)
        ranges  = [np.arange(s, e + 1, dtype=np.int32) for s, e in zip(starts, stops)]
    else:
        lengths = (stops - starts).astype(np.int64)
        ranges  = [np.arange(s, e, dtype=np.int32) for s, e in zip(starts, stops)]

    frames = np.concatenate(ranges, axis=0) if len(ranges) else np.array([], dtype=np.int32)
    agent  = np.repeat(df['agent_id'].to_numpy(), lengths).astype(np.int16, copy=False)
    target = np.repeat(df['target_id'].to_numpy(), lengths).astype(np.int16, copy=False)
    action = np.repeat(df['action'].to_numpy(),    lengths)

    out = pd.DataFrame(
        {'frame': frames, 'agent_id': agent, 'target_id': target, 'action': action},
        copy=False
    ).sort_values('frame', kind='mergesort', ignore_index=True)

    out['action'] = out['action'].astype('category')
    return out

def extract_tracked_body_parts(df):
    body_parts_tracked_str = df['body_parts_tracked'].unique()
    body_parts_tracked_arr = []

    for i in body_parts_tracked_str:
        body_parts_tracked_arr.append(ast.literal_eval(i))

    body_parts_tracked_list = []

    for arr in body_parts_tracked_arr:
        for i in arr:
            if i not in body_parts_tracked_list:
                body_parts_tracked_list.append(i)
                
    return body_parts_tracked_list

import torch
import pandas as pd
import os
import random
import numpy as np
from tqdm import tqdm

def calculate_robust_stats(feature_root_dir, save_path="robust_stats.pt", sample_ratio=0.1):
    """
    Calculates Median and IQR (Interquartile Range) for Robust Scaling.
    Uses Pandas to handle NaNs gracefully.
    """
    print(f"Scanning {feature_root_dir}...")
    
    files = []
    for root, _, filenames in os.walk(feature_root_dir):
        for f in filenames:
            if f.endswith('.parquet'):
                files.append(os.path.join(root, f))
    
    if not files:
        print("No files found.")
        return

    # Randomly sample files to estimate stats
    sample_size = max(1, int(len(files) * sample_ratio))
    sampled_files = random.sample(files, sample_size)
    print(f"Sampling {sample_size}/{len(files)} files for robust statistics...")

    # We iterate and build a large DataFrame/Array for accurate quantiles
    # To save memory, we only take a subset of frames from each file
    sampled_data = []

    for f in tqdm(sampled_files):
        try:
            df = pd.read_parquet(f)
            # Downsample frames (every 10th frame) to fit in RAM
            # Pandas handles the mixed types/NaNs better here
            subset = df.iloc[::10].select_dtypes(include=[np.number])
            sampled_data.append(subset)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not sampled_data:
        print("No valid data found.")
        return

    print("Concatenating samples...")
    full_df = pd.concat(sampled_data, axis=0)
    
    print(f"Computing quantiles on shape: {full_df.shape}")
    
    # Calculate quantiles ignoring NaNs
    # This ensures a missing tail point doesn't break the scaler
    median_series = full_df.median()
    q25_series = full_df.quantile(0.25)
    q75_series = full_df.quantile(0.75)
    
    iqr_series = q75_series - q25_series
    
    # Safety: Prevent division by zero
    iqr_series = iqr_series.replace(0, 1.0)

    # Convert to PyTorch Tensors for the loader
    stats = {
        "median": torch.tensor(median_series.values, dtype=torch.float32),
        "iqr": torch.tensor(iqr_series.values, dtype=torch.float32)
    }
    
    torch.save(stats, save_path)
    print(f"Robust stats saved to {save_path}")

# --- Usage ---
# calculate_robust_stats("/content/drive/Shareddrives/Feature", "robust_stats.pt")