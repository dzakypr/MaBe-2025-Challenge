import os
import pandas as pd
import numpy as np

def load_and_process_video(lab_id, video_id, data_path):
    tracking_path = os.path.join(data_path, 'train_tracking', lab_id, f'{video_id}.parquet')
    if not os.path.exists(tracking_path):
        return None
    df_long = pd.read_parquet(tracking_path)
    pivot_x = df_long.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='x')
    pivot_y = df_long.pivot(index='video_frame', columns=['mouse_id', 'bodypart'], values='y')
    pivot_x.columns = [f"mouse{m}_{bp}_x" for m, bp in pivot_x.columns]
    pivot_y.columns = [f"mouse{m}_{bp}_y" for m, bp in pivot_y.columns]
    df_wide = pd.concat([pivot_x, pivot_y], axis=1).sort_index(axis=1)
    return df_wide

def equalising_data(df,num_mouse,body_parts):
    universal_columns = [f"mouse{m}_{bp}_{coord}" 
                         for m in range(1,num_mouse+1) 
                         for bp in body_parts 
                         for coord in ['x', 'y']]
    for col in universal_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df.reindex(columns=universal_columns)
    return df