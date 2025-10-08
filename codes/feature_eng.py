import os
import panda as pd

def load_and_process_video(video_id, lab_id, data_path):
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

def equalising_data(df,mouse,body_parts):
    universal_columns = []
    for i in mouse:
        for j in body_parts:
            universal_columns.append(f"{i}_{j}_x")
            universal_columns.append(f"{i}_{j}_y")
    new_df = pd.DataFrame()
    for i in universal_columns:
        if(i not in list(df.columns)):
            x = pd.DataFrame({i: [np.nan]})
            new_df = pd.concat([new_df, x],axis=1)
        else:
            new_df = pd.concat([new_df, df[i]],axis=1)
    return new_df
    
