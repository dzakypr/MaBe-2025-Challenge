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

