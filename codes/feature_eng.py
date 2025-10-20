import os
import pandas as pd
import numpy as np
import itertools

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

class feat_eng_class:
    def __init__(self, col, num_mouse, metadata):
        self.col = col
        self.num_mouse = num_mouse
        self.video_width_pix = metadata["video_width_pix"]
        self.video_height_pix = metadata["video_height_pix"]
        self.fps = metadata["frames_per_second"]
        self.pix_per_cm = metadata["pix_per_cm_approx"]

    def smooth_series(self, s, win=5):
        return s.rolling(win, center=True, min_periods=1).median()

    # Convert px to cm
    def px_to_cm(self, df, col=None):
        if col == None:
            col = self.col
        for i in list(df.columns):
            df[i] = df[i]/self.pix_per_cm
        return df

    # Make (0,0) The Center
    def relocate_center(self,df,col=None):
        if col == None:
            col = self.col
        for i in range(1, self.num_mouse+1):
            for c in col:
                sel_col = f"mouse{i}_{c}"
                df[f"{sel_col}_x"] = df[f"{sel_col}_x"] - (self.video_width_pix/2)
                df[f"{sel_col}_y"] = df[f"{sel_col}_y"] - (self.video_height_pix/2)
        return df

    # Movement Dynamics
    def calc_vel(self, df, col=None):
        if col == None:
            col = self.col
        for i in range(1, self.num_mouse+1):
            for c in col:
                sel_col = f"mouse{i}_{c}"
                xs, ys = self.smooth_series(df[f"{sel_col}_x"]), self.smooth_series(df[f"{sel_col}_y"])
                dx, dy = xs.diff(), ys.diff()
                df[f"{sel_col}_vx"] = dx * self.fps
                df[f"{sel_col}_vy"] = dy * self.fps
                df[f"{sel_col}_vel"] = np.hypot(dx,dy) * self.fps
        return df

    def calc_acc(self, df, col=None):
        if col == None:
            col = self.col
        for i in range(1, self.num_mouse+1):
            for c in col:
                sel_col = f"mouse{i}_{c}"
                df[f"{sel_col}_ax"] = df[f"{sel_col}_vx"].diff() * self.fps
                df[f"{sel_col}_ay"] = df[f"{sel_col}_vy"].diff() * self.fps
                df[f"{sel_col}_acc"] = np.hypot(df[f"{sel_col}_ax"], df[f"{sel_col}_ay"])
        return df

    # Distance Metrics
    def calc_dist_pair(self, df, col=None):
        if col == None:
            col = self.col
        for m in list(itertools.combinations(range(1,self.num_mouse+1),2)):
            for i in list(itertools.product(range(len(col)),repeat=2)):
                sel_col = f"mouse{m[0]}_{col[i[0]]}_mouse{m[1]}_{col[i[1]]}_dist"
                m0_col_x = f"mouse{m[0]}_{col[i[0]]}_x"
                m0_col_y = f"mouse{m[0]}_{col[i[0]]}_y"
                m1_col_x = f"mouse{m[1]}_{col[i[1]]}_x"
                m1_col_y = f"mouse{m[1]}_{col[i[1]]}_y"
                df[f"{sel_col}"] = np.hypot(df[m0_col_x] - df[m1_col_x],df[m0_col_y] - df[m1_col_y])
        return df

    # Angle Metrics
    def ang_wrap(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def calc_heading(self, df, col=None):
        if col == None:
            col = self.col
        for m in range(1,self.num_mouse+1):
            for i in list(itertools.permutations(range(len(col)),2)):
                sel_col = f"mouse{m}_{col[i[0]]}_to_{col[i[1]]}"
                m_col0_x, m_col0_y = f"mouse{m}_{col[i[0]]}_x", f"mouse{m}_{col[i[0]]}_y"
                m_col1_x, m_col1_y = f"mouse{m}_{col[i[1]]}_x" , f"mouse{m}_{col[i[1]]}_y"

                th = np.arctan2(df[m_col0_y]-df[m_col1_y], df[m_col0_x]-df[m_col1_x])
                df[f"{sel_col}_sin"] = np.sin(th)
                df[f"{sel_col}_cos"] = np.cos(th)

                th_unw = np.unwrap(th.to_numpy())
                df[f"mouse{i}_angvel"] = np.gradient(th_unw) * self.fps
        return df

    def calc_pair_angle(self, df, col=None):
        if col == None:
            col = self.col
        for m in list(itertools.combinations(range(1,self.num_mouse+1),2)):
            for i in list(itertools.product(range(len(col)),repeat=2)):
                sel_col = f"mouse{m[0]}_{col[i[0]]}_mouse{m[1]}_{col[i[1]]}_dist"
                m0_col_x,m0_col_y = f"mouse{m[0]}_{col[i[0]]}_x", f"mouse{m[0]}_{col[i[0]]}_y"
                m1_col_x,m1_col_y = f"mouse{m[1]}_{col[i[1]]}_x",f"mouse{m[1]}_{col[i[1]]}_y"

                dx, dy = df[m1_col_x]-df[m0_col_x], df[m1_col_y]-df[m0_col_y]
                beta = np.arctan2(dy, dx)
                s, c = df.get(f"mouse{m[0]}_{col[i[0]]}_to_{col[i[1]]}_sin"), df.get(f"mouse{m[0]}_{col[i[0]]}_to_{col[i[1]]}_cos")
                if s is not None and c is not None:
                    theta_i = np.arctan2(s, c)
                    face = self.ang_wrap((theta_i-beta))
                    df[f"mouse{m[0]}_{col[i[0]]}_to_mouse{m[1]}_{col[i[1]]}_sinFace"] = np.sin(face)
                    df[f"mouse{m[0]}_{col[i[0]]}_to_mouse{m[1]}_{col[i[1]]}_cosFace"] = np.cos(face)
        return df