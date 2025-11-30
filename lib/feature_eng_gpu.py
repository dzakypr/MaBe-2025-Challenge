import cudf as pd
import cupy as cp
import numpy as np
import itertools
import gc
import os

def safe_rolling(series, window, func, min_periods=None):
    """Safe rolling operation with NaN handling"""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def _scale(n_frames_at_30fps, fps, ref=30.0):
    """Scale a frame count defined at 30 fps to the current video's fps."""
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    """Signed version of _scale for forward/backward shifts (keeps at least 1 frame when |n|>=1)."""
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

def _speed(cx: pd.Series, cy: pd.Series, fps: float) -> pd.Series:
    return np.hypot(cx.diff(), cy.diff()).fillna(0.0) * float(fps)

def _roll_future_mean(s: pd.Series, w: int, min_p: int = 1) -> pd.Series:
    # mean over [t, t+w-1]
    return s.iloc[::-1].rolling(w, min_periods=min_p).mean().iloc[::-1]

def _roll_future_var(s: pd.Series, w: int, min_p: int = 2) -> pd.Series:
    # var over [t, t+w-1]
    return s.iloc[::-1].rolling(w, min_periods=min_p).var().iloc[::-1]


def add_curvature_features(X, center_x, center_y, fps):
    """Trajectory curvature (window lengths scaled by fps)."""
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)  # invariant to time scaling

    for w in [30, 60]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    w = 30
    ws = _scale(w, fps)
    X[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 6)).sum()

    return X

def add_multiscale_features(X, center_x, center_y, fps):
    """Multi-scale temporal features (speed in cm/s; windows scaled by fps)."""
    # displacement per frame is already in cm (pix normalized earlier); convert to cm/s
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [10, 40, 160]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)

    return X

def add_state_features(X, center_x, center_y, fps):
    """Behavioral state transitions; bins adjusted so semantics are fps-invariant."""
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        # Original bins (cm/frame): [-inf, 0.5, 2.0, 5.0, inf]
        # Convert to cm/s by multiplying by fps to keep thresholds consistent across fps.
        bins = [-np.inf, 0.5 * fps, 2.0 * fps, 5.0 * fps, np.inf]
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        for window in [60, 120]:
            ws = _scale(window, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    X[f's{state}_{window}'] = (
                        (speed_states == state).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 6)).mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f'trans_{window}'] = state_changes.rolling(ws, min_periods=max(1, ws // 6)).sum()
    except Exception:
        pass

    return X

def add_longrange_features(X, center_x, center_y, fps):
    """Long-range temporal features (windows & spans scaled by fps)."""
    for window in [120, 240]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    # EWM spans also interpreted in frames
    for span in [60, 120]:
        s = _scale(span, fps)
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    for window in [60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            X[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    return X

def add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base: int = 180, colname: str = "path_cum180"):
    L = max(1, _scale(horizon_frames_base, fps))  # frames
    # step length (cm per frame since coords are cm)
    step = np.hypot(cx.diff(), cy.diff())
    # centered rolling sum over ~2L+1 frames (acausal)
    path = step.rolling(2*L + 1, min_periods=max(5, L//6), center=True).sum()
    X[colname] = path.fillna(0.0).astype(np.float32)
    return X


def add_groom_microfeatures(X, df, fps):
    parts = df.columns.get_level_values(0)
    if 'body_center' not in parts or 'nose' not in parts:
        return X

    cx = df['body_center']['x']; cy = df['body_center']['y']
    nx = df['nose']['x']; ny = df['nose']['y']

    cs = (np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)).fillna(0)
    ns = (np.sqrt(nx.diff()**2 + ny.diff()**2) * float(fps)).fillna(0)

    w30 = _scale(30, fps)
    X['head_body_decouple'] = (ns / (cs + 1e-3)).clip(0, 10).rolling(w30, min_periods=max(1, w30//3)).median()

    r = np.sqrt((nx - cx)**2 + (ny - cy)**2)
    X['nose_rad_std'] = r.rolling(w30, min_periods=max(1, w30//3)).std().fillna(0)

    if 'tail_base' in parts:
        ang = np.arctan2(df['nose']['y']-df['tail_base']['y'], df['nose']['x']-df['tail_base']['x'])
        dang = np.abs(ang.diff()).fillna(0)
        X['head_orient_jitter'] = dang.rolling(w30, min_periods=max(1, w30//3)).mean()

    return X


def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    """Social interaction features (windows scaled by fps)."""
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    # per-frame velocities (cm/frame)
    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()

    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)

    for window in [30, 60]:
        ws = _scale(window, fps)
        X[f'A_ld{window}'] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        X[f'B_ld{window}'] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    approach = -rel_dist.diff()  # decreasing distance => positive approach
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    X[f'chase_{w}'] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = np.sqrt(A_vx**2 + A_vy**2)
        B_sp = np.sqrt(B_vx**2 + B_vy**2)
        X[f'sp_cor{window}'] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    return X

# ===============================================================
# 1) Past–vs–Future speed asymmetry (acausal, continuous)
#    Δv = mean_future(speed) - mean_past(speed)
# ===============================================================
def add_speed_asymmetry_future_past_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float,
    horizon_base: int = 30, agg: str = "mean"
) -> pd.DataFrame:
    w = max(3, _scale(horizon_base, fps))
    v = _speed(cx, cy, fps)
    if agg == "median":
        v_past = v.rolling(w, min_periods=max(3, w//4), center=False).median()
        v_fut  = v.iloc[::-1].rolling(w, min_periods=max(3, w//4)).median().iloc[::-1]
    else:
        v_past = v.rolling(w, min_periods=max(3, w//4), center=False).mean()
        v_fut  = _roll_future_mean(v, w, min_p=max(3, w//4))
    X["spd_asym_1s"] = (v_fut - v_past).fillna(0.0)
    return X

# ===============================================================
# 2) Distribution shift (future vs past) via symmetric KL of
#    Gaussian fits on speed
# ===============================================================
def add_gauss_shift_speed_future_past_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float,
    window_base: int = 30, eps: float = 1e-6
) -> pd.DataFrame:
    w = max(5, _scale(window_base, fps))
    v = _speed(cx, cy, fps)

    mu_p = v.rolling(w, min_periods=max(3, w//4)).mean()
    va_p = v.rolling(w, min_periods=max(3, w//4)).var().clip(lower=eps)

    mu_f = _roll_future_mean(v, w, min_p=max(3, w//4))
    va_f = _roll_future_var(v, w, min_p=max(3, w//4)).clip(lower=eps)

    # KL(Np||Nf) + KL(Nf||Np)
    kl_pf = 0.5 * ((va_p/va_f) + ((mu_f - mu_p)**2)/va_f - 1.0 + np.log(va_f/va_p))
    kl_fp = 0.5 * ((va_f/va_p) + ((mu_p - mu_f)**2)/va_p - 1.0 + np.log(va_p/va_f))
    X["spd_symkl_1s"] = (kl_pf + kl_fp).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X

def transform_single_gpu(single_mouse, body_parts_tracked, fps):
    """
    GPU-Accelerated version of transform_single.
    Expects 'single_mouse' to be a cuDF DataFrame.
    """
    # cuDF doesn't support level access by name in some versions, 
    # so we assume standard columns or extract via standard indexing if needed.
    # Here we assume the input is already level-dropped or we access by column tuples.
    
    avail_parts = single_mouse.columns.get_level_values(0).unique().to_pandas().tolist()
    
    # Base Distances
    # Note: cuDF dictionary comprehension works similar to pandas
    dist_cols = {
        f"{p1}+{p2}": (single_mouse[p1] - single_mouse[p2])**2
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in avail_parts and p2 in avail_parts
    }
    
    # We sum x^2 + y^2 manually because cuDF sum(axis=1) can be slower or tricky with MultiIndex
    # But here p1 is (x, y), so single_mouse[p1] is a DataFrame with x, y cols
    # It's faster to do column math explicitly
    final_cols = {}
    for key, val_df in dist_cols.items():
        # val_df has columns x, y (squared differences)
        final_cols[key] = val_df['x'] + val_df['y']
        
    X = pd.DataFrame(final_cols)
    
    # Speed (Ear/Tail lag)
    if all(p in avail_parts for p in ['ear_left', 'ear_right', 'tail_base']):
        lag = _scale(10, fps)
        # Shift on GPU
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(lag)
        
        # Calculate speeds manually using cupy or cudf series math
        # (Using pure cudf series math is usually safest for index alignment)
        for part in ['ear_left', 'ear_right']:
            diff = single_mouse[part] - shifted[part]
            X[f'sp_{part[:2]}'] = diff['x']**2 + diff['y']**2
            
    # Geometry
    if 'nose' in avail_parts and 'body_center' in avail_parts and 'tail_base' in avail_parts:
        # Create vectors
        v1_x = single_mouse['nose']['x'] - single_mouse['body_center']['x']
        v1_y = single_mouse['nose']['y'] - single_mouse['body_center']['y']
        v2_x = single_mouse['tail_base']['x'] - single_mouse['body_center']['x']
        v2_y = single_mouse['tail_base']['y'] - single_mouse['body_center']['y']
        
        dot = v1_x * v2_x + v1_y * v2_y
        mag1 = (v1_x**2 + v1_y**2).sqrt()
        mag2 = (v2_x**2 + v2_y**2).sqrt()
        X['body_ang'] = dot / (mag1 * mag2 + 1e-6)

    # Rolling Windows (cuDF supports rolling)
    if 'body_center' in avail_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
        
        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            # cuDF rolling
            X[f'cx_m{w}'] = cx.rolling(ws, min_periods=1, center=True).mean()
            X[f'cy_m{w}'] = cy.rolling(ws, min_periods=1, center=True).mean()
            # Speed/Disp
            dx = cx.diff()
            dy = cy.diff()
            disp = (dx**2 + dy**2).sqrt()
            X[f'disp{w}'] = disp.rolling(ws, min_periods=1).sum()

    return X.astype('float32')

def transform_pair_chunked_gpu(mouse_pair, body_parts_tracked, fps, output_dir, pair_name):
    """
    GPU-Accelerated Chunked Pair Transform.
    """
    files_created = []
    df_A = mouse_pair['A']
    df_B = mouse_pair['B']
    
    # Helper to convert MultiIndex to list for checking
    avail_A = df_A.columns.get_level_values(0).unique().to_pandas().tolist()
    avail_B = df_B.columns.get_level_values(0).unique().to_pandas().tolist()

    # --- Stage 1: Distances ---
    # print(f"  > GPU Stage 1: Distances ({pair_name})")
    
    # Calculate squared diffs
    dist_data = {}
    for p1, p2 in itertools.product(body_parts_tracked, repeat=2):
        if p1 in avail_A and p2 in avail_B:
            # Vectorized subtraction on GPU
            dx = df_A[p1]['x'] - df_B[p2]['x']
            dy = df_A[p1]['y'] - df_B[p2]['y']
            dist_data[f"12+{p1}+{p2}"] = dx**2 + dy**2

    df_dist = pd.DataFrame(dist_data, dtype='float32')
    
    # Saving to Parquet (cuDF supports this)
    fname_dist = os.path.join(output_dir, f"{pair_name}_dist.parquet")
    
    # Add index context (Creating MultiIndex in cuDF)
    # Note: cuDF MultiIndex creation can be strict.
    # We rename columns directly if needed or use from_product if supported.
    df_dist.columns = pd.MultiIndex.from_product([[pair_name], df_dist.columns], names=['pair_id', 'feature'])
    df_dist.to_parquet(fname_dist)
    files_created.append(fname_dist)
    
    del df_dist, dist_data
    # Force GPU memory release
    cp.get_default_memory_pool().free_all_blocks()

    # --- Stage 2: Velocities ---
    # print(f"  > GPU Stage 2: Velocities ({pair_name})")
    df_vel = pd.DataFrame(index=df_A.index, dtype='float32')
    
    if 'ear_left' in avail_A and 'ear_left' in avail_B:
        lag = _scale(10, fps)
        # Shift
        shA_x = df_A['ear_left']['x'].shift(lag)
        shA_y = df_A['ear_left']['y'].shift(lag)
        shB_x = df_B['ear_left']['x'].shift(lag)
        shB_y = df_B['ear_left']['y'].shift(lag)
        
        df_vel['sp_A'] = (df_A['ear_left']['x'] - shA_x)**2 + (df_A['ear_left']['y'] - shA_y)**2
        df_vel['sp_B'] = (df_B['ear_left']['x'] - shB_x)**2 + (df_B['ear_left']['y'] - shB_y)**2
        df_vel['sp_AB'] = (df_A['ear_left']['x'] - shB_x)**2 + (df_A['ear_left']['y'] - shB_y)**2

    if 'nose' in avail_A and 'nose' in avail_B:
        dist_now = np.square(df_A['nose'] - df_B['nose']).sum(axis=1, skipna=False)
        lag = _scale(10, fps)
        shA_n = df_A['nose'].shift(lag)
        shB_n = df_B['nose'].shift(lag)
        dist_past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        df_vel['appr'] = dist_now - dist_past

    if not df_vel.empty:
        fname_vel = os.path.join(output_dir, f"{pair_name}_vel.parquet")
        df_vel.columns = pd.MultiIndex.from_product([[pair_name], df_vel.columns], names=['pair_id', 'feature'])
        df_vel.to_parquet(fname_vel)
        files_created.append(fname_vel)
    
    del df_vel
    cp.get_default_memory_pool().free_all_blocks()

    df_geo = pd.DataFrame(index=df_A.index, dtype=np.float32)
    # Relative orientation
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = df_A['nose'] - df_A['tail_base']
        dir_B = df_B['nose'] - df_B['tail_base']
        df_geo['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)

    # Distance bins (cm; unchanged by fps)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = np.sqrt((df_A['body_center']['x'] - df_B['body_center']['x'])**2 +
                     (df_A['body_center']['y'] - df_B['body_center']['y'])**2)
        df_geo['v_cls'] = (cd < 5.0).astype(float)
        df_geo['cls']   = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        df_geo['med']   = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        df_geo['far']   = (cd >= 30.0).astype(float)
        
    if not df_geo.empty:
        fname_geo = os.path.join(output_dir, f"{pair_name}_stage3_geo.parquet")
        df_geo.columns = pd.MultiIndex.from_product([[pair_name], df_geo.columns], names=['pair_id', 'feature'])
        df_geo.to_parquet(fname_geo)
        files_created.append(fname_geo)

    del df_geo
    cp.get_default_memory_pool().free_all_blocks()

    # --- Stage 4: Dynamics (Rolling) ---
    # print(f"  > GPU Stage 4: Dynamics ({pair_name})")
    if 'body_center' in avail_A and 'body_center' in avail_B:
        # Re-calc distance sq
        dx = df_A['body_center']['x'] - df_B['body_center']['x']
        dy = df_A['body_center']['y'] - df_B['body_center']['y']
        cd_sq = dx**2 + dy**2
        
        df_dyn = pd.DataFrame(index=df_A.index, dtype='float32')
        
        for w in [5, 30, 60]:
            ws = _scale(w, fps)
            # Rolling on GPU
            df_dyn[f'd_m{w}'] = cd_sq.rolling(ws, min_periods=1, center=True).mean()
            df_dyn[f'd_s{w}'] = cd_sq.rolling(ws, min_periods=1, center=True).std()
            
        fname_dyn = os.path.join(output_dir, f"{pair_name}_dyn.parquet")
        df_dyn.columns = pd.MultiIndex.from_product([[pair_name], df_dyn.columns], names=['pair_id', 'feature'])
        df_dyn.to_parquet(fname_dyn)
        files_created.append(fname_dyn)
        
        del df_dyn, cd_sq
        cp.get_default_memory_pool().free_all_blocks()

    return files_created

def generate_features_gpu(df_wide_pandas, fps, output_dir, body_parts_map=None):
    """
    Main entry point. Takes PANDAS DataFrame, converts to GPU, runs, saves to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if body_parts_map:
        df_wide_pandas = df_wide_pandas.rename(columns=body_parts_map, level='bodypart')

    # print("Moving data to GPU...")
    # Convert Pandas -> cuDF
    df_gpu = pd.from_pandas(df_wide_pandas).astype('float32')
    
    # Free CPU memory
    del df_wide_pandas
    gc.collect()
    
    mice_ids = df_gpu.columns.get_level_values(0).unique().to_pandas().tolist()
    available_parts = df_gpu.columns.get_level_values(1).unique().to_pandas().tolist()
    
    saved_files = []

    # 1. Single
    for mouse in mice_ids:
        # print(f"GPU Processing Mouse: {mouse}")
        # Slicing in cuDF works similar to pandas
        single_mouse_gpu = df_gpu[mouse] # This is a view/copy
        
        feats = transform_single_gpu(single_mouse_gpu, available_parts, fps)
        
        feats.columns = pd.MultiIndex.from_product([[mouse], feats.columns], names=['mouse_id', 'feature'])
        fname = os.path.join(output_dir, f"feats_single_{mouse}.parquet")
        feats.to_parquet(fname)
        saved_files.append(fname)
        
        del feats, single_mouse_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # 2. Pairs
    if len(mice_ids) > 1:
        for mouse_A, mouse_B in itertools.combinations(mice_ids, 2):
            pair_name = f"{mouse_A}_vs_{mouse_B}"
            pair_data = {
                'A': df_gpu[mouse_A],
                'B': df_gpu[mouse_B]
            }
            
            pair_files = transform_pair_chunked_gpu(pair_data, available_parts, fps, output_dir, pair_name)
            saved_files.extend(pair_files)
            
            del pair_data
            cp.get_default_memory_pool().free_all_blocks()
            
    # print("GPU Generation Complete.")
    return saved_files