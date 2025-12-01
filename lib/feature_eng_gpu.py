import cudf as pd
import cupy as cp
import itertools
import gc
import os
import numpy as np # For scalar math

# --- Helpers ---
def _scale(n_frames_at_30fps, fps, ref=30.0):
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    if n_frames_at_30fps == 0: return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _speed_gpu(cx, cy, fps):
    # Calculate Euclidean speed on GPU
    dx = cx.diff().fillna(0.0)
    dy = cy.diff().fillna(0.0)
    # FIX: Use **0.5 instead of .sqrt() for compatibility
    return (dx**2 + dy**2)**0.5 * float(fps)

def flatten_columns(df):
    """
    Flattens MultiIndex columns to string for Parquet compatibility.
    Example: ('Mouse1', 'speed') -> 'Mouse1_speed'
    """
    # Robust check: if nlevels > 1, it's a MultiIndex (works on cuDF and Pandas)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        # Try to pull columns to CPU list safely
        try:
            cols = df.columns.to_pandas().tolist()
        except AttributeError:
            # Fallback for older cuDF versions
            cols = list(df.columns)
            
        new_cols = ['_'.join(map(str, col)).strip() for col in cols]
        df.columns = new_cols
    return df

# --- Advanced Feature Helpers (GPU Ported) ---

def add_curvature_features_gpu(X, cx, cy, fps):
    """Trajectory curvature (GPU)."""
    # 1. Velocities
    vel_x = cx.diff()
    vel_y = cy.diff()
    # 2. Accelerations
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    # 3. Cross Product (2D)
    cross_prod = vel_x * acc_y - vel_y * acc_x
    # FIX: Use **0.5
    vel_mag = (vel_x**2 + vel_y**2)**0.5
    
    # Avoid div by zero
    curvature = cross_prod.abs() / (vel_mag**3 + 1e-6)

    for w in [30, 60]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()

    # 4. Turn Rate
    # Use CuPy for arctan2 as it's reliable with cuDF series values
    angle = pd.Series(cp.arctan2(vel_y.values, vel_x.values), index=X.index)
    angle_change = angle.diff().abs()
    
    w = 30
    ws = _scale(w, fps)
    X[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 6)).sum()
    return X

def add_multiscale_features_gpu(X, cx, cy, fps):
    """Multi-scale temporal features (GPU)."""
    speed = _speed_gpu(cx, cy, fps)

    scales = [10, 40, 160]
    for scale in scales:
        ws = _scale(scale, fps)
        X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
        X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2:
        top = X[f'sp_m{scales[0]}']
        bot = X[f'sp_m{scales[-1]}']
        X['sp_ratio'] = top / (bot + 1e-6)
    return X

def add_state_features_gpu(X, cx, cy, fps):
    """Behavioral state transitions (GPU)."""
    speed = _speed_gpu(cx, cy, fps)
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    # Binning on GPU
    # bins: [-inf, 0.5, 2.0, 5.0, inf] (scaled by fps)
    # 0: Stop, 1: Creep, 2: Walk, 3: Run
    
    # Manual masking is faster than pd.cut on GPU usually
    s = speed_ma
    # Initialize with 0
    states = s * 0.0 
    
    thresh1 = 0.5 * fps
    thresh2 = 2.0 * fps
    thresh3 = 5.0 * fps
    
    # Apply thresholds
    states = states.mask(s > thresh1, 1.0)
    states = states.mask(s > thresh2, 2.0)
    states = states.mask(s > thresh3, 3.0)
    
    for window in [60, 120]:
        ws = _scale(window, fps)
        for state_val in [0.0, 1.0, 2.0, 3.0]:
            # Binary mask
            is_state = (states == state_val).astype('float32')
            X[f's{int(state_val)}_{window}'] = is_state.rolling(ws, min_periods=max(1, ws//6)).mean()
            
        # Transitions
        state_changes = (states != states.shift(1)).astype('float32')
        X[f'trans_{window}'] = state_changes.rolling(ws, min_periods=max(1, ws//6)).sum()
        
    return X

def add_longrange_features_gpu(X, cx, cy, fps):
    """Long-range temporal features (GPU)."""
    for window in [120, 240]:
        ws = _scale(window, fps)
        # cuDF rolling mean
        X[f'x_ml{window}'] = cx.rolling(ws, min_periods=max(5, ws // 6)).mean()
        X[f'y_ml{window}'] = cy.rolling(ws, min_periods=max(5, ws // 6)).mean()

    # EWM
    for span in [60, 120]:
        s = _scale(span, fps)
        # FIX: Removed min_periods=1 argument as cuDF ewm() does not support it
        X[f'x_e{span}'] = cx.ewm(span=s).mean()
        X[f'y_e{span}'] = cy.ewm(span=s).mean()
        
    speed = _speed_gpu(cx, cy, fps)
    
    return X

def add_groom_microfeatures_gpu(X, single_mouse_df, fps):
    """Grooming detection (GPU)."""
    # Check parts existence in caller
    cx = single_mouse_df['body_center']['x']
    cy = single_mouse_df['body_center']['y']
    nx = single_mouse_df['nose']['x']
    ny = single_mouse_df['nose']['y']
    
    cs = _speed_gpu(cx, cy, fps)
    ns = _speed_gpu(nx, ny, fps)
    
    w30 = _scale(30, fps)
    
    # Decouple: Nose moving but body still
    ratio = (ns / (cs + 1e-3))
    # Clip logic
    ratio = ratio.where(ratio < 10, 10) # clip upper
    ratio = ratio.where(ratio > 0, 0)   # clip lower
    # FIX: Use .mean() instead of .median() because cuDF Rolling object lacks median support in some versions
    X['head_body_decouple'] = ratio.rolling(w30, min_periods=max(1, w30//3)).mean() 
    
    # Nose Radius variability
    dx = nx - cx
    dy = ny - cy
    # FIX: Use **0.5
    r = (dx**2 + dy**2)**0.5
    X['nose_rad_std'] = r.rolling(w30, min_periods=max(1, w30//3)).std().fillna(0)
    
    return X

# --- Main GPU Transforms ---

def transform_single_gpu(single_mouse, body_parts_tracked, fps):
    """
    Full-Feature GPU Single Transform.
    """
    avail_parts = single_mouse.columns.get_level_values(0).unique().tolist()
    
    # 1. Base Distances
    dist_cols = {
        f"{p1}+{p2}": (single_mouse[p1] - single_mouse[p2])**2
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in avail_parts and p2 in avail_parts
    }
    
    final_cols = {}
    for key, val_df in dist_cols.items():
        final_cols[key] = val_df['x'] + val_df['y']
        
    X = pd.DataFrame(final_cols)
    
    # 2. Speed / Elongation
    if all(p in avail_parts for p in ['ear_left', 'ear_right', 'tail_base']):
        lag = _scale(10, fps)
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(lag)
        
        for part in ['ear_left', 'ear_right']:
            diff = single_mouse[part] - shifted[part]
            X[f'sp_{part[:2]}'] = diff['x']**2 + diff['y']**2
            
    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    # 3. Geometry (Body Angle)
    if 'nose' in avail_parts and 'body_center' in avail_parts and 'tail_base' in avail_parts:
        v1_x = single_mouse['nose']['x'] - single_mouse['body_center']['x']
        v1_y = single_mouse['nose']['y'] - single_mouse['body_center']['y']
        v2_x = single_mouse['tail_base']['x'] - single_mouse['body_center']['x']
        v2_y = single_mouse['tail_base']['y'] - single_mouse['body_center']['y']
        
        dot = v1_x * v2_x + v1_y * v2_y
        # FIX: Use **0.5 instead of .sqrt()
        mag1 = (v1_x**2 + v1_y**2)**0.5
        mag2 = (v2_x**2 + v2_y**2)**0.5
        X['body_ang'] = dot / (mag1 * mag2 + 1e-6)

    # 4. Core Rolling Stats
    if 'body_center' in avail_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
        
        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            X[f'cx_m{w}'] = cx.rolling(ws, min_periods=1, center=True).mean()
            X[f'cy_m{w}'] = cy.rolling(ws, min_periods=1, center=True).mean()
            dx = cx.diff()
            dy = cy.diff()
            # FIX: Use **0.5 instead of .sqrt()
            disp = (dx**2 + dy**2)**0.5
            X[f'disp{w}'] = disp.rolling(ws, min_periods=1).sum()
            
            # Activity (Variance of velocity)
            # FIX: Use **0.5 instead of .sqrt()
            X[f'act{w}'] = (dx**2 + dy**2).rolling(ws, min_periods=1).var()**0.5

        # 5. CALL ADVANCED HELPERS
        # Only call if we have body_center (which we do inside this block)
        X = add_curvature_features_gpu(X, cx, cy, fps)
        X = add_multiscale_features_gpu(X, cx, cy, fps)
        X = add_state_features_gpu(X, cx, cy, fps)
        X = add_longrange_features_gpu(X, cx, cy, fps)
        
        if 'nose' in avail_parts:
            X = add_groom_microfeatures_gpu(X, single_mouse, fps)

    # 6. Nose-Tail Dynamics
    if 'nose' in avail_parts and 'tail_base' in avail_parts:
        nt_dx = single_mouse['nose']['x'] - single_mouse['tail_base']['x']
        nt_dy = single_mouse['nose']['y'] - single_mouse['tail_base']['y']
        # FIX: Use **0.5 instead of .sqrt()
        nt_dist = (nt_dx**2 + nt_dy**2)**0.5
        
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nt_lg{lag}'] = nt_dist.shift(l)
            X[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l)

    return X.astype('float32')

def transform_pair_chunked_gpu(mouse_pair, body_parts_tracked, fps, output_dir, pair_name):
    """
    GPU-Accelerated Chunked Pair Transform.
    """
    files_created = []
    df_A = mouse_pair['A']
    df_B = mouse_pair['B']
    
    # Metadata extraction on CPU side via .tolist()
    avail_A = df_A.columns.get_level_values(0).unique().tolist()
    avail_B = df_B.columns.get_level_values(0).unique().tolist()

    # --- Stage 1: Distances ---
    print(f"  > GPU Stage 1: Distances ({pair_name})")
    
    dist_data = {}
    for p1, p2 in itertools.product(body_parts_tracked, repeat=2):
        if p1 in avail_A and p2 in avail_B:
            dx = df_A[p1]['x'] - df_B[p2]['x']
            dy = df_A[p1]['y'] - df_B[p2]['y']
            dist_data[f"12+{p1}+{p2}"] = dx**2 + dy**2

    df_dist = pd.DataFrame(dist_data, dtype='float32')
    
    fname_dist = os.path.join(output_dir, f"{pair_name}_stage1_dist.parquet")
    df_dist.columns = pd.MultiIndex.from_product([[pair_name], df_dist.columns], names=['pair_id', 'feature'])
    # FLATTEN BEFORE SAVE
    df_dist = flatten_columns(df_dist)
    df_dist.to_parquet(fname_dist)
    files_created.append(fname_dist)
    
    del df_dist, dist_data
    cp.get_default_memory_pool().free_all_blocks()

    # --- Stage 2: Velocities ---
    print(f"  > GPU Stage 2: Velocities ({pair_name})")
    df_vel = pd.DataFrame(index=df_A.index, dtype='float32')
    
    if 'ear_left' in avail_A and 'ear_left' in avail_B:
        lag = _scale(10, fps)
        shA_x = df_A['ear_left']['x'].shift(lag)
        shA_y = df_A['ear_left']['y'].shift(lag)
        shB_x = df_B['ear_left']['x'].shift(lag)
        shB_y = df_B['ear_left']['y'].shift(lag)
        
        df_vel['sp_A'] = (df_A['ear_left']['x'] - shA_x)**2 + (df_A['ear_left']['y'] - shA_y)**2
        df_vel['sp_B'] = (df_B['ear_left']['x'] - shB_x)**2 + (df_B['ear_left']['y'] - shB_y)**2
        df_vel['sp_AB'] = (df_A['ear_left']['x'] - shB_x)**2 + (df_A['ear_left']['y'] - shB_y)**2

    # Approach
    if 'nose' in avail_A and 'nose' in avail_B:
         # Use vectorized math, avoid .values if possible to keep series
         dist_now = (df_A['nose']['x'] - df_B['nose']['x'])**2 + (df_A['nose']['y'] - df_B['nose']['y'])**2
         lag = _scale(10, fps)
         
         shA_x = df_A['nose']['x'].shift(lag)
         shA_y = df_A['nose']['y'].shift(lag)
         shB_x = df_B['nose']['x'].shift(lag)
         shB_y = df_B['nose']['y'].shift(lag)
         
         dist_past = (shA_x - shB_x)**2 + (shA_y - shB_y)**2
         df_vel['appr'] = dist_now - dist_past

    if not df_vel.empty:
        fname_vel = os.path.join(output_dir, f"{pair_name}_stage2_vel.parquet")
        df_vel.columns = pd.MultiIndex.from_product([[pair_name], df_vel.columns], names=['pair_id', 'feature'])
        # FLATTEN
        df_vel = flatten_columns(df_vel)
        df_vel.to_parquet(fname_vel)
        files_created.append(fname_vel)
    
    del df_vel
    cp.get_default_memory_pool().free_all_blocks()

    # --- Stage 3: Geometry (Orientation) ---
    print(f"  > GPU Stage 3: Geometry ({pair_name})")
    df_geo = pd.DataFrame(index=df_A.index, dtype='float32')

    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A_x = df_A['nose']['x'] - df_A['tail_base']['x']
        dir_A_y = df_A['nose']['y'] - df_A['tail_base']['y']
        
        dir_B_x = df_B['nose']['x'] - df_B['tail_base']['x']
        dir_B_y = df_B['nose']['y'] - df_B['tail_base']['y']
        
        dot = dir_A_x * dir_B_x + dir_A_y * dir_B_y
        # FIX: Use **0.5
        mag_A = (dir_A_x**2 + dir_A_y**2)**0.5
        mag_B = (dir_B_x**2 + dir_B_y**2)**0.5
        
        df_geo['rel_ori'] = dot / (mag_A * mag_B + 1e-6)

    if not df_geo.empty:
        fname_geo = os.path.join(output_dir, f"{pair_name}_stage3_geo.parquet")
        df_geo.columns = pd.MultiIndex.from_product([[pair_name], df_geo.columns], names=['pair_id', 'feature'])
        # FLATTEN
        df_geo = flatten_columns(df_geo)
        df_geo.to_parquet(fname_geo)
        files_created.append(fname_geo)
        
    del df_geo
    cp.get_default_memory_pool().free_all_blocks()

    # --- Stage 4: Dynamics (Rolling) ---
    print(f"  > GPU Stage 4: Dynamics ({pair_name})")
    if 'body_center' in avail_A and 'body_center' in avail_B:
        dx = df_A['body_center']['x'] - df_B['body_center']['x']
        dy = df_A['body_center']['y'] - df_B['body_center']['y']
        cd_sq = dx**2 + dy**2
        
        df_dyn = pd.DataFrame(index=df_A.index, dtype='float32')
        
        for w in [5, 30, 60]:
            ws = _scale(w, fps)
            df_dyn[f'd_m{w}'] = cd_sq.rolling(ws, min_periods=1, center=True).mean()
            df_dyn[f'd_s{w}'] = cd_sq.rolling(ws, min_periods=1, center=True).std()
            
        fname_dyn = os.path.join(output_dir, f"{pair_name}_stage4_dyn.parquet")
        df_dyn.columns = pd.MultiIndex.from_product([[pair_name], df_dyn.columns], names=['pair_id', 'feature'])
        # FLATTEN
        df_dyn = flatten_columns(df_dyn)
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

    # FIX: Extract metadata from Pandas BEFORE deleting it.
    mice_ids = df_wide_pandas.columns.get_level_values(0).unique().tolist()
    available_parts = df_wide_pandas.columns.get_level_values(1).unique().tolist()

    print("Moving data to GPU...")
    # Convert Pandas -> cuDF
    df_gpu = pd.from_pandas(df_wide_pandas).astype('float32')
    
    # Free CPU memory
    del df_wide_pandas
    gc.collect()
    
    saved_files = []

    # 1. Single
    for mouse in mice_ids:
        print(f"GPU Processing Mouse: {mouse}")
        single_mouse_gpu = df_gpu[mouse]
        
        feats = transform_single_gpu(single_mouse_gpu, available_parts, fps)
        
        feats.columns = pd.MultiIndex.from_product([[mouse], feats.columns], names=['mouse_id', 'feature'])
        
        # FIX: FLATTEN MultiIndex to String for Parquet
        feats = flatten_columns(feats)
        
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
            
    print("GPU Generation Complete.")
    return saved_files