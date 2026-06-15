def generate_smooth_transition(start_pose, end_pose, duration_sec=1.0, fps=30):
    """
    Interpolate using a 5th-degree polynomial (Minimum Jerk / Smootherstep).
    Provides a smoother transition than Cosine because acceleration is zero at start and end.
    """
    if not start_pose:
        return [end_pose]
        
    steps = int(duration_sec * fps)
    if steps <= 0:
        return []

    transition_motions = []
    for i in range(1, steps + 1):
        # t goes from 0 to 1
        t = i / steps
        
        # Smootherstep formula (5th-degree polynomial): 6t^5 - 15t^4 + 10t^3
        # Creates a perfect S-curve for robot motion
        alpha = t * t * t * (t * (t * 6 - 15) + 10)
        
        interp_pose = {}
        for joint, target_angle in end_pose.items():
            start_angle = start_pose.get(joint, 0.0)
            interp_pose[joint] = start_angle + alpha * (target_angle - start_angle)
            
        transition_motions.append(interp_pose)
        
    return transition_motions


def generate_global_trajectory(execution_plan, neutral_pose, blend_duration=0.5, fps=30):
    """
    Combine all actions in the execution_plan into a single trajectory,
    applying smooth blending at crossover points.
    Removes excess frames outside the f2_start -> f1_cut region.
    """
    segments = [step['motions'][step['f2_start'] : step['f1_cut']] for step in execution_plan]
    
    if not segments:
        return []

    # 1. Create opening segment (from t=0 to t_start of the first action)
    t_start_0 = execution_plan[0]['t_start']
    f_start_0 = int(t_start_0 * fps)
    trans_len = int(0.6 * fps) # 0.6 seconds to transition from neutral to first pose
    
    trajectory = []
    if f_start_0 >= trans_len:
        # Hold neutral pose
        trajectory.extend([neutral_pose.copy() for _ in range(f_start_0 - trans_len)])
        # Smooth transition to the first pose
        trajectory.extend(generate_smooth_transition(neutral_pose, segments[0][0], 0.6, fps))
    else:
        if f_start_0 > 0:
            duration_sec = f_start_0 / fps
            trajectory.extend(generate_smooth_transition(neutral_pose, segments[0][0], duration_sec, fps))
        else:
            trajectory.append(segments[0][0].copy())
            
    # Append the rest of the first segment
    trajectory.extend(segments[0][1:])
    
    # 2. Append and smoothly blend the following segments
    W = int(blend_duration * fps)
    for i in range(1, len(segments)):
        seg = segments[i]
        if not seg:
            continue
            
        prev_step = execution_plan[i-1]
        curr_step = execution_plan[i]
        
        # Calculate the gap between previous and current action
        t_end_prev = prev_step['t_start'] + prev_step['f1_cut'] / float(fps)
        t_start_curr = curr_step['t_start'] + curr_step['f2_start'] / float(fps)
        
        gap_frames = int(round((t_start_curr - t_end_prev) * fps))
        
        if gap_frames > W:
            # Enough time to return to neutral and hold
            pose_end = trajectory[-1]
            pose_start = seg[0]
            
            trans_len = min(W, gap_frames // 2)
            trans_to_neutral = generate_smooth_transition(pose_end, neutral_pose, trans_len / float(fps), fps)
            trans_from_neutral = generate_smooth_transition(neutral_pose, pose_start, trans_len / float(fps), fps)
            
            hold_frames = gap_frames - 2 * trans_len
            
            trajectory.extend(trans_to_neutral)
            if hold_frames > 0:
                trajectory.extend([neutral_pose.copy() for _ in range(hold_frames)])
            trajectory.extend(trans_from_neutral)
            
            trajectory.extend(seg)
        else:
            # Gap is too short (<= W) or no gap (crossover)
            if gap_frames > 0:
                pose_end = trajectory[-1]
                trajectory.extend([pose_end.copy() for _ in range(gap_frames)])
                
            boundary_idx = len(trajectory)
            trajectory.extend(seg)
            
            # Apply smooth blending over the crossover boundary
            W_half = W // 2
            start_idx = max(0, boundary_idx - W_half)
            end_idx = min(len(trajectory), boundary_idx + W_half)
            
            if end_idx - start_idx > 1:
                pose_start = trajectory[start_idx]
                pose_end_blend = trajectory[end_idx - 1]
                transition = generate_smooth_transition(pose_start, pose_end_blend, (end_idx - start_idx) / float(fps), fps)
                for j in range(end_idx - start_idx):
                    trajectory[start_idx + j] = transition[j]
            
    return trajectory