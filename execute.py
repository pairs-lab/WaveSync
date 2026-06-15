import json
import os
import time
import sys
import pickle
import random
import threading
import numpy as np
from scipy.signal import savgol_filter
from scipy.io import wavfile
import librosa
from gtts import gTTS
import os
os.environ["TQDM_DISABLE"] = "1"

# Disable tqdm progress bars globally (safe patching)
try:
    import tqdm
    orig_init = tqdm.tqdm.__init__
    def new_init(self, *args, **kwargs):
        kwargs['disable'] = True
        orig_init(self, *args, **kwargs)
    tqdm.tqdm.__init__ = new_init
except ImportError:
    pass

# Add directories to path
sys.path.append("./src")
sys.path.append("./AI_module")

# --- CORE IMPORTS ---
from src.core.siw_simple import load_and_merge_mock, generate_full_audio, get_real_timestamps, match_whisper_to_importance, create_siw_map
from src.core.kinematic_wavefront import optimize_kinematic_plan, KinematicOptimizer, ParabolicPulse
from src.core.interpolation import generate_smooth_transition, generate_global_trajectory
from src.core.dmp import ExpressiveDMP, get_emotion_params

# --- UTILS & SIMULATION ---
from src.audio.audio_utils import play_audio_bg
from src.audio.audio_aligner import AudioAligner
from src.simulation.pybullet_env import PyBulletEnv


def load_action_library(sequence):
    action_library = {}
    for seg_data in sequence:
        action_name = seg_data.get('body_language', {}).get('action', 'talk_motion')
        if action_name in action_library:
            continue
            
        motion_file = f"motion_data/Robot_{action_name}.pkl"
        if os.path.exists(motion_file):
            with open(motion_file, 'rb') as f:
                raw_motions = pickle.load(f)
            action_library[action_name] = {'duration': len(raw_motions) / 30.0, 'raw_motions': raw_motions}
        else:
            print(f"Warning: Motion file not found: {motion_file}")
    return action_library


def generate_execution_plan(segments, action_library, p_times, t_starts):
    execution_plan = []
    
    for i, seg in enumerate(segments):
        action = seg['action']
        raw_motions = action_library[action]['raw_motions']
        
        joint_names = sorted(raw_motions[0].keys())
        matrix_data = np.array([[frame.get(k, 0.0) for k in joint_names] for frame in raw_motions])
        
        filter_window = min(50, len(raw_motions))
        if filter_window % 2 == 0: filter_window -= 1
        if filter_window > 2: 
            matrix_data = savgol_filter(matrix_data, filter_window, 2, axis=0)

        p_exa, _, p_ant, p_rand, spatial_offset = get_emotion_params(seg['emotion'])
        std_devs = np.std(matrix_data, axis=0)
        matrix_data[:, std_devs > 0.01] += spatial_offset

        dmp = ExpressiveDMP(n_bfs=80)
        dmp.train(matrix_data, dt=1/30.0, joint_names=joint_names)
        expressive_motions = dmp.generate(p_exa=random.uniform(0.8, 1.2) * p_exa, p_time=p_times[i], p_ant=p_ant, p_rand=p_rand)

        execution_plan.append({
            'action': action, 't_start': t_starts[i], 'motions': expressive_motions,
            'has_crossover': False, 'f1_cut': len(expressive_motions), 'f2_start': 0
        })
        
    return execution_plan


def calculate_pre_delays(execution_plan, sequence, audio_files, p_times, t_starts, action_library, aligner, sr=16000):
    for i in range(len(execution_plan)):
        step = execution_plan[i]
        y_seg, _ = librosa.load(audio_files[i], sr=sr)
        T_aud = len(y_seg) / sr
        T_act = action_library[step['action']]['duration'] * p_times[i]

        if step['action'] == "talk_motion":
            pre_delay_sec = 0.0
        else:
            emphasized = sequence[i].get('emphasized_word', '').lower()
            word_peak_rel = T_aud / 2.0 
            if emphasized:
                words, _ = get_real_timestamps(audio_files[i], aligner=aligner)
                if words:
                    for w in words:
                        if emphasized in w['word'].lower():
                            word_peak_rel = w['start'] + (w['end'] - w['start']) / 2.0
                            break

            # Calculate pre-delay so emphasized word hits the mechanical peak
            t_peak_act = T_act / 2.0
            pre_delay_sec = max(0.0, t_peak_act - word_peak_rel)

        # Prevent Voice overlap
        if i < len(execution_plan) - 1:
            next_step = execution_plan[i+1]
            voice_end_time = step['t_start'] + pre_delay_sec + T_aud
            if next_step['t_start'] < voice_end_time + 0.05:
                shift = (voice_end_time + 0.05) - next_step['t_start']
                next_step['t_start'] += shift
                t_starts[i+1] += shift 

        step['y_seg'] = y_seg
        step['T_aud'] = T_aud
        step['T_act'] = T_act
        step['pre_delay_sec'] = pre_delay_sec


def recalculate_crossovers(execution_plan, action_library, p_times, dummy_opt):
    for i in range(len(execution_plan) - 1):
        p1 = ParabolicPulse(action_library[execution_plan[i]['action']]['duration'], p_times[i])
        p2 = ParabolicPulse(action_library[execution_plan[i+1]['action']]['duration'], p_times[i+1])
        crossover = dummy_opt._find_crossover_point(p1, execution_plan[i]['t_start'], p2, execution_plan[i+1]['t_start'])
        if crossover:
            t_cross = crossover[0]
            execution_plan[i]['has_crossover'] = True
            execution_plan[i]['f1_cut'] = max(0, int((t_cross - execution_plan[i]['t_start']) * 30))
            execution_plan[i+1]['f2_start'] = max(0, int((t_cross - execution_plan[i+1]['t_start']) * 30))


def process_offline_audio(execution_plan, sr=16000):
    # Calculate the end time of the last action's audio to determine array size
    last_step = execution_plan[-1]
    total_duration = last_step['t_start'] + last_step['pre_delay_sec'] + last_step['T_aud'] + 5.0
    master_audio_length = int(total_duration * sr)
    master_audio = np.zeros(master_audio_length, dtype=np.float32)

    for i, step in enumerate(execution_plan):
        pre_delay_sec = step['pre_delay_sec']
        y_seg = step['y_seg']
        
        # Absolute start time in the global timeline
        start_time = step['t_start'] + pre_delay_sec
        start_idx = int(start_time * sr)
        end_idx = start_idx + len(y_seg)
        
        if end_idx > len(master_audio): 
            master_audio = np.pad(master_audio, (0, end_idx - len(master_audio)))
            
        # Add audio segment un-scaled and un-truncated!
        master_audio[start_idx:end_idx] += y_seg

    master_path = "out/master_audio.wav"
    wavfile.write(master_path, sr, master_audio)
    return [master_path]


def run_simulation(env, execution_plan, trajectory_frames):
    time.sleep(1)
    fps = 30
    frame_time = 1.0 / fps
    
    audio_played = False
    
    for frame_idx, pose in enumerate(trajectory_frames):
        loop_start = time.time()
        
        # Play master audio at the first frame
        if not audio_played:
            threading.Thread(target=play_audio_bg, args=("out/master_audio.wav",)).start()
            audio_played = True
                
        # Play the frame
        for joi_name, angle in pose.items():
            if joi_name in env.joint_name_to_id:
                joi_index = env.joint_name_to_id[joi_name]
                import pybullet as pb
                pb.resetJointState(env.robot_id, joi_index, angle)
                env.current_pose[joi_name] = angle
        import pybullet as pb
        pb.stepSimulation()
        
        # Maintain FPS
        elapsed = time.time() - loop_start
        if frame_time > elapsed:
            time.sleep(frame_time - elapsed)
            
    # Transition to neutral at the end
    env.play_frames(generate_smooth_transition(env.current_pose, env.neutral_pose, 0.6, 30))
    time.sleep(2)
    env.disconnect()

def cleanup(files):
    for f in files:
        if os.path.exists(f): 
            os.remove(f)


def resolve_scene_path(scene_arg):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    if not scene_arg:
        return os.path.join(base_dir, 'mock_llm_response.json')
    if os.path.exists(scene_arg):
        return os.path.abspath(scene_arg)
    if scene_arg.isdigit():
        path = os.path.join(base_dir, f'scene/scene{scene_arg}.json')
        if os.path.exists(path):
            return path
    if scene_arg.startswith("scene") and not scene_arg.endswith(".json"):
        path = os.path.join(base_dir, f'scene/{scene_arg}.json')
        if os.path.exists(path):
            return path
    path = os.path.join(base_dir, f'scene/{scene_arg}')
    if os.path.exists(path):
        return path
    if not scene_arg.endswith(".json"):
        path = os.path.join(base_dir, f'scene/{scene_arg}.json')
        if os.path.exists(path):
            return path
    return scene_arg


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run HRI Pipeline with scene file selection")
    base_dir = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--scene", "-s", type=str, default=os.path.join(base_dir, "mock_llm_response.json"), help="Path, name, or number of the scene file to run")
    parser.add_argument("--record", "-r", type=str, default="", help="Record output to this MP4 file")
    args = parser.parse_args()
    
    mock_path = resolve_scene_path(args.scene)
    
    aligner = AudioAligner()
    
    result = load_and_merge_mock(mock_path)
    if len(result) != 4 or not result[0]:
        print("Error: Could not load JSON mock file.")
        return
    full_text, word_importance_map, segment_boundaries, all_words = result

    temp_full_audio, duration = generate_full_audio(full_text)
    whisper_words, duration = get_real_timestamps(temp_full_audio, aligner=aligner)
    if not whisper_words:
        print("Error: Whisper could not recognize audio. Please check language config.")
        return
        
    word_importance_list, emphasized_words = match_whisper_to_importance(whisper_words, word_importance_map, all_words)
    siw_map = create_siw_map(whisper_words, word_importance_list, duration)
    
    with open(mock_path, 'r', encoding='utf-8') as f:
        sequence = json.load(f).get('sequence', [])

    action_library = load_action_library(sequence)
    
    segments = []
    audio_files = []
    os.makedirs("out", exist_ok=True)
    
    for idx, seg_data in enumerate(sequence):
        tmp_audio = f"out/seg_audio_{idx}.mp3"
        gTTS(seg_data['text'], lang='en').save(tmp_audio)
        audio_files.append(tmp_audio)

        action_name = seg_data.get('body_language', {}).get('action', 'talk_motion')
        if action_name not in action_library:
            continue

        word_start_idx = segment_boundaries[idx] if idx < len(segment_boundaries) else len(all_words)
        word_end_idx = segment_boundaries[idx + 1] if idx + 1 < len(segment_boundaries) else len(all_words)
        siw_start = whisper_words[word_start_idx]['start'] if word_start_idx < len(whisper_words) else duration
        siw_end = whisper_words[word_end_idx - 1]['end'] if (0 < word_end_idx <= len(whisper_words)) else duration
        
        segments.append({
            'id': idx, 
            'action': action_name, 
            'emotion': seg_data.get('emotion', 'neutral'), 
            'siw_start': min(siw_start, siw_end), 
            'siw_end': max(siw_start, siw_end)
        })

    siw_values = siw_map['continuous_importance']
    siw_times = siw_map['continuous_time']
    kine_results = optimize_kinematic_plan(siw_values, siw_times, segments, action_library)
    t_starts = kine_results['t_starts']
    p_times = kine_results['p_times']
    
    dummy_opt = KinematicOptimizer(siw_values, siw_times, segments, action_library)
    execution_plan = generate_execution_plan(segments, action_library, p_times, t_starts)

    calculate_pre_delays(execution_plan, sequence, audio_files, p_times, t_starts, action_library, aligner)
    recalculate_crossovers(execution_plan, action_library, p_times, dummy_opt)

    synced_audio_files = process_offline_audio(execution_plan)

    temp_video = "out/temp_raw_video.mp4" if args.record else None
    env = PyBulletEnv(video_out=temp_video)
    trajectory_frames = generate_global_trajectory(execution_plan, env.neutral_pose, blend_duration=0.5, fps=30)

    run_simulation(env, execution_plan, trajectory_frames)

    if args.record:
        print(f"\n[*] Merging Video and Audio to {args.record}...")
        master_audio = "out/master_audio.wav"
        cmd = f"ffmpeg -y -i {temp_video} -i {master_audio} -filter_complex \"[0:v]setpts=2.0*PTS[v]\" -map \"[v]\" -map 1:a -r 30 -c:v libx264 -c:a aac {args.record} -loglevel error"
        os.system(cmd)
        if os.path.exists(temp_video):
            os.remove(temp_video)
        print(f"Successfully saved Video Demo to: {args.record}")

    # Cleanup
    files_to_remove = audio_files + synced_audio_files + ([temp_full_audio] if temp_full_audio else [])
    cleanup(files_to_remove)


if __name__ == '__main__':
    main()