"""
Module 2: Kinematic Wave-front Generation & Optimization
Temporal alignment of humanoid gestures with speech using Semantic Importance Wave
[UPDATED]: STRICT SEQUENTIAL CROSSOVER LAWS (Down-slope -> Up-slope)
"""
import numpy as np
from scipy.signal import correlate
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class ParabolicPulse:
    def __init__(self, duration: float, p_time: float = 1.0):
        self.duration_orig = duration
        self.p_time = max(p_time, 0.6)
        self.duration_scaled = duration * self.p_time

    def evaluate(self, t: np.ndarray, t_start: float) -> np.ndarray:
        t_peak = t_start + self.duration_scaled / 2.0
        t_half = self.duration_scaled / 2.0
        pulse = np.maximum(0, 1.0 - ((t - t_peak) / t_half) ** 2)
        return pulse

    def get_peak_time(self, t_start: float) -> float:
        return t_start + self.duration_scaled / 2.0

    def get_end_time(self, t_start: float) -> float:
        return t_start + self.duration_scaled

class KinematicOptimizer:
    def __init__(self, siw_map: np.ndarray, siw_times: np.ndarray, segments: List[Dict], action_library: Dict):
        self.siw_map = siw_map
        self.siw_times = siw_times
        self.segments = segments
        self.action_library = action_library

        self.pulses: List[ParabolicPulse] = []
        self.t_starts: List[float] = []
        self.p_times: List[float] = []
        self.audio_delays: List[float] = []
        self.kinematic_trajectory: Optional[np.ndarray] = None

        self.min_p_time = 0.6
        self.crossover_safe_threshold = 2.0 / 3.0

    def _find_optimal_start_time(self, segment_idx: int, action: ParabolicPulse, search_window: Tuple[float, float]) -> float:
        t_min, t_max = search_window
        mask = (self.siw_times >= t_min) & (self.siw_times <= t_max)
        siw_window = self.siw_map[mask]
        siw_times_window = self.siw_times[mask]

        if len(siw_window) == 0:
            return (t_min + t_max) / 2.0

        best_score = -np.inf
        best_t_start = t_min
        
        best_score = -np.inf
        best_t_start = t_min
        
        t_max_safe = max(t_min, t_max - action.duration_scaled)
        
        for t_start in np.linspace(t_min, t_max_safe, 20):
            action_pulse = action.evaluate(siw_times_window, t_start)
            score = np.sum(action_pulse * siw_window)
            if score > best_score:
                best_score = score
                best_t_start = t_start

        return best_t_start

    def _find_crossover_point(self, pulse1: ParabolicPulse, t_start1: float, pulse2: ParabolicPulse, t_start2: float) -> Optional[Tuple[float, float]]:
        t_end1 = pulse1.get_end_time(t_start1)
        if t_start2 > t_end1: return None 
        
        t_search = np.linspace(max(t_start1, t_start2), min(t_end1, pulse2.get_end_time(t_start2)), 100)
        p1_vals = pulse1.evaluate(t_search, t_start1)
        p2_vals = pulse2.evaluate(t_search, t_start2)

        diff = np.abs(p1_vals - p2_vals)
        idx_cross = np.argmin(diff)
        
        if diff[idx_cross] < 0.05:
            return (t_search[idx_cross], p1_vals[idx_cross])
        return None

    def _is_crossover_safe(self, pulse1: ParabolicPulse, t_start1: float, pulse2: ParabolicPulse, t_start2: float, t_cross: float) -> bool:
        if t_start2 < t_start1:
            return False

        t_peak1 = pulse1.get_peak_time(t_start1)
        t_peak2 = pulse2.get_peak_time(t_start2)

        t_peak1 = pulse1.get_peak_time(t_start1)
        t_peak2 = pulse2.get_peak_time(t_start2)

        if t_cross < t_peak1:
            return False

        if t_cross > t_peak2:
            return False

        height = pulse2.evaluate(np.array([t_cross]), t_start2)[0]
        return height <= self.crossover_safe_threshold

    def _calculate_minimal_delay(self, pulse1: ParabolicPulse, t_start1: float, pulse2: ParabolicPulse, t_start2_original: float) -> float:
        start_delay = 0.0
        if t_start2_original < t_start1:
            start_delay = t_start1 - t_start2_original

        for delay in np.arange(start_delay, 5.0, 0.01): 
            t_start2_test = t_start2_original + delay
            crossover = self._find_crossover_point(pulse1, t_start1, pulse2, t_start2_test)
            
            if crossover is None:
                if t_start2_test >= t_start1:
                    return delay
                continue
                
            t_cross, height = crossover
            
            if self._is_crossover_safe(pulse1, t_start1, pulse2, t_start2_test, t_cross):
                return delay
                
        return pulse1.get_end_time(t_start1) - t_start2_original + 0.1

    def optimize(self) -> Dict:
        self.pulses = []
        self.t_starts = []
        self.p_times = []
        self.audio_delays = []

        t_min = self.siw_times[0]
        t_max = self.siw_times[-1]

        for seg_idx, segment in enumerate(self.segments):
            action_name = segment.get('action')
            if action_name not in self.action_library:
                continue
                
            action_duration = self.action_library[action_name]['duration']
            search_window = (segment['siw_start'], segment['siw_end']) if 'siw_start' in segment else (t_min, t_max)

            segment_length = search_window[1] - search_window[0]
            optimal_p_time = np.clip(segment_length / action_duration, self.min_p_time, 1.15)

            pulse = ParabolicPulse(action_duration, p_time=optimal_p_time)
            if action_name == "talk_motion":
                t_start = search_window[0]
            else:
                t_start = self._find_optimal_start_time(seg_idx, pulse, search_window)
            p_time = optimal_p_time

            if seg_idx > 0:
                prev_pulse = self.pulses[-1]
                prev_t_start = self.t_starts[-1]

                def is_setup_safe(test_p_time, test_t_start):
                    test_pulse = ParabolicPulse(action_duration, p_time=test_p_time)
                    cross = self._find_crossover_point(prev_pulse, prev_t_start, test_pulse, test_t_start)
                    if cross is None:
                        return test_t_start >= prev_t_start
                    else:
                        return self._is_crossover_safe(prev_pulse, prev_t_start, test_pulse, test_t_start, cross[0])

                if not is_setup_safe(p_time, t_start):
                    for p_time_test in np.linspace(p_time, self.min_p_time, 20):
                        if is_setup_safe(p_time_test, t_start):
                            p_time = p_time_test
                            pulse = ParabolicPulse(action_duration, p_time=p_time)
                            break
                    else:
                        pulse = ParabolicPulse(action_duration, p_time=self.min_p_time)
                        p_time = self.min_p_time
                        audio_delay = self._calculate_minimal_delay(prev_pulse, prev_t_start, pulse, t_start)
                        t_start += audio_delay
                        self.audio_delays.append(audio_delay)
                        if audio_delay > 0:
                            pass

            self.pulses.append(pulse)
            self.t_starts.append(t_start)
            self.p_times.append(p_time)
            
            if seg_idx >= len(self.audio_delays):
                self.audio_delays.append(0.0)

        self._generate_kinematic_trajectory()
        
        return {
            'pulses': self.pulses,
            't_starts': self.t_starts,
            'p_times': self.p_times,
            'audio_delays': self.audio_delays,
            'kinematic_trajectory': self.kinematic_trajectory
        }

    def _generate_kinematic_trajectory(self):
        trajectory = np.zeros_like(self.siw_map)
        for pulse, t_start in zip(self.pulses, self.t_starts):
            pulse_vals = pulse.evaluate(self.siw_times, t_start)
            trajectory = np.maximum(trajectory, pulse_vals)
        self.kinematic_trajectory = trajectory

def visualize_optimization(siw_map: np.ndarray, siw_times: np.ndarray, optimizer: KinematicOptimizer, output_path: str = "/tmp/kinematic_optimization.png"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].fill_between(siw_times, 0, siw_map, alpha=0.5, color='#FF6B6B', label='SIW Map')
    axes[0].plot(siw_times, siw_map, color='#C92A2A', linewidth=2)
    axes[0].set_ylabel('SIW Importance', fontweight='bold')
    axes[0].set_title('Module 1: Semantic Importance Wave', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(siw_times[0], siw_times[-1])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(optimizer.pulses)))
    for i, (pulse, t_start) in enumerate(zip(optimizer.pulses, optimizer.t_starts)):
        pulse_vals = pulse.evaluate(siw_times, t_start)
        axes[1].plot(siw_times, pulse_vals, color=colors[i], linewidth=2, label=f'Action {i} (p_time={optimizer.p_times[i]:.2f})')
        if i > 0:
            prev_pulse = optimizer.pulses[i-1]
            prev_t_start = optimizer.t_starts[i-1]
            crossover = optimizer._find_crossover_point(prev_pulse, prev_t_start, pulse, t_start)
            if crossover:
                axes[1].scatter([crossover[0]], [crossover[1]], s=100, marker='x', color='red', linewidths=2)
                
    axes[1].set_ylabel('Pulse Amplitude', fontweight='bold')
    axes[1].set_title('Module 2: Individual Parabolic Pulses (Strict Sequential Blend)', fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(siw_times[0], siw_times[-1])
    
    axes[2].fill_between(siw_times, 0, optimizer.kinematic_trajectory, alpha=0.5, color='#4ECDC4')
    axes[2].plot(siw_times, optimizer.kinematic_trajectory, color='#1A535C', linewidth=2.5)
    axes[2].set_ylabel('Kinematic Amplitude', fontweight='bold')
    axes[2].set_xlabel('Time (seconds)', fontweight='bold')
    axes[2].set_title('Module 2: Blended Kinematic Trajectory', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(siw_times[0], siw_times[-1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def optimize_kinematic_plan(siw_map: np.ndarray, siw_times: np.ndarray, segments: List[Dict], action_library: Dict) -> Dict:
    optimizer = KinematicOptimizer(siw_map, siw_times, segments, action_library)
    results = optimizer.optimize()
    return results