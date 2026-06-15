"""
SIW Pipeline - Full Audio Processing
Load Mock JSON → Full Text → Generate Audio → Whisper Timestamps → SIW Map → Visualize
"""

import json
import os
import sys
import numpy as np

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.audio.audio_aligner import AudioAligner
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import librosa
from gtts import gTTS


def load_and_merge_mock(mock_path=None):
    """
    Load mock JSON and merge all text + word importance scores
    """
    # print("[1] Loading mock JSON...")
    
    if mock_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        mock_path = os.path.join(base_dir, 'mock_llm_response.json')
    
    try:
        with open(mock_path, 'r', encoding='utf-8') as f:
            response = json.load(f)
    except Exception as e:
        # print(f"❌ Error loading JSON: {e}")
        return None, None, None, None
    
    sequence = response.get("sequence", [])
    
    # Extract all texts + word importance scores
    full_text = ""
    all_words = []
    word_importance_map = {}  # {"word": importance_score}
    segment_boundaries = []  # [word_index_0, word_index_1, ...] for each segment
    
    for idx, segment in enumerate(sequence):
        text = segment.get("text", "").strip()
        word_importance = segment.get("word_importance", {})  # Read from JSON
        
        if text:
            # Track start word index of this segment
            segment_boundaries.append(len(all_words))
            
            # Extract words from this segment
            segment_words = text.lower().split()
            for word in segment_words:
                clean_word = word.strip(".,!?;:")
                if clean_word:
                    all_words.append(clean_word)
                    full_text += clean_word + " "
                    
                    # Get importance score from JSON for this word
                    # If not found in word_importance, use default 0.30
                    if clean_word in word_importance:
                        word_importance_map[len(all_words) - 1] = word_importance[clean_word]
                    else:
                        word_importance_map[len(all_words) - 1] = 0.30
    
    full_text = full_text.strip()
    
    # print(f"✓ Loaded {len(sequence)} segments, {len(all_words)} words")
    # print(f"✓ Segment boundaries: {segment_boundaries}")
    
    return full_text, word_importance_map, segment_boundaries, all_words


def generate_full_audio(full_text, output_path="/tmp/full_audio.mp3"):
    """
    Generate audio from full text using gTTS
    """
    # print(f"\n[2] Generating audio from text...")
    
    try:
        # Generate audio
        tts = gTTS(text=full_text, lang='en', slow=False)
        tts.save(output_path)
        
        # Get duration
        import librosa
        y, sr = librosa.load(output_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        # print(f"✓ Audio generated ({duration:.2f}s)")
        
        return output_path, duration
    except Exception as e:
        # print(f"❌ Error: {e}")
        return None, 0


def get_real_timestamps(audio_path, aligner=None):
    """
    Use Whisper to transcribe audio and get word-level timestamps
    """
    # print(f"\n[3] Extracting timestamps from Whisper...")
    
    try:
        if aligner is None:
            aligner = AudioAligner()
        result = aligner.transcribe_audio(audio_path)
        
        if not result or len(result.get("words", [])) == 0:
            # print("❌ Whisper returned no words")
            return None, 0
        
        words_with_times = result["words"]
        duration = result["duration"]
        
        # print(f"✓ Transcribed {len(words_with_times)} words, duration {duration:.2f}s")
        
        return words_with_times, duration
    except Exception as e:
        # print(f"❌ Error: {e}")
        return None, 0


def match_whisper_to_importance(whisper_words, word_importance_map, all_words):
    """
    Match Whisper transcribed words to importance scores from JSON
    word_importance_map: {word_idx: importance_score}
    all_words: list of words extracted from JSON
    """
    # print(f"\n[4] Matching Whisper words to JSON importance scores...")
    
    word_importance_list = []
    emphasized_words = {}  # For visualization
    
    for whisper_word in whisper_words:
        w = whisper_word["word"].lower().strip(".,!?;:")
        
        # Find matching word in all_words and get its importance score
        importance = 0.30  # Default
        for idx, json_word in enumerate(all_words):
            if json_word == w:
                if idx in word_importance_map:
                    importance = word_importance_map[idx]
                break
        
        word_importance_list.append(importance)
        
        # Track emphasized words (importance >= 0.85) for visualization
        if importance >= 0.85:
            emphasized_words[w] = {
                "timestamp": whisper_word["start"],
                "importance": importance
            }
    
    # Show matched importance scores
    # print(f"✓ Matched {len(whisper_words)} Whisper words to JSON importance scores")
    # print(f"✓ Found {len(emphasized_words)} emphasized words")
    
    return word_importance_list, emphasized_words


def create_siw_map(whisper_words, word_importance_list, duration):
    """
    Create SIW map using Whisper timestamps and importance scores from JSON
    Light Gaussian smoothing to preserve peaks
    """
    # print(f"\n[5] Creating SIW map...")
    
    all_words = [w["word"] for w in whisper_words]
    
    # Equation 2: t_c = (t_start + t_end) / 2
    word_times = [(w["start"] + w["end"]) / 2.0 for w in whisper_words]
    
    # Create continuous timeline
    continuous_time = np.linspace(0, duration, 1000)
    
    # Equation 3: s(t) = sum(alpha * exp(-(t - t_c)^2 / (2 * sigma^2)))
    sigma = duration / (len(all_words) * 5.0)  # controls the temporal spread
    
    s_t = np.zeros_like(continuous_time)
    for alpha, t_c in zip(word_importance_list, word_times):
        s_t += alpha * np.exp(-((continuous_time - t_c)**2) / (2 * sigma**2))
        
    # Scale to maintain a safe lower bound for robot movement
    continuous_importance = np.clip(s_t, 0.25, 1.0)
    
    # print(f"✓ SIW map created: {len(all_words)} words, range {min(word_importance_list):.2f}-{max(word_importance_list):.2f}")
    
    return {
        "words": all_words,
        "word_importance": word_importance_list,
        "word_times": np.array(word_times),
        "continuous_time": continuous_time,
        "continuous_importance": continuous_importance,
        "duration": duration
    }


def visualize_siw(siw_map, emphasized_words, audio_path, segment_boundaries, output_path="/tmp/siw_output.png"):
    """
    Visualize SIW + audio waveform + segment boundaries
    Paper-quality visualization (IROS conference format)
    """
    # print(f"\n[6] Creating publication-quality visualization...")
    
    # Set style for paper
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 1.8,
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5
    })
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    duration = siw_map["duration"]
    time = siw_map["continuous_time"]
    importance = siw_map["continuous_importance"]
    word_times = siw_map["word_times"]
    
    # ===== Panel (a): SIW + Emphasized Words =====
    ax1.fill_between(time, 0, importance, alpha=0.35, color='#E74C3C', edgecolor='none')
    ax1.plot(time, importance, color='#C0392B', linewidth=2.0, label='SIW')
    
    # Draw segment boundaries (vertical dashed lines)
    if segment_boundaries and len(segment_boundaries) > 1:
        for word_idx in segment_boundaries[1:]:
            if word_idx < len(word_times):
                boundary_time = word_times[word_idx]
                ax1.axvline(x=boundary_time, color='#888888', linestyle='--', 
                           linewidth=1.2, alpha=0.6, zorder=2)
    
    # Mark emphasized words (high importance)
    emphasized_list = []
    for emphasized, info in emphasized_words.items():
        ts = info["timestamp"]
        importance_val = info["importance"]
        
        # Find value at this timestamp
        idx = np.argmin(np.abs(time - ts))
        imp_value = importance[idx]
        
        # Draw marker
        ax1.scatter(ts, imp_value, color='#E74C3C', s=180, marker='o', 
                   zorder=5, edgecolor='#8B0000', linewidth=1.5, alpha=0.95)
        
        # Add text label
        ax1.text(ts, imp_value + 0.13, emphasized.upper(), 
                fontsize=9, ha='center', fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5B4B4', 
                         edgecolor='#C0392B', linewidth=0.8, alpha=0.88))
        emphasized_list.append(emphasized)
    
    ax1.set_ylim(-0.02, 1.2)
    ax1.set_ylabel('Importance Level', fontsize=11, fontweight='bold')
    ax1.set_xlim(0, duration)
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=11, 
            fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.8))
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    
    # Legend with segment and emphasized info
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#C0392B', linewidth=2.0, label='Speech Importance'),
        Line2D([0], [0], color='#888888', linestyle='--', linewidth=1.2, label='Segment boundary')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True, 
              framealpha=0.95, edgecolor='black', fancybox=True)
    
    # ===== Panel (b): Audio Waveform =====
    time_audio = np.linspace(0, len(y) / sr, len(y))
    ax2.plot(time_audio, y, color='#3498DB', linewidth=0.7, alpha=0.9)
    
    # Draw segment boundaries on waveform
    if segment_boundaries and len(segment_boundaries) > 1:
        for word_idx in segment_boundaries[1:]:
            if word_idx < len(word_times):
                boundary_time = word_times[word_idx]
                ax2.axvline(x=boundary_time, color='#888888', linestyle='--', 
                           linewidth=1.2, alpha=0.6, zorder=2)
    
    ax2.fill_between(time_audio, y, alpha=0.3, color='#3498DB', edgecolor='none')
    ax2.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, duration)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=11, 
            fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.8))
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    
    # Add overall title
    fig.suptitle('Module 1: Semantic Importance Wave (SIW)', 
                fontsize=13, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    # print(f"✓ Publication-quality visualization saved: {output_path}")
    plt.close()


