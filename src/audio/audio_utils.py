import os
import time
import numpy as np
import matplotlib.pyplot as plt

def play_audio_bg(audio_path):
    """
    Play audio in the background using the appropriate system command.
    """
    if audio_path.endswith('.wav'):
        os.system(f"aplay -q {audio_path} &")
    else:
        os.system(f"mpg123 -q {audio_path} &")

def delayed_play_audio(filename, delay):
    """
    Play audio after a specific delay.
    """
    if delay > 0:
        time.sleep(delay)
    play_audio_bg(filename)

def visualize_audio_blocks(vis_data, master_audio, sr, output_path="audio_sync_vis.png"):
    """
    Visualize audio and motion blocks for syncing analysis.
    """
    fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    time_axis = np.arange(len(master_audio)) / sr
    ax[0].plot(time_axis, master_audio, color='gray', alpha=0.8)
    ax[0].set_title("Master Audio Waveform", fontweight='bold')
    ax[0].set_ylabel("Amplitude")
    ax[0].grid(True, alpha=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, len(vis_data)))
    for i, data in enumerate(vis_data):
        y_pos = len(vis_data) - i
        
        # Action Block (Light)
        ax[1].barh(y_pos, data['block_end'] - data['block_start'], left=data['block_start'],
                   height=0.5, color=colors[i], alpha=0.3, edgecolor='none')

        # Audio Block (Dark)
        total_audio_len = data['speech_end'] - data['speech_start']
        if total_audio_len > 0:
            ax[1].barh(y_pos, total_audio_len, left=data['speech_start'],
                       height=0.5, color=colors[i], alpha=1.0, edgecolor='black', linewidth=1.5)

        ax[1].text(data['speech_start'], y_pos + 0.35, f"[{i}] {data['action']}", fontsize=10, fontweight='bold')

    ax[1].set_title("Timeline: Motion Block (Light) vs Voice+Delay Block (Dark)", fontweight='bold')
    ax[1].set_xlabel("Time (seconds)", fontweight='bold')
    ax[1].set_yticks([])
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
