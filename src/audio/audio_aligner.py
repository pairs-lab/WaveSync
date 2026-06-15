import whisper_timestamped as whisper
import torch
import numpy as np


class AudioAligner:
    def __init__(self):
        # print("Đang tải Whisper Model...")
        # Chọn thiết bị (ưu tiên GPU nếu có)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model "base" (nhẹ, chạy nhanh)
        self.model = whisper.load_model("base", device=self.device)
        # print("✓ Whisper model loaded")
        
        self.last_result = None
        self.last_duration = None

    def transcribe_audio(self, audio_file):
        """
        Nhận diện toàn bộ audio và trả về kết quả với timestamp từng chữ
        
        Returns:
            {
                "text": "Toàn bộ text",
                "duration": 8.5,
                "words": [
                    {"word": "xin", "start": 0.1, "end": 0.5},
                    {"word": "chào", "start": 0.5, "end": 1.0},
                    ...
                ]
            }
        """
        try:
            audio = whisper.load_audio(audio_file)
            result = whisper.transcribe(self.model, audio, language="en")
            
            # Extract duration từ audio
            import librosa
            y, sr = librosa.load(audio_file, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Extract all words with timestamps
            words = []
            full_text = ""
            
            for segment in result["segments"]:
                for word_info in segment["words"]:
                    word_text = word_info["text"].lower().strip(".,!?;:")
                    if word_text:
                        words.append({
                            "word": word_text,
                            "start": word_info["start"],
                            "end": word_info["end"]
                        })
                        full_text += word_text + " "
            
            output = {
                "text": full_text.strip(),
                "duration": duration,
                "words": words
            }
            
            self.last_result = output
            self.last_duration = duration
            
            # print(f"✓ Transcribed {len(words)} words from {duration:.2f}s audio")
            
            return output
            
        except Exception as e:
            # print(f"❌ Transcription error: {e}")
            return None

    def get_word_timestamp(self, audio_file, target_word):
        """Lấy timestamp của một từ cụ thể"""
        if not self.last_result:
            self.transcribe_audio(audio_file)
        
        if not self.last_result:
            return 0.0
        
        target = target_word.lower().strip()
        
        for word_info in self.last_result["words"]:
            if word_info["word"] == target:
                return word_info["start"]
        
        return 0.0
    
    def get_all_word_timestamps(self, audio_file):
        """Lấy timestamp của tất cả các từ"""
        result = self.transcribe_audio(audio_file)
        if result:
            return result["words"], result["duration"]
        return [], 0.0
    
    def find_emphasized_timestamps(self, audio_file, emphasized_words):
        """
        Tìm timestamp của các từ nhấn mạnh
        
        Args:
            audio_file: Path to audio
            emphasized_words: List các từ emphasized từ Gemini
                             Example: ["chào", "RoBL-01", "vui"]
        
        Returns:
            {
                "chào": 0.5,
                "RoBL-01": 2.3,
                "vui": 4.1,
                ...
            }
        """
        result = self.transcribe_audio(audio_file)
        if not result:
            return {}
        
        timestamps = {}
        emphasized_lower = [w.lower().strip() for w in emphasized_words]
        
        for word_info in result["words"]:
            word = word_info["word"]
            if word in emphasized_lower:
                timestamps[word] = word_info["start"]
        
        # print(f"✓ Found timestamps for {len(timestamps)} emphasized words")
        return timestamps