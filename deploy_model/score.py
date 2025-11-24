import os
import json
import torch
import librosa
import base64
import tempfile
import numpy as np
from transformers import AutoModelForAudioClassification

model = None

def init():
    global model
    print("Đang tải model WavLM...")
    # Tải model Hugging Face
    model_name = "3loi/SER-Odyssey-Baseline-WavLM-Arousal"
    model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print("Model đã sẵn sàng!")

def run(raw_data):
    try:
        # 1. Nhận Base64 audio từ Client
        input_json = json.loads(raw_data)
        audio_b64 = input_json.get("data") # Client gửi key là "data"
        
        if not audio_b64:
            return {"error": "Thiếu dữ liệu 'data' (base64 audio)"}

        # 2. Ghi ra file tạm để Librosa đọc
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(base64.b64decode(audio_b64))
            temp_path = temp_audio.name

        try:
            # 3. Xử lý Audio
            # Model này thường train ở 16000Hz, nên load đúng sr=16000
            raw_wav, _ = librosa.load(temp_path, sr=16000)
            
            # Chuẩn hóa (Theo logic model của bạn)
            mean = model.config.mean
            std = model.config.std
            norm_wav = (raw_wav - mean) / (std + 0.000001)
            
            mask = torch.ones(1, len(norm_wav))
            wavs = torch.tensor(norm_wav).unsqueeze(0)
            
            # 4. Dự đoán
            with torch.no_grad():
                pred = model(wavs, mask)
            
            # Lấy kết quả thô
            arousal_raw = float(pred[0][0]) 
            
            return {"arousal_raw": arousal_raw} 

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        # Trả về Dict lỗi
        return {"error": str(e)}