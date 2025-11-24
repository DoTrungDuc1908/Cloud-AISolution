import os
import time
import math
import numpy as np
import torch
import librosa
from dotenv import load_dotenv
import requests
import base64
import json
from concurrent.futures import ThreadPoolExecutor

# Azure SDK imports
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk

# Hugging Face imports
from transformers import AutoModelForAudioClassification

# --- CẤU HÌNH CÁC LOẠI CẢM XÚC ---

EMOTION_ANGLES = {
    # --- Quadrant I (0° - 90°): Positive Valence, High Arousal ---
    "Happy":      (0, 30),    # Thiên về Positive
    "Delighted":  (30, 60),   # Cân bằng
    "Excited":    (60, 90),   # Thiên về Arousal

    # --- Quadrant II (90° - 180°): Negative Valence, High Arousal ---
    "Tense":      (90, 120),  # Thiên về Arousal
    "Angry":      (120, 150), # Cân bằng
    "Frustrated": (150, 180), # Thiên về Negative

    # --- Quadrant III (180° - 270°): Negative Valence, Low Arousal ---
    "Depressed":  (180, 210), # Thiên về Negative
    "Bored":      (210, 240), # Cân bằng
    "Tired":      (240, 270), # Thiên về Low Arousal

    # --- Quadrant IV (270° - 360°): Positive Valence, Low Arousal ---
    "Calm":       (270, 300), # Thiên về Low Arousal
    "Relaxed":    (300, 330), # Cân bằng
    "Content":    (330, 360), # Thiên về Positive

    # --- Special Case ---
    "Neutral":    None        # Nằm tại tâm (0,0)
}

session = requests.Session()

# --- CÁC HÀM XỬ LÝ ---

def recognize_from_file(api_key, region, language, audio_file_path):
    """
    Phiên bản tối ưu: Dùng recognize_once cho file ngắn (< 30s)
    Giúp giảm thời gian chờ từ 15s -> 2-3s
    """
    # Cấu hình Speech
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.speech_recognition_language = language

    # Kiểm tra file
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy file tại {audio_file_path}")

    # Cấu hình Audio
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Đang nhận diện giọng nói (Single Shot)...")
    
    # Dùng recognize_once thay vì continuous
    # Hàm này sẽ tự động dừng khi hết câu hoặc hết file
    result = speech_recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("Không nhận diện được giọng nói.")
        return ""
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Đã bị hủy: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Lỗi chi tiết: {cancellation_details.error_details}")
        return ""
        
    return ""

def get_audio_arousal_from_cloud(audio_path, endpoint_url, api_key):
    """
    Gửi Audio lên Azure Endpoint để lấy điểm Arousal.
    Đã bỏ cơ chế tự động Retry.
    """

    # 1. Kiểm tra file
    if not os.path.exists(audio_path):
        print(f"Lỗi: Không tìm thấy file {audio_path}")
        return 0.0

    # 2. Đọc và mã hóa
    try:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return 0.0

    # 3. Cấu hình Request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "azureml-model-deployment": "green"
    }
    payload = {"data": audio_b64}

    # 4. Gọi API
    try:
        # Timeout: (Connect Timeout, Read Timeout)
        # 3.05s để kết nối, 180s để chờ server xử lý
        resp = session.post(endpoint_url, json=payload, headers=headers, timeout=(3.05, 180))
        
        if resp.status_code == 200:
            result = resp.json()
            
            # --- XỬ LÝ AN TOÀN: CHỐNG LỖI DOUBLE JSON ---
            # Phòng trường hợp server trả về string thay vì dict
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    print(f"Lỗi: Server trả về chuỗi không phải JSON: {result}")
                    return 0.0
            # --------------------------------------------

            return float(result.get("arousal_raw", 0.0))
            
        else:
            print(f"Lỗi API ({resp.status_code}): {resp.text}")
            return 0.0

    # Đã bỏ catch requests.exceptions.RetryError
    except requests.exceptions.Timeout:
        print("Lỗi: Request Timeout (Server xử lý quá lâu).")
        return 0.0
    except Exception as e:
        print(f"Lỗi không xác định khi gọi API: {e}")
        return 0.0


def rescale_valence_score(result):
    """Chuyển đổi kết quả Azure Sentiment thành thang đo -1 đến 1"""
    positive_percent = result.confidence_scores.positive
    negative_percent = result.confidence_scores.negative
    # Neutral không ảnh hưởng trực tiếp đến dấu, chỉ làm giảm độ lớn của score
    valence_score = positive_percent - negative_percent
    return valence_score

def rescale_arousal_score(arousal_val):
    """Chuyển đổi thang đo Arousal (giả sử 0-1) sang -1 đến 1"""
    orig_scale_range = 1.0
    afterwards_scale_range = 2.0 # range (-1 -> 1) là 2 đơn vị
    
    # Tính toán đơn giản: (val * 2) - 1
    afterwards_dist_from_left = arousal_val * afterwards_scale_range / orig_scale_range
    arousal_rescaled = -1 + afterwards_dist_from_left
    return arousal_rescaled

def compute_verbal_sentiment(arousal, valence, angle_dict):
    """Kết hợp Valence và Arousal để suy ra nhãn cảm xúc"""
    
    # 1. Xử lý trường hợp tâm (Neutral)
    radius = math.sqrt(valence**2 + arousal**2)
    if radius < 0.1:
        return "Neutral"

    # 2. Tính góc (Degrees)
    angle = math.degrees(math.atan2(arousal, valence))

    # 3. Chuẩn hóa góc về 0 - 360
    if angle < 0:
        angle += 360
        
    # Xử lý biên 360 độ về 0 để khớp với logic
    angle = angle % 360

    # 4. So khớp với Dictionary
    found_label = "Unknown"
    for label, r in angle_dict.items():
        if r is None: continue 
        
        low, high = r
        if low <= angle < high:
            found_label = label
            break
            
    return found_label

# --- MAIN EXECUTION ---

def main():
    # 1. Load Environment Variables
    load_dotenv(override=True)
    
    audio_path = "Azure AI/audio/5s-audio-only (1).wav" 
    language = "en-US" # Hoặc "en-US" tùy bạn chọn
    
    # Lấy Key từ .env
    LANGUAGE_ENDPOINT = os.getenv("LANGUAGE_ENDPOINT")
    LANGUAGE_KEY = os.getenv("LANGUAGE_KEY")
    SPEECH_KEY = os.getenv("SPEECH_KEY")
    SPEECH_REGION = os.getenv("SPEECH_REGION")
    ML_ENDPOINT_URL = os.getenv("ML_ENDPOINT_URL")
    ML_API_KEY = os.getenv("ML_API_KEY")

    if not all([LANGUAGE_ENDPOINT, LANGUAGE_KEY, SPEECH_KEY, SPEECH_REGION]):
        print("Lỗi: Thiếu biến môi trường trong file .env")
        return

    try:
        # Chuẩn bị Client sẵn ở ngoài
        credential = AzureKeyCredential(LANGUAGE_KEY)
        text_client = TextAnalyticsClient(endpoint=LANGUAGE_ENDPOINT, credential=credential)

        # --- ĐỊNH NGHĨA 2 PIPELINE LỚN ---

        # --- KHỞI TẠO MỐC THỜI GIAN ---
        start_time = time.time()

        def process_full_text_pipeline():
            """
            Luồng 1 (Chain): Speech-to-Text -> Text Sentiment
            """
            try:
                # BƯỚC A: Speech to Text
                elapsed = time.time() - start_time
                print(f"[{elapsed:.2f}s] Luồng 1: Đang nhận diện giọng nói...")
                
                text_content = recognize_from_file(SPEECH_KEY, SPEECH_REGION, language=language, audio_file_path=audio_path)
                
                elapsed = time.time() - start_time
                print(f"[{elapsed:.2f}s] Luồng 1 [Text]: {text_content}")
                
                if not text_content.strip():
                    print(f"[{elapsed:.2f}s] Luồng 1: Không có văn bản, bỏ qua phân tích sentiment.")
                    return 0.0

                # BƯỚC B: Text Sentiment Analysis
                elapsed = time.time() - start_time
                print(f"[{elapsed:.2f}s] Luồng 1: Đang phân tích cảm xúc văn bản...")
                
                results = text_client.analyze_sentiment([text_content])
                
                elapsed = time.time() - start_time
                print(f"[{elapsed:.2f}s] Luồng 1: Phân tích xong cảm xúc văn bản")
                
                val_result = results[0]
                
                if val_result.is_error:
                    print(f"[{elapsed:.2f}s] Luồng 1 Lỗi: {val_result.error.message}")
                    return 0.0
                
                # Trả về điểm Valence
                return rescale_valence_score(val_result)
                
            except Exception as e:
                print(f"[{time.time() - start_time:.2f}s] Lỗi tại Luồng 1 (Text Pipeline): {e}")
                return 0.0

        def process_audio_pipeline():
            """
            Luồng 2 (Independent): Audio Arousal Analysis (WavLM)
            """
            try:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.2f}s] Luồng 2: Đang gửi Audio lên Cloud (WavLM)...")
                
                # Gọi API Audio
                raw_val = get_audio_arousal_from_cloud(audio_path, endpoint_url=ML_ENDPOINT_URL, api_key=ML_API_KEY)
                
                elapsed = time.time() - start_time
                print(f"[{elapsed:.2f}s] Luồng 2 [Raw Arousal]: {raw_val}")
                
                # Tính toán điểm Arousal
                return rescale_arousal_score(raw_val)
                
            except Exception as e:
                print(f"[{time.time() - start_time:.2f}s] Lỗi tại Luồng 2 (Audio Pipeline): {e}")
                return 0.0

        # --- THỰC THI SONG SONG (PARALLEL EXECUTION) ---
        print(f"{'-'*20}\n[{0.00}s] BẮT ĐẦU CHẠY 2 LUỒNG SONG SONG\n{'-'*20}")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Gửi 2 pipeline lớn vào xử lý cùng lúc
            future_text_chain = executor.submit(process_full_text_pipeline)
            future_audio_chain = executor.submit(process_audio_pipeline)

            # Chờ kết quả
            valence_score = future_text_chain.result()
            arousal_score = future_audio_chain.result()

        total_time = time.time() - start_time
        print(f"{'-'*20}\n[{total_time:.2f}s] ĐÃ HOÀN THÀNH CẢ 2 LUỒNG\n{'-'*20}")

        # ---------------------------------------------------------
        # BƯỚC 3: Tổng hợp kết quả
        # ---------------------------------------------------------
        
        final_emotion = compute_verbal_sentiment(arousal_score, valence_score, EMOTION_ANGLES)
        
        print(f"KẾT QUẢ CUỐI CÙNG: {final_emotion.upper()}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()