import numpy as np
import librosa
import tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt

# Đường dẫn tới file mô hình đã lưu
MODEL_PATH = "ok_model.keras"
LABELS = ["cat", "dog", "bird"]

# Hàm chuyển file âm thanh thành ma trận phổ Mel (giống như khi train)
def extract_mel_spectrogram(file_path, n_mels=128, max_len=128):
    y, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] > max_len:
        mel_spec_db = mel_spec_db[:, :max_len]
    else:
        padding = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, padding)), mode='constant')
    return mel_spec_db

# Hàm dự đoán nhãn của file âm thanh
def predict_audio_class(model, file_path):
    mel_spec = extract_mel_spectrogram(file_path)  #chuyển tệp âm thanh thành phổ Mel
    mel_spec = mel_spec[np.newaxis, ..., np.newaxis]  # Thêm batch dimension và channel dimension, dạng (1, 128, 128, 1)
    prediction = model.predict(mel_spec) #Trả về xác suất của từng nhãn
    predicted_label = np.argmax(prediction) #Lấy nhãn có xác suất cao nhất
    return LABELS[predicted_label]

# Tải mô hình đã lưu
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Đường dẫn tới file âm thanh cần dự đoán
AUDIO_FILE = "C:\\Users\WINDOWS\\OneDrive - Hanoi University of Science and Technology\Documents\\0.Temp GitHub\\Voice-DOG-CAT-BIRD\\test\\dog\\test_dog_2ce7.wav"  # Thay bằng đường dẫn file thực tế

# Dự đoán và in kết quả
print(f"Predicting class for file: {AUDIO_FILE}")
predicted_class = predict_audio_class(model, AUDIO_FILE)
print(f"The audio file is predicted to be: {predicted_class}")


#ve do thi
