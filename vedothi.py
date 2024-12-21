import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Hàm trích xuất và vẽ các Spectral Features
def plot_spectral_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Trích xuất Spectral Features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    times = librosa.times_like(spectral_centroid, sr=sr)
    
    # Vẽ các Spectral Features
    plt.figure(figsize=(10, 8))
    
    # Spectral Centroid
    plt.subplot(4, 1, 1)
    plt.semilogy(times, spectral_centroid, label='Spectral Centroid', color='blue')
    plt.ylabel("Hz")
    plt.title("Spectral Centroid")
    plt.legend()
    
    # Spectral Bandwidth
    plt.subplot(4, 1, 2)
    plt.semilogy(times, spectral_bandwidth, label='Spectral Bandwidth', color='orange')
    plt.ylabel("Hz")
    plt.title("Spectral Bandwidth")
    plt.legend()
    
    # Spectral Roll-off
    plt.subplot(4, 1, 3)
    plt.semilogy(times, spectral_rolloff, label='Spectral Roll-off', color='green')
    plt.ylabel("Hz")
    plt.title("Spectral Roll-off")
    plt.legend()
    
    # Spectral Contrast
    plt.subplot(4, 1, 4)
    for i, band in enumerate(spectral_contrast):
        plt.plot(times, band, label=f'Band {i+1}')
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.title("Spectral Contrast")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Đường dẫn tới file .wav cần phân tích
AUDIO_FILE = "C:\\Users\\WINDOWS\\OneDrive - Hanoi University of Science and Technology\\Documents\\0.Temp GitHub\\Voice-DOG-CAT-BIRD\\test\\dog\\test_dog_2ce7.wav"

# Gọi hàm hiển thị
if __name__ == "__main__":
    if os.path.isfile(AUDIO_FILE):
        print(f"Analyzing Spectral Features for file: {AUDIO_FILE}")
        plot_spectral_features(AUDIO_FILE)
    else:
        print(f"File not found: {AUDIO_FILE}")
