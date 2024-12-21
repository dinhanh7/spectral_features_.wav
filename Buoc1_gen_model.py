import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

# Đường dẫn tới dataset
DATASET_PATH = "data"  # Thư mục chứa các tệp âm thanh, chia thành các thư mục con (cat, dog, bird)
LABELS = ["cat", "dog", "bird"]

# Hàm trích xuất Spectral Features
def extract_spectral_features(file_path, max_len=256):
    y, sr = librosa.load(file_path, sr=None)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    features = np.vstack([
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        spectral_contrast
    ])

    # Padding để đảm bảo kích thước ma trận đồng nhất
    if features.shape[1] > max_len:
        features = features[:, :max_len]
    else:
        padding = max_len - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, padding)), mode='constant')
    return features

# Hàm load dữ liệu từ thư mục
def load_data(dataset_path):
    data = []
    targets = []
    for label_idx, label in enumerate(LABELS):
        label_dir = os.path.join(dataset_path, label)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            try:
                spectral_features = extract_spectral_features(file_path)
                data.append(spectral_features)
                targets.append(label_idx)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return np.array(data), np.array(targets)

# Load dữ liệu
print("Loading dataset...")
data, targets = load_data(DATASET_PATH)
data = data[..., np.newaxis]  # Thêm chiều channel (dạng ảnh)

# Chia tập dữ liệu: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Xây dựng mô hình CNN
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), 
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=(1, 1)),  
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=(1, 1)),
        Dropout(0.25),
        
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Khởi tạo và huấn luyện mô hình
print("Building CNN model...")
model = build_cnn(input_shape=X_train[0].shape, num_classes=len(LABELS))

# Thiết lập EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Theo dõi sự cải thiện của loss trên tập validation
    patience=20,         # Số epoch không cải thiện trước khi dừng
    restore_best_weights=True  # Khôi phục trọng số của mô hình tốt nhất
)

print("Training model with Early Stopping...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping]
)

# Đánh giá mô hình trên tập test
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Lưu mô hình đã huấn luyện
MODEL_PATH = "improved_model.keras"
model.save(MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")
