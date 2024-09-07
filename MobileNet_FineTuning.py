from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os

# GPU 장치 선택 (첫 번째 GPU 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0번 GPU 사용

# GPU 설정: 메모리 점진적 할당
gpus = tf.config.list_physical_devices('GPU')  # GPU 장치 목록 가져오기
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # GPU 메모리 점진적 할당
        print(f"{len(gpus)} GPU(s) are being used")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")

# TensorFlow가 GPU를 사용하는지 확인
print(f"TensorFlow is using the following device: {tf.test.gpu_device_name()}")

# 데이터 증강 및 전처리 설정
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # 픽셀 값을 [0, 1]로 정규화
    rotation_range=20,  # 회전
    width_shift_range=0.2,  # 가로 이동
    height_shift_range=0.2,  # 세로 이동
    shear_range=0.2,  # 전단 변환
    zoom_range=0.2,  # 확대
    horizontal_flip=True,  # 좌우 반전
    fill_mode='nearest'  # 빈 공간을 채움
)

# 학습 데이터 생성
train_generator = datagen.flow_from_directory(
    'Dataset/train',  # 학습 데이터 경로
    target_size=(224, 224),  # 이미지 크기
    batch_size=32,  # 배치 크기
    class_mode='categorical'  # 다중 클래스 분류
)

# 검증 데이터 생성
validation_generator = datagen.flow_from_directory(
    'Dataset/val',  # 검증 데이터 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 테스트 데이터 생성
test_generator = datagen.flow_from_directory(
    'Dataset/test',  # 테스트 데이터 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 사전 학습된 MobileNet 모델 로드, ImageNet 가중치 사용
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 기본 모델의 가중치 고정 (학습하지 않음)
for layer in base_model.layers:
    layer.trainable = False

# GlobalAveragePooling2D로 차원 축소 후 Dense 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)  # 차원 축소
x = Dense(1024, activation='relu')(x)  # 밀집층 추가
num_classes = train_generator.num_classes  # 클래스 수 (의류 카테고리 수)
predictions = Dense(num_classes, activation='softmax')(x)  # 출력층

# 새로운 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 고정된 레이어로 모델 학습 (새로운 레이어만 학습)
model.fit(train_generator, epochs=100, validation_data=validation_generator)

# 일부 레이어의 고정 해제 (상위 50개 레이어만 학습)
for layer in base_model.layers[-50:]:
    layer.trainable = True

# 파인튜닝 모델 컴파일
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 파인튜닝
model.fit(train_generator, epochs=100, validation_data=validation_generator)

# 테스트 데이터로 평가
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# 학습된 모델 저장
model.save('clothing_classification_mobilenet.h5')

# TensorFlow Lite 모델로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 변환된 모델 저장
with open('clothing_classification_mobilenet.tflite', 'wb') as f:
    f.write(tflite_model)
