import tensorflow as tf

# 사용 가능한 GPU 장치 목록을 출력합니다.
gpu_devices = tf.config.list_physical_devices('GPU')

print(f"Num GPUs Available: {len(gpu_devices)}")

if gpu_devices:
    print("TensorFlow가 다음 GPU를 사용합니다:", gpu_devices)
else:
    print("TensorFlow가 GPU를 찾지 못했습니다. CPU로 실행됩니다.")