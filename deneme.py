import tensorflow as tf

print("TensorFlow version:", tf.__version__)

try:
    from tensorflow.keras.models import Sequential
    print("Keras is available in TensorFlow.")
except ImportError:
    print("Keras is not available in TensorFlow.")
