from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.optimizers import Adam
import tensorflow as tf

def build_model(input_shape, num_actions, learning_rate=0.00025):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Dense(num_actions, activation='linear'))
    
    model.compile(optimizer=Adam(learning_rate), loss=tf.keras.losses.Huber())
    return model

