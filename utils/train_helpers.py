import tensorflow as tf
import numpy as np
import datetime



class ActivationWars:

    def __init__(self, act_config, datasets=None, input_shape=None):
        self.act_config = act_config
        if datasets == None:
            self.datasets = self.load_datasets()
        else:
            self.datasets = datasets

        if input_shape == None:
            self.input_shape = (28, 28, 1)
        else:
            self.input_shape = input_shape

    def load_datasets(self):
        try:
            datasets = {
                        'mnist': tf.keras.datasets.mnist,
                        'fashion_mnist': tf.keras.datasets.fashion_mnist,
                        'cifar10': tf.keras.datasets.cifar10
                    }
            return datasets
        except Exception as e:
            raise e

    # Unified Model Builder
    def create_model(self, act_config, input_shape=(28, 28, 1)):
        try:
            layers = []

            layers.append(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
            layers.append(self.act_config if isinstance(act_config, tf.keras.layers.Layer) else tf.keras.layers.Activation(act_config))
            
            layers.append(tf.keras.layers.MaxPooling2D(2, 2))
            layers.append(tf.keras.layers.Conv2D(64, (3, 3)))
            layers.append(act_config if isinstance(act_config, tf.keras.layers.Layer) else tf.keras.layers.Activation(act_config))
            
            layers.append(tf.keras.layers.MaxPooling2D(2, 2))
            layers.append(tf.keras.layers.Flatten())
            layers.append(tf.keras.layers.Dense(64))
            layers.append(act_config if isinstance(act_config, tf.keras.layers.Layer) else tf.keras.layers.Activation(act_config))
            
            # Output layer - always softmax for classification
            layers.append(tf.keras.layers.Dense(10, activation='softmax'))

            model = tf.keras.Sequential(layers)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            raise e

    def train_model(self):
        try:
            # Dataset Loop with TensorBoard Logging
            for dataset_name, loader in self.datasets.items():
                (x_train, y_train), (x_test, y_test) = loader.load_data()

                if dataset_name == 'cifar10':
                    x_train, x_test = x_train / 255.0, x_test / 255.0
                else:
                    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
                    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

                for act_name, act_config in self.act_config.items():
                    print(f"Training on {dataset_name} with {act_name}")

                    model = self.create_model(act_config, input_shape=x_train.shape[1:])
                    log_dir = f"logs/{dataset_name}/{act_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                                histogram_freq=1, 
                                                                write_graph=True,
                                                                write_images=True,
                                                                update_freq='epoch')

                    model.fit(x_train, y_train, epochs=10, batch_size=64,
                            validation_data=(x_test, y_test),
                            callbacks=[tb_callback], verbose=2)
        except Exception as e:
            raise e