from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers.experimental.preprocessing import StringLookup
from os import listdir

import pickle
import numpy as np
import os
import re

import tensorflow as tf

tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)


def check_recognition(dir_name):
    np.random.seed(42)
    tf.random.set_seed(42)

    base_path = "./"
    base_image_path = os.path.join(base_path, dir_name)

    t_images = []

    for f in listdir(base_image_path):
        t_images_path = os.path.join(base_image_path, f)
        t_images.append(t_images_path)

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    t_images.sort(key=natural_keys)

    with open("./characters", "rb") as fp:  # Unpickling
        b = pickle.load(fp)

    AUTOTUNE = tf.data.AUTOTUNE

    char_to_num = StringLookup(vocabulary=b, mask_token=None)

    num_to_chars = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

    batch_size = 64
    padding_token = 99
    image_width = 128
    image_height = 32
    max_len = 21

    def distortion_free_resize(image, img_size):
        w, h = img_size
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]

        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        image = tf.pad(
            image, paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )
        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.image.flip_left_right(image)
        return image

    def preprocess_image(image_path, img_size=(image_width, image_height)):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, 1)
        image = distortion_free_resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def process_images(image_path):
        image = preprocess_image(image_path)
        return {"image": image}

    def prepare_test_images(image_paths):
        dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(
            process_images, num_parallel_calls=AUTOTUNE
        )

        return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

    inf_images = prepare_test_images(t_images)

    class CTCLayer(keras.layers.Layer):

        def __init__(self, name=None):
            super().__init__(name=name)
            self.loss_fn = keras.backend.ctc_batch_cost

        def call(self, y_true, y_pred):
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            loss = self.loss_fn(y_true, y_pred, input_length, label_length)
            self.add_loss(loss)

            return y_pred

    def build_model():
        input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
        labels = keras.layers.Input(name="label", shape=(None,))

        x = keras.layers.Conv2D(
            32, (3, 3), activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1"
        )(input_img)
        x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

        x = keras.layers.Conv2D(
            64, (3, 3), activation="relu", kernel_initializer="he_normal",
            padding="same",
            name="Conv2"
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

        new_shape = ((image_width // 4), (image_height // 4) * 64)
        x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
        x = keras.layers.Dropout(0.2)(x)

        x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        x = keras.layers.Dense(len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2")(x)
        output = CTCLayer(name="ctc_loss")(labels, x)

        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
        )

        opt = keras.optimizers.Adam()
        model.compile(optimizer=opt)
        return model

    model = build_model()

    custom_objects = {"CTCLayer": CTCLayer}

    reconstructed_model = keras.models.load_model("./ocr_model_50_epoch.h5", custom_objects=custom_objects)
    prediction_model = keras.models.Model(reconstructed_model.get_layer(name="image").input,
                                          reconstructed_model.get_layer(name="dense2").output)

    pred_test_text = []

    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_len]

        output_text = []

        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(num_to_chars(res)).numpy().decode("utf-8")
            output_text.append(res)

        return output_text

    for batch in inf_images.take(3):
        batch_images = batch["image"]

        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)
        pred_test_text.append(pred_texts)

        for i in range(16):
            img = batch_images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")

    flat_list = [item for sublist in pred_test_text for item in sublist]

    sentence = ' '.join(flat_list)
    return sentence
