import tensorflow as tf


# https://github.com/bnsreenu/python_for_microscopists/blob/master/074-Defining%20U-net%20in%20Python%20using%20Keras.py


def create_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    kernel_init = "he_normal"
    padding = "same"
    activation = "relu"
    inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = tf.keras.layers.Conv2D(
        16,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(
        16,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(c5)

    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding=padding)(
        c5
    )
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding=padding)(
        c6
    )
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding=padding)(
        c7
    )
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding=padding)(
        c8
    )
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(
        16,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(
        16,
        (3, 3),
        activation=activation,
        kernel_initializer=kernel_init,
        padding=padding,
    )(c9)

    output = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[output])

    # model.summary()

    return model
