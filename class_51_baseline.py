from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

from class_31_hyperparameters import HyperParamters

class BaseLineModel(HyperParamters):
    """:arg
    This is only baseline model

    """
    def __init__(self):
        """
        Inheritte from HyperParamters
        """
        # inherite
        HyperParamters.__init__(self)

    def baseline_cnn(self):
        """
        11 layer simple cnn model and end with binary classification dense layer

        """
        input_0 = layers.Input(shape=(150, 150, 1), name="input_0")
        # when we pu Conv2D as input layer, we need state input_shape
        # while the input images are not all 150*150, i guess we can choose a lower resuloution？
        # RGB color means thee channel
        # we use 32 3*3 size filters and, default stride is (1,1), defalut padding is 'valid'=no padding
        # output is (150-3)/1 + 1 =148, stride is 1 so denominaotr is one. Finally we got a new tensor (148, 148, 32)
        c2d_1 = layers.Conv2D(32, (3, 3), activation='relu', name='input_1')(input_0)
        # after Convlustoi layer, follow up a MaxPooling layer, output is (148/2=74) (74,74,32)
        pool_2 = layers.MaxPooling2D((2, 2), name="pool_2")(c2d_1)
        # next conv2D, output dimensino is (74-3)/1+1=72  (72,72,64)
        c2d_3 = layers.Conv2D(64, (3, 3), activation='relu', name='c2d_3')(pool_2)
        # after maxpooling, output is (72/2=36) (36,36,64)
        pool_4 = layers.MaxPooling2D(pool_size=(2, 2), name='pool_4')(c2d_3)
        # next conv2D, output dimension is (36-3)/1+1=34 (34,34,128)
        c2d_5 = layers.Conv2D(128, (3, 3), activation='relu', name='c2d_5')(pool_4)
        # after maxpooling, output is (34/2=17) (17,17,128)
        pool_6 = layers.MaxPooling2D(pool_size=(2, 2), name="pool_6")(c2d_5)
        # last conv2D, output dimension is (17-3)/1+1 = 15 (15,15,128)
        c2d_7 = layers.Conv2D(128, (3, 3), activation='relu', name="c2d_7")(pool_6)
        # last maxpooling, output is (15/2=7) (7,7,128)
        pool_8 = layers.MaxPooling2D(pool_size=(2, 2), name='pool_8')(c2d_7)
        # add a flatten layer, output is 7*7*128 = 6272
        flatten_9 = layers.Flatten()(pool_8)
        # add a dense layer
        dense_10 = layers.Dense(512, activation='relu', name='dense_10')(flatten_9)
        # add judgement layer
        dense_11 = layers.Dense(1, activation='sigmoid', name='dense_11')(dense_10)
        # construst into a model
        model = Model(inputs=input_0, outputs=dense_11)
        # display model summary
        model.summary()
        plot_model(model)

        return model