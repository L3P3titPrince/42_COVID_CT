from tensorflow.keras.preprocessing.image import ImageDataGenerator
# we need use this function to display processed image result and its resulotion
import matplotlib.pyplot as plt

from class_31_hyperparameters import HyperParamters

class ImageGen(HyperParamters):
    """:arg
    After we split image data into its train/test/validation folder and provide them categorical floder,
    Images are still raw data which can not be directly read by our model. We need transform them to binary
    and import into model first import layer.

    data should be formatted into appropriately pre-processed floating point tensors before being fed into our network.
    Currently, our data sits on a drive as PNG OR JPG files, so the steps for getting it into our network are roughly:
    1.read the picture files
    2.decode the PNG and JPG content to RGB grid of pixels
    3.convert these into floating point tensors
    4.rescale pixel values (between 0 and 255) to the [0,1] interval
    (as you know, neural networks prefer to deal with small input values).

    It may seem a bit daunting, but thankfully Keras has utilities to take care of these steps automatically.
    Keras has a module with image processing helper tools, located at keras.preprocessing.image.
    In particular, it contains the class ImageDataGenerator which allows to quickly set up Python generators
    that can automatically turn image files on disk into batches of pre-processed tensors.
    This is what we will use here.


    """

    def __init__(self):
        """:arg

        """
        HyperParamters.__init__(self)

    def image_gen(self, train_dir, valid_dir, test_dir):
        """

        Argus:
        ------


        """
        # all images will be rescaled by 1./255, which means every images will be divied by 255
        # in this function, we only use ImageDataGenerator() a small piece function: rescale
        # in Data Augmentation we will use more arguments in ImageDataGenerator() function
        train_datagen = ImageDataGenerator(rescale = 1./255)
        test_datagen = ImageDataGenerator(rescale = 1./ 255)
        # due to our images restored in folders, so we use flow_from_directory()
        print(train_dir)
        train_generator = train_datagen.flow_from_directory(
            # this is target directory
            train_dir,
            # all image will be resized to 150*150
            target_size=(150, 150),
            # set color to grey mode
            color_mode = 'grayscale',
            # set class name
            # classes = ['COVID', 'NonCOVID']
            class_mode = 'binary',
            # set tensor batch size
            batch_size=20
        )

        # ************test model****************
        for data_batch, labels_batch in train_generator:
            print('data batch size is:', data_batch.shape)
            print('label batch size is :', labels_batch.shape)
            plt.imshow(data_batch[8])
            plt.show()
            break

        valid_generator = train_datagen.flow_from_directory(
            # this is target directory
            train_dir,
            # all image will be resized to 150*150
            target_size=(150, 150),
            # set color to grey mode
            color_mode = 'grayscale',
            # set class name
            # classes = ['COVID', 'NonCOVID']
            class_mode = 'binary',
            # set tensor batch size
            batch_size=20
        )

        return train_generator, valid_generator

