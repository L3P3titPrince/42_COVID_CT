# record running time
from time import time
# need spcify lr optimizer
from tensorflow.keras import optimizers
# need use identify loss function
from tensorflow.keras import losses
# for classfication problem, we can use lots of metric to evaluate result
from tensorflow.keras import metrics


from class_31_hyperparameters import HyperParamters

class CompileAndFit(HyperParamters):
    """:arg
    """

    def __init__(self):
        """:arg
        """
        HyperParamters.__init__(self)


    def compile(self, model, train_generator, valid_generator):
        """:arg
        This function contain model compile and model fit process, input a model and output history and trained model

        """
        start_time = time()
        print("*" * 40, "Start {} Processing".format(model._name), "*" * 40)

        # we use a lot of metric to evalute our binary classification result
        METRICS = [
              metrics.TruePositives(name='tp'),
              metrics.FalsePositives(name='fp'),
              metrics.TrueNegatives(name='tn'),
              metrics.FalseNegatives(name='fn'),
              metrics.BinaryAccuracy(name='binary_accuracy'),
              #metrics.CategoricalAccuracy(name='accuracy'),
              metrics.Precision(name='precision'),
              metrics.Recall(name='recall'),
              metrics.AUC(name='auc'),
              # F1Score(num_classes = int(y_train.shape[1]), name='F1')
        ]

        # define a optimizer
        opt_rms = optimizers.RMSprop(lr = 1e-4, decay = 1e-5)
        # define compile parameters
        model.compile(loss = 'binary_crossentropy', optimizer = opt_rms, metrics = ['accuracy'])
        # start to fit
        history = model.fit(
            train_generator,
            steps_per_epoch=20,
            epochs=5,
            validation_data=valid_generator,
            validation_steps=20
        )

        return history
