# main.py only contain main() function and running function
# read each class from root file
# class_xx_xxx is the file name that first part state the file type is class, second is sequence
from class_31_hyperparameters import HyperParamters
from class_32_import import ImportData
from class_33_generator import ImageGen
from class_59_compile import CompileAndFit
from class_51_baseline import BaseLineModel

def main():
    """
    We use this function to call each seperate module on sequence
    """
    # ************** Import Data******************************
    # If i complete this preprocess it should not run again, in the future, we will invole auto decide code
    import_class = ImportData()
    # class import_data() function
    (train_dir, valid_dir, test_dir)= import_class.import_data

    # **********************Data Generator
    generator_class = ImageGen()
    train_generator, valid_generator= generator_class.image_gen(train_dir, valid_dir, test_dir)

    # ***************create baseline model***********************
    base_class = BaseLineModel()
    model_1 = base_class.baseline_cnn()

    # ****************compile and fit*******************
    compile_class = CompileAndFit()
    history_1 = compile_class.compile(model_1, train_generator, valid_generator)

    return (train_dir, valid_dir, test_dir, model_1, history_1)



if __name__ == '__main__':
    """
    """
    (train_dir, valid_dir, test_dir, model_1, history_1) = main()
    print('over')
