# from this project 03_data only used for small cache, big file recommend read D:\03_data

import class_31_hyperparameters
import pandas as pd
# measure running time
from time import time
# access operation system directory
import os
# used to copy file by operation directory
import shutil


class ImportData(object):
    """
    This data set come from https://github.com/UCSD-AI4H/COVID-CT/
    For phase 1 we only have 349 + 397 images
    For phase 2, i will import more images from open data sources.

    """

    def __init__(self):
        """
        Nothing
        """
        pass

    @property
    def import_data(self):
        """
        1.create train, test, and validataion folder
        2.put image into category folder

        Args:
        -------
        path:string
            directory of file we use the


        Returns:
        --------
        train_covid_dir:str
            the directory of train/COVID/*.png
        valid_covid_dir:str
            the directory of validation/COVID/*.png *.jpg

        """
        print("*" * 50, "32 Start import_data", "*" * 50)
        start_time = time()

        # root directory, typicall we use current 03_data as our source,
        # but sometimes, we need to access outside data source
        # current base_dir is this project root directory
        base_dir = r'D:\OneDrive\03_Academic\23_Github\42_COVID_CT\03_data'

        # ***********Seperate version***************
        # because we don't have image label on each image, just like numerical file,
        # so we need create file structure to provide extra image label information
        train_dir = os.path.join(base_dir, 'train')
        # if this folder is exist, os.path.isdir=True. (not os.path.isdir)=False, excute 'else print('exist')
        # if this floder is not exist, os.path.idir=False (not os.path.isdir)=True, excute forward os.mkdir
        os.mkdir(train_dir) if not os.path.isdir(train_dir) else print('%s has exist' % train_dir)
        # test part
        test_dir = os.path.join(base_dir, 'test')
        # create test folder
        os.mkdir(test_dir) if not os.path.isdir(test_dir) else print('%s has exist' % test_dir)
        # validation part
        valid_dir = os.path.join(base_dir, 'validation')
        # create validation folder
        os.mkdir(valid_dir) if not os.path.isdir(valid_dir) else print('%s has exist' % valid_dir)

        # *************Create COVID and NonCOVID in each train/test/validation folder
        train_covid_dir = os.path.join(train_dir, 'COVID')
        os.mkdir(train_covid_dir) if not os.path.isdir(train_covid_dir) else print('%s has exist' % train_covid_dir)
        train_non_dir = os.path.join(train_dir, 'NonCOVID')
        os.mkdir(train_non_dir) if not os.path.isdir(train_non_dir) else print('%s has exist' % train_non_dir)
        test_covid_dir = os.path.join(test_dir, 'COVID')
        os.mkdir(test_covid_dir) if not os.path.isdir(test_covid_dir) else print('%s has exist' % test_covid_dir)
        test_non_dir = os.path.join(test_dir, 'NonCOVID')
        os.mkdir(test_non_dir) if not os.path.isdir(test_non_dir) else print('%s has exist' % test_non_dir)
        valid_covid_dir = os.path.join(valid_dir, 'COVID')
        os.mkdir(valid_covid_dir) if not os.path.isdir(valid_covid_dir) else print('%s has exist' % valid_covid_dir)
        valid_non_dir = os.path.join(valid_dir, 'NonCOVID')
        os.mkdir(valid_non_dir) if not os.path.isdir(valid_non_dir) else print('%s has exist' % valid_non_dir)

        # ****************Copy file to categorical folder
        # set up the constant ratio number for each category
        # train set will contain 60%
        TRAIN_RATIO = 0.6
        # validation set will contain 20%
        VALID_RATIO = 0.2
        # test set will occupy 20%
        TEST_RATIO = 0.2
        # orginal file location, we can set the variable as new resources
        covid_dir = r'D:\OneDrive\03_Academic\23_Github\42_COVID_CT\03_data\29_CT_COVID'
        non_dir = r'D:\OneDrive\03_Academic\23_Github\42_COVID_CT\03_data\30_CT_NonCOVID'
        # how many image files contains in COVID folder
        COVID_LEN = len(os.listdir(covid_dir))
        # how many image files contain in NonCOVID folder
        NON_LEN = len(os.listdir(non_dir))
        # train set will contain how many COVID image
        TRAIN_COVID_END = int(COVID_LEN * TRAIN_RATIO)
        # validation set in COVID images
        VALID_COVID_END = int(COVID_LEN * TRAIN_RATIO) + int(COVID_LEN * VALID_RATIO)
        # train set will contain how many COVID image
        TRAIN_NON_END = int(NON_LEN * TRAIN_RATIO)
        # validation set in COVID images
        VALID_NON_END = int(NON_LEN * TRAIN_RATIO) + int(NON_LEN * VALID_RATIO)


        # one line if condintal sentence actually like a operator action, for instance a = (1 if True else 2)
        # so, we can not use a+=1 to accomplish accumulation. But list function is working
        # we can create a empty list and append() elements and then count elements total number to get
        # how many times this list.append() has been called
        count_list = []
        # iterate files in original dataset covide_dir and select first TRAIN_COVID_END images
        for dirpath, dirnames, filenames in os.walk(covid_dir):
            # iterate all *.png files
            for filename in [f for f in filenames if f.endswith('.png')][:TRAIN_COVID_END]:
                # source is COVID orginal files
                src = os.path.join(covid_dir, filename)
                # destination is train -> COVID
                dst = os.path.join(train_covid_dir, filename)
                shutil.copyfile(src, dst) if not os.path.exists(dst) else count_list.append(dst)
        # iteration file for validation set
        for dirpath, dirnames, filenames in os.walk(covid_dir):
            for filename in [f for f in filenames if f.endswith('.png')][TRAIN_COVID_END:VALID_COVID_END]:
                src = os.path.join(covid_dir, filename)
                dst = os.path.join(valid_covid_dir, filename)
                shutil.copyfile(src, dst) if not os.path.exists(dst) else count_list.append(dst)
        # iteration file for validation set
        for dirpath, dirnames, filenames in os.walk(covid_dir):
            for filename in [f for f in filenames if f.endswith('.png')][VALID_COVID_END:]:
                src = os.path.join(covid_dir, filename)
                dst = os.path.join(test_covid_dir, filename)
                shutil.copyfile(src, dst) if not os.path.exists(dst) else count_list.append(dst)


        # iterate files in original dataset covide_dir and select first TRAIN_COVID_END images
        for dirpath, dirnames, filenames in os.walk(non_dir):
            # iterate all *.png and *.jpg files
            for filename in [f for f in filenames if f.endswith(('.png','.jpg'))][:TRAIN_NON_END]:
                # source is COVID orginal files
                src = os.path.join(non_dir, filename)
                # destination is train -> COVID
                dst = os.path.join(train_non_dir, filename)
                shutil.copyfile(src, dst) if not os.path.exists(dst) else count_list.append(dst)
        # iteration file for validation set
        for dirpath, dirnames, filenames in os.walk(non_dir):
            for filename in [f for f in filenames if f.endswith(('.png','.jpg'))][TRAIN_NON_END:VALID_NON_END]:
                src = os.path.join(non_dir, filename)
                dst = os.path.join(valid_non_dir, filename)
                shutil.copyfile(src, dst) if not os.path.exists(dst) else count_list.append(dst)
        # iteration file for validation set
        for dirpath, dirnames, filenames in os.walk(non_dir):
            for filename in [f for f in filenames if f.endswith(('.png','.jpg'))][VALID_NON_END:]:
                src = os.path.join(non_dir, filename)
                dst = os.path.join(test_non_dir, filename)
                shutil.copyfile(src, dst) if not os.path.exists(dst) else count_list.append(dst)

        print('There are %s already exist' % len(count_list))
        print('total training COVID images:', len(os.listdir(train_covid_dir)))
        print('total validation COVID images:', len(os.listdir(valid_covid_dir)))
        print('total testing COVID images:', len(os.listdir(test_covid_dir)))
        print('total training NonCOVID images:', len(os.listdir(train_non_dir)))
        print('total validation NonCOVID images:', len(os.listdir(valid_non_dir)))
        print('total testing NonCOVID images:', len(os.listdir(test_non_dir)))





        """
        # we use a loop to create three different folder at once
        for x in ['train', 'test', 'validation']:
            # combine x and base_dir to create a new path
            path_dir = os.path.join(base_dir, x)
            # if this folder is exist, os.path.isdir=True. (not os.path.isdir)=False, excute 'else print('exist')
            # if this folder is not exist, os.path.isdir=False (not os.path.isdir)=True, excute forward os.mkdir
            os.mkdir(path_dir) if not os.path.isdir(path_dir) else print('%s has exist' %path_dir)
            # in train/test/validation each folder, they all have COVID/NonCOVID two categories seperate folder
            for y in ['COVID', 'NonCOVID']:
                path_cate_dir = os.path.join(path_dir, y)
                if not os.path.isdir(path_cate_dir):
                    os.mkdir(path_cate_dir)
                    src = os.path.join()

                else:
                    print('%s has exist and contain %d' %(path_cate_dir, len(os.listdir(path_cate_dir))))


        # second step, we split COVID and NonCOVID into train/test/valication folder
        covid_dir = r'D:\OneDrive\03_Academic\23_Github\42_COVID_CT\03_data\29_CT_COVID'
        non_dir = r'D:\OneDrive\03_Academic\23_Github\42_COVID_CT\03_data\30_CT_NonCOVID'


        # set up the constant ratio number for each category
        TRAIN_RATIO = 0.6
        VALID_RATIO = 0.2
        TEST_RATIO = 0.2
        # get the total image number of COVID and NonCOVID folder
        COVID_LEN = len(os.listdir(covid_dir))
        NON_LEN = len(os.listdir(non_dir))
        for dirpath, dirnames, filenames in os.walk(covid_dir):
            for filename in [f for f in filenames if f.endswith('.png')][:COVID_LEN*TRAIN_RATIO]:
                src = os.path.join(covid_dir, filename)
                dst = os.path.join()
        """

        """
        path_covid_dir = os.path.join(path_dir, 'COVID')
        path_noncovid_dir = os.path.join(path_dir, 'NonCOVID')
        if not os.path.isdir(path_covid_dir):
            # if not exist, create folder
            os.mkdir(path_dir)
        else:
            print('%s has exist and contain %d' %(path_dir, len(os.listdir(path_dir))))
        """

        cost_time = round((time() - start_time), 4)
        print("*" * 40, "End import_data() with {} second".format(cost_time), "*" * 40, end='\n\n')

        return (train_dir, valid_dir, test_dir)
