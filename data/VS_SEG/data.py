import csv
import os


def load_data(opt):
    logger = opt.logger

    train_files, val_files, test_files = ([], []), [], []

    split_file = opt.split
    if not os.path.isfile(split_file) and not os.path.isabs(split_file):
        split_file = os.path.join(os.path.dirname(__file__), split_file)  # set path relative to current module path

    with open(split_file) as csvfile:
        csvReader = csv.reader(csvfile)

        for row in csvReader:
            if row[1] == "T1":
                image_name = os.path.join(opt.data_folder, row[0], 'vs_gk_t1_refT1.nii.gz')
                label_name = os.path.join(opt.data_folder, row[0], 'mask_T1.nii.gz')
                domain = 0
            elif row[1] == "T2":
                image_name = os.path.join(opt.data_folder, row[0], 'vs_gk_t2_refT2.nii.gz')
                label_name = os.path.join(opt.data_folder, row[0], 'mask_T2.nii.gz')
                domain = 1
            if row[2] == "training":
                train_files[domain].append({"image": image_name, "label": label_name})
            elif row[2] == "validation":
                val_files.append({"image": image_name, "label": label_name})
            elif row[2] == "test":
                test_files.append({"image": image_name, "label": label_name})

    # check if all files exist
    for file_dict in train_files[0] + train_files[1] + val_files + test_files:
        assert (os.path.isfile(file_dict['image'])), f" {file_dict['image']} is not a file"
        assert (os.path.isfile(file_dict['label'])), f" {file_dict['label']} is not a file"

    logger.info("Number of images in training (source)  = {}".format(len(train_files[0])))
    logger.info("Number of images in training (target)  = {}".format(len(train_files[1])))
    logger.info("Number of images in validation         = {}".format(len(val_files)))
    logger.info("Number of images in test set           = {}".format(len(test_files)))
    if opt.debug:
        logger.info("training set (source)  = {}".format(train_files[0]))
        logger.info("training set (target)  = {}".format(train_files[1]))
        logger.info("validation set         = {}".format(val_files))
        logger.info("test set               = {}".format(test_files))

    # return as dictionaries of image/label pairs
    return train_files, val_files, test_files
