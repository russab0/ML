import cv2
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

from time import time  # measuring time since start
from shutil import rmtree  # remove directory with its files
from os import mkdir, listdir  # viewing files inside folder, creating folder
from skimage.transform import rotate  # image rotation to given anglle
from skimage.util import random_noise  # adding random noise to the image
from sklearn.ensemble import RandomForestClassifier  # classifier
from sklearn.metrics import classification_report, accuracy_score  # metrics


def read_traffic_signs(rootpath, dataset_type=None):
    """Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels"""
    assert dataset_type is not None, "Dataset is not specified"
    images = []  # images
    labels = []  # corresponding labels

    for c in range(0, 43):  # loop over all 42 classes
        if dataset_type == "training":
            prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
            gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        else:
            prefix = rootpath + '/'  # subdirectory for class
            gtFile = open(prefix + 'GT-final_test.csv')  # annotations file

        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header

        for row in gtReader:  # loop over all images in current annotations file
            images.append(plt.imread(prefix + row[0]).astype('uint8'))  # the 1th column is the filename
            labels.append(int(row[7]))  # the 8th column is the label
        gtFile.close()
    return images, labels


def make_square(image):
    """
    Squares the given image by adding black pixels (so that H and W will be the same)
    :param image: image to square
    :return: squared image
    """
    h, w, c = image.shape
    if h == w:
        return image
    if h < w:  # vertical
        appendix = np.zeros((w - h, w, 3), dtype='uint8')
        return np.concatenate((image, appendix), axis=0)
    if h > w:  # horizontal
        appendix = np.zeros((h, h - w, 3), dtype='uint8')
        return np.concatenate((image, appendix), axis=1)


def resize(images, image_size, draw=False):
    """
    Resizes given images `in place`
    :param images: images to resize
    :param image_size: integer value, the size of the image
    :param draw: whether to draw resized images or not
    :return: None
    """
    for i in range(len(images)):
        squared = make_square(images[i])
        assert squared.shape[0] == squared.shape[1]

        if draw:
            plt.imshow(squared)
            plt.savefig(f"trash/squared-{i}")
            plt.clf()

        images[i] = cv2.resize(squared, (image_size, image_size))
        assert images[i].shape[0] == images[i].shape[1]

        if draw:
            plt.imshow(images[i])
            plt.savefig(f'trash/resized-{i}')
            plt.clf()


def shuffle_dataset(datasetX, datasetY):
    """
    Shuffles the given dataset
    :param datasetX: responses of the dataset
    :param datasetY: targets of the dataset
    :return: shuffled responses and targets
    """
    arr = list(zip(datasetX, datasetY))
    random.shuffle(arr)
    datasetX = [elem[0] for elem in arr]
    datasetY = [elem[1] for elem in arr]
    arr.clear()
    return datasetX, datasetY


def train_valid_split(datasetX, datasetY):
    """
    Splits the dataset to training and validation parts in propotion 80/20
    :param datasetX: responses of the dataset
    :param datasetY: targets of the dataset
    :return: tuple(trainX, trainY, validX, validY) - responses and targets of training and validation sets
    """
    assert len(datasetX) == len(datasetY)
    trainX, trainY, validX, validY = [list() for _ in range(4)]
    cls = 0
    y_prev = -1
    i = 0
    while i < len(datasetX):
        y = datasetY[i]
        if y != y_prev:
            cls += 1
        y_prev = y

        are_going_to_train = random.random() < 0.8
        for j in range(30):
            if are_going_to_train:
                trainX.append(datasetX[i])
                trainY.append(datasetY[i])
            else:
                validX.append(datasetX[i])
                validY.append(datasetY[i])
            i += 1

    trainX, trainY = shuffle_dataset(trainX, trainY)
    validX, validY = shuffle_dataset(validX, validY)
    assert len(trainX) == len(trainY)
    assert len(validX) == len(validY)
    assert len(trainX) + len(validX) == len(datasetX)
    assert type(trainX) == type(validX) == type(list())
    datasetX.clear()
    datasetY.clear()
    return trainX, trainY, validX, validY


def classes_distribution(datasetY):
    """
    Returns frequesncies of each class in the dataset
    :param datasetY: lables of dataset
    :return: list of frequencies
    """
    cnt = [0] * 43
    for label in datasetY:
        cnt[label] += 1
    return cnt


def plot_classes_distribution(datasetY, filename):
    """
    Plots the classes distribution frequencies
    :param datasetY: labels of dataset
    :param filename: filename to save picture
    :return: None
    """
    cnt = classes_distribution(datasetY)
    plt.bar(list(range(43)), cnt)
    plt.xlabel("Class")
    plt.ylabel("Examples")
    plt.title("Histogram of 43 classes with their number of examples")
    plt.savefig(f"output/{filename}")
    plt.clf()


def transform(image):
    angle = random.randint(-25, 25)
    image = rotate(image, angle)
    if random.random() > 0.5:
        image = random_noise(image)
    x = random.random()
    if x < 1 / 3:
        image = image * random.uniform(1, 3)
    elif x < 2 / 3:
        image = image / random.uniform(1, 3)
    return image


def augment(datasetX, datasetY, draw=False):
    cnt = classes_distribution(datasetY)
    max_cnt = max(cnt)

    while sum(cnt) != 43 * max_cnt:
        for i in range(len(datasetX)):
            x = datasetX[i]
            y = datasetY[i]

            if cnt[y] < max_cnt:
                new_x, new_y = transform(x), y
                datasetX.append(new_x)
                datasetY.append(new_y)
                cnt[y] += 1
                if draw:
                    plot_images([x, new_x], ["original", "augmented"], f"augmented-{i}.png")

    datasetX, datasetY = shuffle_dataset(datasetX, datasetY)
    return datasetX, datasetY


def plot_images(imgs, names=None, filename=None):
    fig, axs = plt.subplots(ncols=len(imgs), figsize=(16, 8))
    for i, ax in enumerate(axs):
        ax.imshow(imgs[i])
        if names and i < len(names):
            ax.set_title(names[i], fontsize=15)
    plt.show()
    if filename:
        plt.savefig(filename)


def normalize_and_flatten(imgs):
    for i in range(len(imgs)):
        imgs[i] = imgs[i].flatten() / 255


def train_model(trainX, trainY, validX, validY):
    parameters = {'max_depth': [10, 20],  # + 30 (25, 30, 35),
                  'n_estimators': [10, 20],  # + 50 (45, 50, 55)
                  }

    best_score = best_max_depth = best_n_estimators = 0
    best_model = None

    for max_depth in parameters['max_depth']:
        for n_estimators in parameters['n_estimators']:
            model = RandomForestClassifier(
                max_depth=max_depth,
                n_estimators=n_estimators,
                n_jobs=-1
            )
            model.fit(trainX, trainY)

            valid_pred = model.predict(validX)
            cur_score = accuracy_score(validY, valid_pred)
            if cur_score > best_score:
                best_score = cur_score
                best_max_depth = max_depth
                best_n_estimators = n_estimators
                best_model = model
            print(f"Params: {max_depth}, {n_estimators}. Score: {cur_score}")

    print("best_params_set", best_max_depth, best_n_estimators)
    return best_model


def main(do_augm, image_size):
    if 'trash' in listdir():
        rmtree('trash')
    mkdir('trash')

    trainX, trainY = read_traffic_signs('./data/Final_Training/Images', 'training')
    resize(trainX, image_size)

    trainX, trainY, validX, validY = train_valid_split(trainX, trainY)
    plot_classes_distribution(trainY, "initial_train.png")
    plot_classes_distribution(validY, "initial_valid.png")

    if do_augm:
        trainX, trainY = augment(trainX, trainY, draw=False)
        plot_classes_distribution(trainY, "augmented_train.png")

    normalize_and_flatten(trainX)
    normalize_and_flatten(validX)

    model = train_model(trainX, trainY, validX, validY)
    valid_pred = model.predict(validX)
    print("valid", accuracy_score(validY, valid_pred))

    validX.clear()
    validY.clear()
    trainX.clear()
    trainY.clear()

    print("started testing")
    testX, testY = read_traffic_signs('./data/Final_Test/Images', 'testing')
    resize(testX, image_size)
    normalize_and_flatten(testX)
    test_pred = model.predict(testX)
    for i in range(len(test_pred)):
        if test_pred[i] != testY[i]:
            plot_images([test_pred[i]], f"correct: {testY[i]}\npredicted: {test_pred[i]}", f"misclassified-{i}.png")
    print("test", accuracy_score(testY, test_pred))
    print(classification_report(testY, test_pred))
    testX.clear()
    testY.clear()


for do_augm in [True]:
    for image_size in [30]:
        a = time()
        print(f"EXPREMENT: {do_augm}, {image_size}")
        main(do_augm, image_size)
        b = time()
        print("Execution took", b - a)
        print()
        print()
