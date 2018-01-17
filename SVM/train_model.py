#!/usr/bin/env python2
import glob
import os
import sys
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from common.image_transformation import ApplyingTransformationOnFrame

from common.config import get_config


def print_with_precision(num):
    return "%0.5f" % num


def read_images_transformed(images_transformed_path):
    print("\nReading the transformed images file located at path '{}'...".format(
        images_transformed_path))

    images = []
    labels = []
    with open(images_transformed_path) as images_transformed_file:
        reader = csv.reader(images_transformed_file, delimiter=',')
        for line in reader:
            label = line[0]
            labels.append(label)
            image = line[1:]
            image_int = [int(pixel) for pixel in image]
            image = np.array(image_int)
            images.append(image)

    # print("Done!\n")
    return images, labels


def divide_data_train_test(images, labels, ratio):
    print("\nDividing dataset in the ratio '{}' using `train_test_split()`:".format(ratio))
    ret = train_test_split(images, labels, test_size=ratio, random_state=0)
    # print("Done!\n")
    return ret


def main():
    model_name = "svm"
    model_output_dir_path = get_config('model_{}_output_dir_path'.format("svm"))
    model_stats_file_path = os.path.join(model_output_dir_path, "stats-{}.txt".format("svm"))
    print("Model stats will be written to the file at path '{}'.".format(model_stats_file_path))
    images_transformed_path = get_config('images_transformed_path')

    images_paths = glob.glob(os.path.join("../", "data", 'images', 'train', '*', '*'))
    images, labels = [], []
    # print("img path",images_paths)
    for img in images_paths:
        labels.append(img.split('\\')[-2])
        imgR = cv2.imread(img)
        imgTransformed = ApplyingTransformationOnFrame(imgR)
        imgTransformed = cv2.resize(imgTransformed, (200, 200))
        images.append(imgTransformed.flatten())

    labels = LabelEncoder().fit_transform(labels)
    images = np.asarray(images, dtype=np.float32)
    training_images, testing_images, training_labels, testing_labels = divide_data_train_test(
        images, labels, 0.3)
    grid = svm.SVC(kernel="linear", C=0.001)
    # param_grid = {"C": [0.001, 0.005,0.1, 1, 10, 100]}
    # grid = GridSearchCV(classifier_model, param_grid=param_grid, cv=10)
    svm_detector = grid.fit(training_images,training_labels)
    # c=grid.best_params_['C']
    # print (c)

    with open(model_stats_file_path, "w") as model_stats_file:
        # clfCV=cross_val_score(classifier_model,images,labels,cv=10)
        # print("mean is",np.mean(clfCV))

        # images = []
        print("\nTraining SVM.")
        # classifier_model = classifier_model.fit(testing_images, testing_labels)
        print("Done!\n")

        # model_serialized_path = get_config('model_{}_serialized_path'.format(model_name))
        # print("\nDumping the trained model to disk at path '{}'...".format(model_serialized_path))
        # joblib.dump(classifier_model, model_serialized_path)
        # print("Dumped\n")

        print("\nWriting model stats to file...")
        score = svm_detector.score(testing_images, testing_labels)
        print ("score",score)
        model_stats_file.write("Model score:\n{}\n\n".format(print_with_precision(score)))

        predicted = svm_detector.predict(testing_images).reshape(-1,1)

        print("test shape", testing_images.shape)
        print("pred shape", predicted.shape)

        # print("accuracy == ", accuracy_score(testing_images, predicted))
        print(confusion_matrix(testing_labels, predicted))
        print(classification_report(testing_labels, predicted, digits=3))

        plt.matshow(confusion_matrix(testing_labels, predicted), cmap=plt.cm.binary, interpolation='nearest')
        plt.title('confusion matrix')
        plt.colorbar()
        plt.ylabel('expected label')
        plt.xlabel('predicted label')
        plt.show()
        report = metrics.classification_report(testing_labels, predicted)
        model_stats_file.write(
            "Classification report:\n{}\n\n".format(report))
        print("Done!\n")

    print("\nFinished!\n")


if __name__ == '__main__':
    main()
