#!/usr/bin/env python

##############
#### Your name: Brian Zhu
##############

import numpy as np
import re
import cv2 as cv
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
from skimage.feature import hog
from skimage.measure import LineModelND, ransac
import matplotlib.pyplot as plt



import ransac_score

class ImageClassifier:
    
    def __init__(self):
        self.classifier = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)
        
        #create one large array of image data
        data = io.concatenate_images(ic)
        
        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]
        
        return(data,labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        ######## YOUR CODE HERE
        ########################
        feature_data = []

        for img in data:
            preprocessed_img = color.rgb2gray(img)
            preprocessed_img = filters.gaussian(preprocessed_img)
            feature_data.append(hog(preprocessed_img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(3, 3), block_norm='L2-Hys'))

        # Please do not modify the return type below

        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above
        
        # train model and save the trained model to self.classifier
        
        ########################
        ######## YOUR CODE HERE
        ########################
        self.classifier = svm.LinearSVC()
        self.classifier.fit(train_data, train_labels)

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels
        
        ########################
        ######## YOUR CODE HERE
        ########################
        predicted_labels = self.classifier.predict(data)
        # Please do not modify the return type below
        return predicted_labels

    def line_fitting(self, data):
        # Please do not modify the header

        # fit a line the to arena wall using RANSAC
        # return two lists containing slopes and y intercepts of the line

        ########################
        ######## YOUR CODE HERE
        ########################
        slope = []
        intercept = []
        #preprocessed_img = filters.gaussian(data)
        preprocessed_img = [np.argwhere(auto_canny(img)) for img in data]
        #i = 0
        for img in preprocessed_img:
            model, inliers = ransac(img, LineModelND, min_samples=2, residual_threshold=1, max_trials=1000)

            origin, direction = model.params
            slope.append(direction[0]/direction[1])
            intercept.append(origin[0] - direction[0]/direction[1] * origin[1])

            # plt.imshow(data[i], cmap=plt.cm.gray)
            # line_x = np.arange(0, 320)
            # line_y_robust = model.predict_x(line_x)
            # plt.plot(line_y_robust, color='#ff0000', linewidth=1.5)
            # plt.show()
            # i+=1



        # Please do not modify the return type below
        return slope, intercept

def auto_canny(image, sigma = 0.75):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv.Canny(image, lower, upper)

def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    (wall_raw, _) = img_clf.load_data_from_folder('./wall/')
    
    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)
    
    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))
    
    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))

    # ransac
    print("\nRANSAC results")
    print("=============================")
    s, i = img_clf.line_fitting(wall_raw)
    print(f"Line Fitting Score: {ransac_score.score(s,i)}/10")

if __name__ == "__main__":
    main()
