 #!/usr/bin/env python

import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
import ransac_score
import math

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
        print("data ")
        print("data shape = ", data.shape)
        l = []
        for im in data:
            im_gray = color.rgb2gray(im)

            im_gray = filters.gaussian(im_gray, sigma=1.2)

            f = feature.hog(im_gray, orientations=10, pixels_per_cell=(48, 48), cells_per_block=(2, 2), feature_vector=True, block_norm='L2-Hys')
            l.append(f)

        feature_data = np.array(l)
        print("feature array shape = ", feature_data.shape)
        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        self.classifier = svm.LinearSVC()
        self.classifier.fit(train_data, train_labels)

    def predict_labels(self, data):
        predicted_labels = self.classifier.predict(data)
        return predicted_labels

    def line_fitting(self, data):
        slope = []
        intercept = []

        for im in data:
            n = 2 # number of samples
            t = 0.1 # inlier threshold

            im_gray = color.rgb2gray(im)
            edges = feature.canny(im_gray, sigma=3)
            idx = np.argwhere(edges==True)
            idx = np.flip(idx, axis=1)

            # Implememtation using scikit ransac library
            ransac = linear_model.RANSACRegressor(min_samples=n, residual_threshold=t)
            X = idx[:,0]
            X = X.reshape(-1,1)
            y = idx[:,1]
            ransac.fit(X,y)

            line_X = np.arange(X.min(), X.max())[:, np.newaxis]
            line_y_ransac = ransac.predict(line_X)
            s, i = find_line([line_X[0,0], line_y_ransac[0]],[line_X[-1,0], line_y_ransac[-1]])

            slope.append(s)
            intercept.append(i)          

        return slope, intercept

    def find_line(self, p1, p2):
        # given two points find line corresponding to y=mx+c
        m = (p2[1]-p1[1])/(p2[0]-p1[0])
        c = p2[1] - m*p2[0]
        return m, c

    def find_dist(self, m, c, p):
        return abs(m*p[0]-p[1]+c)/math.sqrt(m**2 + 1)

def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    (wall_raw, _) = img_clf.load_data_from_folder('./wall/')

    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)
    print("test data shape ", train_data[0].shape)
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

    print("\nIncorrectly predicted")
    for i in range(len(test_labels)):
        if(test_labels[i] != predicted_labels[i]):
            print(i)

    # ransac
    print("\nRANSAC results")
    print("=============================")
    s, i = img_clf.line_fitting(wall_raw)
    print(f"Line Fitting Score: {ransac_score.score(s,i)}/10")

if __name__ == "__main__":
    main()
