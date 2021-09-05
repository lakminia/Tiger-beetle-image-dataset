#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------
import h5py
import numpy as np
import os
import glob
import cv2
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

#--------------------
# tunable-parameters
#--------------------
num_trees = 25
test_size = 0.25
seed      = 9
train_path = "x/Training_data"
test_path  = "x/Testing_data"
h5_data    = 'output/data.h5'
h5_labels  = 'output/labels.h5'
scoring    = "accuracy"
fixed_size       = tuple((500, 500))
bins             = 8

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
class_names=train_labels
print(train_labels)

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=seed,multi_class='multinomial',solver='lbfgs')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 1)))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))
models.append(('Ensemble', ExtraTreesClassifier(n_estimators=10,max_features='auto')))

# variables to hold the results and names
results = []
names   = []

# import the feature vector and trained labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

#print(global_features)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)    
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #results = confusion_matrix(testDataGlobal, testLabelsGlobal)
    #print(results)
    # Plot non-normalized confusion matrix
    y_pred = model.fit(trainDataGlobal, trainLabelsGlobal).predict(testDataGlobal)  
    #plot_confusion_matrix(testLabelsGlobal, y_pred, classes=class_names,
    #                  title='Confusion matrix, without normalization')  
    print(msg)
    

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
#pyplot.show()





#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

'''
# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick
    '''

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()





# to visualize results
import matplotlib.pyplot as plt

# create the model - Random Forests
#clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
clf= LogisticRegression(random_state=seed,multi_class='multinomial',solver='lbfgs')# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)


# get the training labels
test_labels = os.listdir(test_path)

# sort the training labels
test_labels.sort()
print(test_labels)


actual_arr = []
predict_arr = []
f_n=0
# loop over the training data sub-folders
for testing_name in test_labels:
    
    
    # join the training data path and each species training folder
    dir = os.path.join(test_path, testing_name)
    #print('dddfdfdf'+testing_name)
    #print(dir)
    name="image_"
    # get the current training label
    current_label = testing_name
    i=1
    # loop over the images in each sub-folder
    #print(os.listdir(dir))
    # loop through the test images

    for x in os.listdir(dir):
        # read the image
        file=dir+"/"+ str(x) 
        image = cv2.imread(file)
        #print(file)
        # resize the image
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        #fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_hu_moments])

        # scale features in the range (0-1)
        scaler            = MinMaxScaler(feature_range=(0, 1))
        #rescaled_features = scaler.fit_transform(global_features)
        rescaled_feature = scaler.fit_transform(global_feature.reshape(-1,1))
        # predict label of test image
        prediction = clf.predict(rescaled_feature.reshape(1,-1))[0]
        print('f_n',f_n)
        print('pred',prediction)
        actual_arr.append(f_n)
        predict_arr.append(prediction)
        ''' 
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(clf, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)    
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        '''
    
   
        
        #print('Actual genus'+'---->'+'predicted genus')
        #print(testing_name+'---->'+train_labels[prediction])
        i += 1
                
    f_n += 1
    


print(accuracy_score(actual_arr, predict_arr))
#print(confusion_matrix(actual_arr, predict_arr))
plot_confusion_matrix(actual_arr, predict_arr, classes=class_names,
                    title='Confusion matrix, without normalization')
pyplot.show()

        # show predicted label on image
        #cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        # display the output image
        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #plt.show()
