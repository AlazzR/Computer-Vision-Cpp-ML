#%%
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import utilities #cv2, np
import numpy as np
import cv2
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, log_loss, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

#%%

train_validation_ratio = 0.85
trainingSize = 12500
train_size = int(trainingSize * train_validation_ratio)
validation_size = int(trainingSize * (1 - train_validation_ratio))
windowSize = 200
nfeatures = 256
numClusters = 64

obj = utilities.LoadingImages(trainingSize, numClusters, nfeatures)#30 clusters within an image

# Setting clusters early on
#numObservations = numClusters * 100
numObservations = 2048

images, labels = obj.loadDataRandomly(numObservations, windowSize)
des, labels = obj.getDescriptors(images, labels)
des, labels = obj.getBOW(np.array(des, np.float32), labels, True)

print(f"#clusters is {obj.bowExtractor.descriptorSize()}")
#%%
def train(obj, training_errors, validation_errors, epochs, batch_size, valid_size, batchs_num, validation_size, train_size, model_type, windowSize, decisionValue, noiseUpdate=0.05, nu=0.9, gamma=0.005, kernel="rbf", C=0.8):
    # cat = 0, dog = 1
    np.random.seed(42)
    if model_type == "svm":
        model = NuSVC(kernel=kernel, nu=nu, gamma=gamma)
        modelTemp = NuSVC(kernel=kernel, nu=nu, gamma=gamma)
    else:
        model = LogisticRegression(solver='sag', penalty="l2", C=C, warm_start=True, max_iter=400, n_jobs=2)#default parameters
        modelTemp = LogisticRegression(solver='sag', penalty="l2", C=C, warm_start=True, max_iter=400, n_jobs=2)#default parameters
    counter = 0
    prevError = np.finfo(np.float32).min

    for epoch in range(0, epochs):
        training_error_avg = []
        #Clear clusters at each epoch.
        obj.resetCounters()
        #obj.bowKmeansClusters.clear()
        #obj.bowExtractor.setVocabulary(np.array([]))
        for batch in range(0, batchs_num):
            #transform = PolynomialFeatures(degrees, interaction_only=True)
            des, labels = obj.loadDescriptorsAndLabels(int(batch_size/2), windowSize) #batch_sizex128
            des, labels = obj.getBOW(des, labels, False)
            #des = transform.fit_transform(des)
            indeces = np.arange(len(des))
            np.random.shuffle(indeces)
            if modelType == "svm":
                modelTemp.fit(des[indeces], labels[indeces]);
                y_pred = modelTemp.predict(des[indeces])
            else:
                modelTemp.fit(des[indeces], labels[indeces]); 
                #y_pred = model.predict(des)
                y_pred = list(map(lambda v: 0 if v[0] > decisionValue else 1, modelTemp.predict_proba(des[indeces])))
            try:
                cnf = confusion_matrix(y_pred, labels[indeces])
                training_error_avg.append(np.sum(np.diag(cnf))/np.sum(cnf))
                #training_error_avg.append(log_loss(labels[indeces], y_pred))
                print(f"{epoch}-{batch} the accuracy is {np.sum(np.diag(cnf))/np.sum(cnf)}")
                print(cnf)

            except Exception as e:
                print(e)
                pass

            if training_error_avg[-1] + np.random.normal(0) * noiseUpdate > prevError :
                print("Update model")
                model = modelTemp
                prevError = training_error_avg[-1]
            else:
                print("Previous model is better")
                counter += 1

            if counter > 15:
                print(f"No update for the parameter for {counter} passes")
                break

        if batchs_num * batch_size < train_size - 1 and counter <= 15:
        # transform = PolynomialFeatures(degrees, interaction_only=True)
            des, labels = obj.loadDescriptorsAndLabels(int(train_size - batchs_num * batch_size), windowSize)
            des, labels = obj.getBOW(des, labels, False)
            indeces = np.arange(len(des))
            np.random.shuffle(indeces)
            if modelType == "svm":    
                #des = transform.fit_transform(des)
                modelTemp.fit(des[indeces], labels[indeces]);
                y_pred = modelTemp.predict(des[indeces]) 
            else:
                modelTemp.fit(des[indeces], labels[indeces]); 
                y_pred = list(map(lambda v: 0 if v[0] > decisionValue else 1, modelTemp.predict_proba(des[indeces])))
                
            try:
                cnf = confusion_matrix(y_pred, labels[indeces])
                training_error_avg.append(np.sum(np.diag(cnf))/np.sum(cnf))
                print(f"Last accuracy is {np.sum(np.diag(cnf))/np.sum(cnf)}") 
                print(cnf)

            except Exception as e:
                    print(e) 
                    pass

            if training_error_avg[-1] > prevError:
                print("Update support Vectors")
                #model = modelTemp
                prevError = training_error_avg[-1]
        training_errors.append(np.mean(training_error_avg))

        #For validation set
        labels = []
        y_pred = []
        for valid_batch in range(0, int(validation_size/valid_size)):
            des, labels_r = obj.loadDescriptorsAndLabels(valid_size, windowSize) #numb_key_ptx128
            des, labels_r = obj.getBOW(des, labels_r, False)
            #transform = PolynomialFeatures(degrees, interaction_only=True)
            #des = transform.fit_transform(des[:])
            labels.extend(labels_r)
            if modelType == "logistic":
                y_pred_r = list(map(lambda v: 0 if v[0] > decisionValue else 1, model.predict_proba(des)))
            else:
                y_pred_r = model.predict(des)

            y_pred.extend(y_pred_r)
        if len(y_pred) < validation_size:
            des, labels_r = obj.loadDescriptorsAndLabels(valid_size, windowSize) #numb_key_ptx128
            des, labels_r = obj.getBOW(des, labels_r, False)
            #transform = PolynomialFeatures(degrees, interaction_only=True)
            #des = transform.fit_transform(des[:])
            labels.extend(labels_r)
            if modelType == "logistic":
                y_pred_r = list(map(lambda v: 0 if v[0] > decisionValue else 1, model.predict_proba(des)))
            else:
                y_pred_r = model.predict(des)
            y_pred.extend(y_pred_r)
        try:
            cnf = confusion_matrix(y_pred, labels)
            validation_errors.append(np.sum(np.diag(cnf))/np.sum(cnf))
            print(cnf)
            print(f"{epoch} the validation accuracy is {np.sum(np.diag(cnf))/np.sum(cnf)}") 

        except Exception as e:
            print(e)
            pass

        if epoch%2 == 0:
            print(f"#epoch: {epoch} and the accuracy training is {training_errors[epoch]}")    
            print(f"#epoch: {epoch} and the accuracy validation is {validation_errors[epoch]}")


    return model, training_errors, validation_errors

def prediction(obj, model, modelType, decisionValue, numImages, windowSize, datasetType, trainOrValid=True, visualize=False):
    batch_size = 256
    numBatches = int(numImages/batch_size)
    if datasetType == "train" and not trainOrValid:
        obj.counts["cat"] = train_size
        obj.counts["dog"] = train_size
    else:
        obj.resetCounters()

    labels = []
    y_pred = []
    counter = 1 
    print(obj.counts)
    for batch in range(0, numBatches):
        if datasetType == "train":
            des, labels_r = obj.loadDescriptorsAndLabels(batch_size, windowSize) #numb_key_ptx128
            des, labels_r = obj.getBOW(des, labels_r, False)
        else:
            images = obj.loadData(batch_size, windowSize, "test", "", "test", False, 1) #batch_sizex128
            des, labels_r = obj.getDescriptors(images, [-1]*batch_size, False)
            des, labels_r = obj.getBOW(des, [-1]*batch_size, False)
        labels.extend(labels_r)
        if modelType == "logistic":
            y_pred_r = list(map(lambda v: 0 if v[0] > decisionValue else 1, model.predict_proba(des)))
        else:
            y_pred_r = model.predict(des)
        y_pred.extend(y_pred_r)
        if visualize:
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.putText(img, 'cat' if y_pred_r[counter] == 0 else 'dog', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
                cv2.imshow("Test_image" + str(counter), img)
                counter += 1
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
    if len(y_pred) < numImages:
        if datasetType == "train":
            des, labels_r = obj.loadDescriptorsAndLabels(numImages - len(y_pred), windowSize) #numb_key_ptx128
            des, labels_r = obj.getBOW(des, labels_r, False)
        else:
            images = obj.loadData(numImages - len(y_pred), windowSize, "test", "", "test", False, 1) #batch_sizex128
            des, labels_r = obj.getDescriptors(images, [-1]*batch_size, False)
            des, labels_r = obj.getBOW(des, [-1]*batch_size, False)
        if modelType == "logistic":
            y_pred_r = list(map(lambda v: 0 if v[0] > decisionValue else 1, model.predict_proba(des)))
        else:
            y_pred_r = model.predict(des)
        y_pred.extend(y_pred_r)
        
    return labels, y_pred

#%%
modelType = "logistic"
training_errors = []
validation_errors = []
batch_size = 2048#For svm
batch_size = 512
batchs_num = int(train_size/batch_size)
valid_size = 256
epochs = 1
decisionValue = 0.52

model, training_errors, validation_errors = train(obj, training_errors, validation_errors, epochs, batch_size, valid_size, batchs_num, validation_size, train_size, modelType, windowSize, decisionValue)


#%%
# Training set
labels, y_pred = prediction(obj, model, modelType, decisionValue, train_size, windowSize, "train", True)

cnf = confusion_matrix(y_pred, labels)
print(cnf)
print(f"Accuracy: {np.sum(np.diag(cnf))/np.sum(cnf)}")
#%%
# Validation set
labels, y_pred = prediction(obj, model, modelType, decisionValue, trainingSize - train_size, windowSize, "train", False)

cnf = confusion_matrix(y_pred, labels)
print(cnf)
print(f"Accuracy: {np.sum(np.diag(cnf))/np.sum(cnf)}")
#%%
# Test Set
labels, y_pred = prediction(obj, model, modelType, decisionValue, 12500, windowSize, "test", True, False)

# %%
