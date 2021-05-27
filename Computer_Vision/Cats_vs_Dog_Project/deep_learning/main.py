#%%
import data_loader
import vgg_models
import numpy as np
import matplotlib.pyplot as plt
import time 
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import DataLoader, ConcatDataset

from sklearn.metrics import confusion_matrix

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

def predict(loader, model):
    y_pred = []
    y_true = []
    counter = 0
    for batch, labels in loader:
        y_p = model(batch)
        y = list(np.array(labels, dtype=np.int8))
        _, y_p_max = torch.max(y_p, dim=1)
        y_pred.extend(y_p_max.tolist())
        y_true.extend(y)
        print(f"batch#{counter}")
        counter += 1
    return y_pred, y_true
# %%
fileNames = np.array(data_loader.loadFileNames())
indeces = np.random.choice(len(fileNames), len(fileNames), replace=False)
trainTestRatio = 0.8

trainIndeces, validationIndeces = indeces[:int(0.8 * len(indeces))], indeces[int(0.8 * len(indeces)):] 

augmentedDataTrain = data_loader.DataAugmentation(224, fileNames[trainIndeces])
originalDataTrain = data_loader.OriginalImages(224, fileNames[trainIndeces])

originalDataValidation = data_loader.OriginalImages(224, fileNames[validationIndeces])


fileNamesTest = data_loader.loadFileNames(False, 1, 12501)
originalDataTest =  data_loader.OriginalImages(224, fileNamesTest)

dataLoaderTrain = DataLoader(ConcatDataset([originalDataTrain, augmentedDataTrain]), batch_size=256, shuffle=True)
dataLoaderValid = DataLoader(originalDataValidation, batch_size=256, shuffle=True)
dataLoaderTest = DataLoader(originalDataTest, batch_size=256, shuffle=True)
# %%
model = vgg_models.VggWithCustomLayers()
print(model)

#model.initialize_parameters(False)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()
validation_errors = []
training_errors = []
print(model)
model =model.to(device)
#%%
for name, param in model.named_parameters():
    print(f"{name} {param.requires_grad}")
# %%
epochs = 1
y_pred_train = []
y_pred_valid = []
y_true_train = []
y_true_valid = []
for epoch in range(0, epochs):
    batchNum = 0
    begin = time.time()
    trainingErrorBatches = []
    validationErrorBatches = []

    for batch, labels in dataLoaderTrain:
        batch = batch.to(device)
        optimizer.zero_grad()
        y_pred = model(batch)
        y = torch.tensor(np.array(labels, dtype=np.int8)).type(torch.LongTensor)
        y = y.to(device)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        if epoch + 1 == epochs:
            y_pred_train.extend(list(np.argmax(y_pred.detach().numpy(), axis=1)))
            y_true_train.extend(y.tolist())
        trainingErrorBatches.append(loss.item())
        print(f"{epoch}:{batchNum}, -log-likelihood {np.round(loss.item(), 3)}")
        cnf = confusion_matrix(y, np.argmax(y_pred.detach().numpy(), axis=1))
        print(f"Accuracy: {np.round(np.sum(np.diag(cnf))/np.sum(cnf), 3)*100}, \n {cnf}")
        batchNum += 1

    for batch, labels in dataLoaderValid:
        optimizer.zero_grad()
        y = torch.tensor(np.array(labels, dtype=np.int8)).type(torch.LongTensor)
        y_pred = model(batch)
        y = y.to(device)
        loss = loss_fn(y_pred, y)
        if epoch + 1 == epochs:
            y_pred_valid.extend(list(np.argmax(y_pred.detach().numpy(), axis=1)))
            y_true_valid.extend(y.tolist())
        validationErrorBatches.append(loss.item())
    training_errors.append(np.mean(trainingErrorBatches))
    validation_errors.append(np.mean(validationErrorBatches))

    end = time.time()
    print(f"{epoch} the avg -log-likelihood for training is {np.round(training_errors[-1], 2)} and it took {(end - begin)/60} min")
    print(f"{epoch} the avg -log-likelihood for validation is {np.round(validation_errors[-1], 2)} and it took {(end - begin)/60} min")

#%%
#torch.save(model.state_dict(), "./model.h5")
#model_2 = vgg_models.VggWithCustomLayers()
#model_2.load_state_dict(torch.load("./model.h5"))
# %%
# Training Performance
#y_pred, y_true = predict(dataLoaderTrain, model)
cnf = confusion_matrix(y_pred_train, y_true_train)
print(f"Accuracy on training set {np.round(np.sum(np.diag(cnf))/np.sum(cnf), 3) * 100} \n {cnf}")

# %%
# Validation Performance
#y_pred, y_true = predict(dataLoaderValid, model)
cnf = confusion_matrix(y_pred_valid, y_true_valid)
#cnf = confusion_matrix(y_pred, y_true)
print(f"Accuracy on validation set {np.round(np.sum(np.diag(cnf))/np.sum(cnf), 3) * 100} \n {cnf}")
# %%
# Test Performance
y_pred, y_true = predict(dataLoaderTrain, model)
df = pd.DataFrame(np.c_[y_pred, np.arange(1, len(y_pred) + 1)], columns=["id", "label"])
df.to_csv("./data.csv", header=True, index=False)

# %%
