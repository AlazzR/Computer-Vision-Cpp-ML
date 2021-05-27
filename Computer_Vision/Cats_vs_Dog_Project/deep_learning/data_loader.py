#%%
import os 
import numpy as np
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import PIL
import matplotlib.pyplot as plt
#Cat is 0 and dof is 1
# %%
def loadFileNames(trainTestFlag=True, start=0, end=12500, root=os.getcwd()):
    fileNames = []
    for ind in range(start, end):
        fileName = []
        if trainTestFlag:
            fileName = ["cat." + str(ind) + ".jpg", "dog." + str(ind) + ".jpg"]
        else:
            fileName = [str(ind) + ".jpg"]
        
        for ind, p in enumerate(fileName):
            if trainTestFlag:
                path = os.path.abspath(os.path.join(root, "../dataset/train/" + p))
                if os.path.exists(path):
                    if ind == 0:
                        fileNames.append((path, 0))
                    else:
                        fileNames.append((path, 1))
            else:
                path = os.path.abspath(os.path.join(root, "../dataset/test/" + p))
                if os.path.exists(path):
                    fileNames.append((path, -1))
    return fileNames

# %%
class DataAugmentation(Dataset):

    def __init__(self, imgSize, fileNames, *args, **kwargs):
        super(DataAugmentation, self).__init__(*args, **kwargs)
        self.files = fileNames
        self.tensorTransform = torchvision.transforms.ToTensor()
        self.resize = torchvision.transforms.Resize((imgSize, imgSize))
        self.tranlationTransform = torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        self.rotationalTransformation = torchvision.transforms.RandomRotation(45, center=(imgSize/2, imgSize/2) )
        self.imgSize = imgSize
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = PIL.Image.open(self.files[index][0])
        label = self.files[index][1]
        img = self.resize(img)
        img = self.tranlationTransform(img)
        img = self.rotationalTransformation(img)
        #img = np.array(img, dtype=np.float32)
        img = self.tensorTransform(img)#Will read it into 3xheightxwidth
        #Standard normalize images
        ravel = img.view(3, -1)
        means = torch.mean(ravel, dim=1)
        ravel -= torch.transpose(torch.broadcast_to(means, (ravel.shape[1], 3)), 0, 1)
        std = torch.std(ravel, dim=1)
        ravel /= torch.transpose(torch.broadcast_to(std, (ravel.shape[1], 3)), 0, 1)
        
        img = ravel.view(3, self.imgSize, self.imgSize)

        return img, label

class OriginalImages(Dataset):

    def __init__(self, imgSize, fileNames, *args, **kwargs):
        super(OriginalImages).__init__(*args, **kwargs)
        self.files = fileNames
        self.tensorTransform = torchvision.transforms.ToTensor()
        self.resize = torchvision.transforms.Resize((imgSize, imgSize))
        self.imgSize = imgSize
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = PIL.Image.open(self.files[index][0])
        label = self.files[index][1]
        img = self.resize(img)
        img = self.tensorTransform(img)
        
        #Standard normalize images
        ravel = img.view(3, -1)
        means = torch.mean(ravel, dim=1)
        ravel -= torch.transpose(torch.broadcast_to(means, (ravel.shape[1], 3)), 0, 1)
        std = torch.std(ravel, dim=1)
        ravel /= torch.transpose(torch.broadcast_to(std, (ravel.shape[1], 3)), 0, 1)
        
        img = ravel.view(3, self.imgSize, self.imgSize)
        return img, label



# %%
if __name__ == "__main__":
    fileNames = np.array(loadFileNames())
    indeces = np.random.choice(len(fileNames), len(fileNames), replace=False)
    trainTestRatio = 0.8
    trainIndeces, validationIndeces = indeces[:int(0.8 * len(indeces))], indeces[int(0.8 * len(indeces)):] 
    augmentedData = DataAugmentation(224, fileNames[trainIndeces])
    originalData = OriginalImages(224, fileNames[trainIndeces])

    dataLoader = DataLoader(ConcatDataset([originalData, augmentedData]), batch_size=256, shuffle=True)

    counter = 0
    for batch, label in dataLoader:
        if counter > 10:
            break
        counter += 1
        plt.imshow(batch[0, 0, :, :], cmap="gray")
        plt.show()

# %%
