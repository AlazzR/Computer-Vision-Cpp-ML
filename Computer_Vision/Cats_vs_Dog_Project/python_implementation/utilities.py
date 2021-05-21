#%%
import os 
import cv2
import numpy as np
#%%
class LoadingImages(object):


    def __init__(self, trainingSize, numClusters=10, nfeatures=10):
        self.counts = {
            "cat": 0,
            "dog": 0,
            "test": 1
        }
        #Use SIFT descriptors
        self.sift = cv2.SIFT_create(nfeatures, 3, 0.0, 10, 1.6, None)
        #Use FLANN as your matcher between the BOW for each cluster in the 128 parameters space.
        FLANN_INDEX_KDTREE = 1 
        flann_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
        matcher = cv2.FlannBasedMatcher(flann_params, {})
        #Compute image descriptor using bag of word method
        self.bowExtractor = cv2.BOWImgDescriptorExtractor(self.sift, matcher)
        self.bowKmeansClusters = cv2.BOWKMeansTrainer(numClusters)
        self.nfeatures  = nfeatures
        self.numClusters = numClusters
        self.trainingSize = trainingSize
        self.indeces = list(np.random.choice(self.trainingSize, self.trainingSize, replace=False))

    def resetCounters(self):
        self.counts = {
            "cat": 0,
            "dog": 0,
            "test": 1
        }

    def loadData(self, numImages, windowSize, countType, p='', trainOrTest="train", showMe=False, start=0):
        images = []

        for c in range(start, numImages):

            try:
                ind = self.indeces[self.counts[countType]]
                img = cv2.imread(os.path.abspath(os.path.join(os.getcwd(), f"..\\dataset\\{trainOrTest}\\" + p + str(ind) + ".jpg")), cv2.IMREAD_GRAYSCALE)

                #histogram equalization to remove skewness in the intensity
                img_center = np.zeros((windowSize, windowSize), dtype=np.ubyte)
                #print(img.shape)

                # Using strides
                # print(img.shape)
                # print(img_center.shape)
                numStrides = [int(np.ceil(img.shape[0]/windowSize)), int(np.ceil(img.shape[1]/windowSize ))]
                numStrides[0] = 2 if numStrides[0] <= 1 else numStrides[0]
                numStrides[1] = 2 if numStrides[1] <= 1 else numStrides[1]
                #print(numStrides)
                img = img[::numStrides[0], ::numStrides[1]]
                img_center[:img.shape[0], :img.shape[1]] = img[:, :]

                #Getting the middle part of the image
                # center = (int(img.shape[0]/2), int(img.shape[1]/2))
                # #Get a centered Verision of the image
                # ptLeftX = center[0] - int(windowSize/2) if center[0] - int(windowSize/2) > 0 else 0
                # ptLeftY = center[1] - int(windowSize/2) if center[1] - int(windowSize/2) > 0 else 0

                # ptRightX = center[0] + int(windowSize/2) if center[0] + int(windowSize/2) < img.shape[0] else img.shape[0]
                # ptRightY = center[1] + int(windowSize/2) if center[1] + int(windowSize/2) < img.shape[1] else img.shape[1]

                # img_center = img[ptLeftX:ptRightX, ptLeftY:ptRightY]
                
                
                img_center = cv2.equalizeHist(img_center)
                img_center = np.asarray(img_center, np.float32)
                means, stddev = cv2.meanStdDev(img_center)
                img_center -= means
                stddev = stddev if stddev != 0 else 1
                img_center /= stddev

                if showMe:
                    cv2.imshow(p + str(ind), img_center)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                    cv2.destroyAllWindows()
                #print(np.mean(img_center))
                #print(np.std(img_center))
                #print(img_center.shape)
                images.append(img_center)
                self.counts[countType] += 1


            except Exception as e:
                #Restart your image loading
                print(e)
                print(self.counts[countType])
                self.counts[countType] = start
                #images = []
                break
        return images

    def getDescriptors(self, images, labels,  visualize=False):
        descriptors = []
        counter = 0
        for img in images:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            kp = self.sift.detect(img)
            des = np.zeros((self.nfeatures, 128))
            if visualize:
                channels = []
                colored = None
                channels.append(img)
                channels.append(img)
                channels.append(img)
                colored = cv2.merge(channels)
                colored = cv2.drawKeypoints(img, kp, colored, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow(str(counter), colored)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                cv2.destroyAllWindows()
            try:
                #tmp = np.ravel(self.sift.compute(img, kp)[-1][0:self.nfeatures]).reshape(1, -1)
                tmp = self.sift.compute(img, kp)[-1]
                des[:len(tmp)] = tmp  
            except:
                pass

            #Make feature vector 1x(128 * #kp)    
            descriptors.append(des)
            counter += 1
        #Return #images x #features x 128, #imagesx x 1

        return descriptors, labels

    def getBOW(self, descriptors, labels, trainFlag=False):
        #add our data to cluster the visual vocabularly on
        #labels_extends = []
        if trainFlag:
            #nfeaturex128
            for arr in descriptors:
                #print(arr.shape)
                self.bowKmeansClusters.add(arr)#This will keep previous data unless we clear the data structure
            vocuabulary = self.bowKmeansClusters.cluster()
            #print(vocuabulary)
            #After finding vocabulary for each cluster, we need to set the BOW for those clusters
            self.bowExtractor.setVocabulary(vocuabulary)
        
        #print(self.bowExtractor.getVocabulary())
        vocuabulary = self.bowExtractor.getVocabulary()#numClusters x 128
        hists = []
        counter = 0
        for arr in descriptors:
            hist = np.zeros((self.numClusters, 1))#histogram for each image with respect to the clusters.
            for row in arr:
                #finding nearest cluster to construct visual word or codeblock
                i = np.argmin(np.sqrt(np.sum(np.power(vocuabulary - row.reshape(1, -1), 2), axis=1)), axis=0)
                hist[i, 0] += 1
            hists.extend(hist.reshape(1, -1))
            #labels_extends.extend([labels[counter]] * self.nfeatures)
            counter += 1
        return np.array(hists, np.float32), np.array(labels).reshape(-1, 1)


    def loadDescriptorsAndLabels(self, dataSize, windowSize=200, visualize=False):
        #Get 2 * dataSize of images
        labels = [0] * dataSize
        labels.extend([1] * dataSize)
        images = self.loadData(dataSize, windowSize, "cat", "cat.")   
        images.extend(self.loadData(dataSize, windowSize, "dog", "dog."))

        des, labels = self.getDescriptors(images, labels, visualize)
        return np.array(des, np.float32) , labels

    def loadDataRandomly(self, numImages, windowSize, showMe=False):
        images = []
        labels = []
        indeces = np.random.choice(12500, numImages, replace=False)
        typeAnimal = np.random.choice(2, numImages)
        lab = {0: "cat.", 1:"dog."}
        for ind in range(0, len(indeces)):

            try:
                img = cv2.imread(os.path.abspath(os.path.join(os.getcwd(), "..\\dataset\\train\\" + lab[typeAnimal[ind]] + str(indeces[ind]) + ".jpg")), cv2.IMREAD_GRAYSCALE)

                #histogram equalization to remove skewness in the intensity
                img_center = np.zeros((windowSize, windowSize), dtype=np.ubyte)
                numStrides = [int(np.ceil(img.shape[0]/windowSize)), int(np.ceil(img.shape[1]/windowSize ))]
                numStrides[0] = 2 if numStrides[0] <= 1 else numStrides[0]
                numStrides[1] = 2 if numStrides[1] <= 1 else numStrides[1]
                #print(numStrides)
                img = img[::numStrides[0], ::numStrides[1]]
                img_center[:img.shape[0], :img.shape[1]] = img[:, :]
                img_center = cv2.equalizeHist(img_center)
                img_center = np.asarray(img_center, np.float32)
                means, stddev = cv2.meanStdDev(img_center)
                img_center -= means
                stddev = stddev if stddev != 0 else 1
                img_center /= stddev

                images.append(img_center)
                labels.append(typeAnimal[ind])

            except Exception as e:
                print(e)

        return images, labels

#%%
#obj = LoadingImages(128, 20)
#images = obj.loadData(64, 200, "cat", "cat.", "train", False)
#images = obj.loadData(64, 200, "cat", "cat.", "train", False)


#des, labels = obj.loadDescriptorsAndLabels(64, 200)
#%%
#des, labels = obj.getBOW(des, labels, True)
# #des, labels = obj.loadDescriptorsAndLabels(64)
# #obj.getBOW(des)
# %%
