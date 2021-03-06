# Computer-Vision-Cpp-ML
**Note:** This project will be continually updated and I sure hope that you enjoy what you see in this repository. But please note that you might find things that you would think it must be optimized or even the code can be condensed in a manner that would be more readable and efficient. Hence, you need to know that I am doing these projects just to show case my capabilities for the recruiters of the jobs that I am currently applying to. Best of wishes to whoever read this and if you enjoyed what you saw here, please don't hesitate to contact me on my LinkedIn account [LinkedIn Account](https://www.linkedin.com/in/rashidalazzoni/).

This repository will contain two projects that I am currently working on, in order to enhance my capabilities and to prevent my personal development from stagnating due to my current situation of unemployment. The two projects that I worked on in this repository are computer vision with emphasis on machine learning with computer vision using C++, and machine learning sub-projects on data that I find in Kaggle and I will incorporate some of the time series analysis in this project.

The computer vision project will focus mainly on machine learning application with incorporating some of the deep learning networks using PyTorch API for C++. Also, I will implement some of the image processing tools, geometric transformation, camera calibration, multiple view geometry analysis and etc. Some of these implementations will be from scratch and these will be marked by (S) and some wouldn't be from scratch and these willn't be marked. For this project, I will utilize the following books:-
#### Computer Vision Books:-
  * Computer Vision: Models, Learning and Inference by Simon J. Prince
  * Concise Computer Vision: an Introduction into Theory and Algorithms by Richard Klette.
  * Digital Image Processing by Rafeal Gonzalez and Richard Woods.

The second project will focus mainly on data science using conventional machine learning algorithms and the good old statistical analysis methods, and I will incorporate some time series analysis projects to ensure that I keep my skills brushed up. I will mostly deal with data that I can find in Kaggle or in UCI Machine Learning Repository. I willn't implement anything from scratch, hence, I will be able to use the full power of whatever package that I found useful on the internet. This project will mainly focus on the following books:-
#### Machine Learning and Time Series Analysis Books:-
  * Pattern Recognition and Machine Learning by Christopher Bishop.
  * Time Series Analysis and its Applications with R Examples by Robert Shumway and David Stoffer


--------------------------------------------------------------------------------------------------------
## Computer Vision
   * **Revision_DIP_Linear_reg_and_Softmax_reg**
       * Image smoothing kernels in spatial domain(S)
       * Image sharpening kernels in spatial domain(S)
       * Histogram plotting and Histogram equalization(S)
       * Linear regression using Eigen3 only(S)
       * Softmax regression using Eigen3 only(S)
       * Blob detection using LoG(S)
       
  * **OpenCV_miniprojects**
       * Feature descriptors(SIFT, FAST, BRISK, ORB, AKAZE) and matching (BF, FLANN, knn)
       * Using Logistic regression and SVM to classify if the person have a diabetes or not
  
  * **Cats_vs_Dogs_Project**
       * Feature extraction and description using SIFT
       * Finding Clusters within the feature space using K-means
       * Constructing the Vocabulary list for the Bag of Visual Words
       * Classification using SVM and Logistic Regression
       * VGG16 with Transfer learning using PyTorch

## Machine Learning and Time Series Analysis
 * **IRIS Dataset Classification**
     * Softmax Regression with data transformation, and visualization using Sklearn, Pandas, PyTorch and Matplotlib
 
 * **LeNet-5 For Single Digit Classification**
     * Using PyTorch and OpenCV to experiment with the LeNet-5


**Packages used for this Repository**
  * ***Python***:
    * scikit-learn==0.22.2.post1
    * scipy==1.4.1
    * pandas==1.0.1
    * numpy==1.18.1
    * PyTorch==1.8.0

 * ***C++17***:
    * OpenCV==4.5.1
    * Eigen3==3.3.9
    * PyTorch(C++ API)==1.8.0
    * CMAKE==3.16.3
    * dlib=19.21
