#ifndef __FEATURE_EXTRACTION__
#define __FEATURE_EXTRACTION__

#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>
#include<string>
#include<vector>

class Feature_Extract_Match
{
private:
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::DMatch> matches;
    cv::Mat descriptors_values;
    std::string windowName;
public:
    Feature_Extract_Match(std::string windowName):windowName(windowName){}
    std::vector<cv::KeyPoint> feature_detection(const cv::Mat& original, std::string detectorType)
    {
        cv::Mat img;
        cv::cvtColor(original, img, cv::COLOR_BGR2GRAY);
        
        //FeatureDetector -> KeyPoint -> detect -> drawKeyPoints.
        cv::Ptr<cv::FeatureDetector> detector;//to allow polymorphism 
        if(detectorType.compare("SIFT") == 0)
        {
            detector = cv::SIFT::create(0, 5, 0.04, 8.0);
        }
        else if(detectorType.compare("FAST") == 0)
        {
            detector = cv::FastFeatureDetector::create();
        }
        else if(detectorType.compare("BRISK") == 0)
        {
            detector = cv::BRISK::create();
        }
        else if(detectorType.compare("ORB") == 0)
        {
            detector = cv::ORB::create();
        }
        else{
            detector = cv::AKAZE::create();
        }
        detector->detect(img, this->keypoints, cv::Mat());
        cv::Mat coloredImage;
        cv::Mat clone = img.clone();
        std::vector<cv::Mat> tmp = {clone, clone, clone};
        cv::merge(tmp, coloredImage);

        cv::drawKeypoints(coloredImage, this->keypoints, coloredImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName);
        cv::imshow(windowName, coloredImage);
        cv::waitKey();
        cv::destroyAllWindows();
        std::cout << "#Keypoints: " << this->keypoints.size() << std::endl;

        return this->keypoints;
    } 
    cv::Mat features_extractor(const cv::Mat& original, std::string detectorType)
    {
        cv::Mat img;
        cv::cvtColor(original, img, cv::COLOR_BGR2GRAY);

        //Descriptor Extraction from the interest points
        //DescriptorExtractor -> compute
        cv::Ptr<cv::DescriptorExtractor> descriptor;//wrapper around most of the detector methods
        cv::Mat descriptors_values;
        if(detectorType.compare("SIFT") == 0)
        {
            descriptor = cv::SIFT::create(0, 5, 0.04, 8.0);
        }
        else if(detectorType.compare("FAST") == 0)
        {
            descriptor = cv::FastFeatureDetector::create();
        }
        else if(detectorType.compare("BRISK") == 0)
        {
            descriptor = cv::BRISK::create();
        }
        else if(detectorType.compare("ORB") == 0)
        {
            descriptor = cv::ORB::create();
        }
        else{
            descriptor = cv::AKAZE::create();
        }
        descriptor->compute(img, keypoints, descriptors_values);
        std::cout << "Descriptors size: " << descriptors_values.size() << std::endl;

        return descriptors_values;
    }

    std::vector<cv::DMatch> feature_matcher(cv::Mat& descriptors_img_1, cv::Mat& descriptors_img_2, std::string matcher_type, std::string selector_type, float minRatio = 0.8, int k=10)
    {
        //DescriptorMatcher -> create -> match and choose best matches.
        cv::Ptr<cv::DescriptorMatcher> matcher;
        if(matcher_type.compare("BF_Matcher") == 0)
        {
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
        }
        else{
            //FLANN based matcher
            if(descriptors_img_1.type() != CV_32F)
            {
                descriptors_img_1.convertTo(descriptors_img_1, CV_32F);
                descriptors_img_2.convertTo(descriptors_img_2, CV_32F);
            }

            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        }
        if(selector_type.compare("BF") == 0 )
        {
            matcher->match(descriptors_img_1, descriptors_img_2, this->matches);
        }
        else
        {
            //by using knn matcher
            std::vector<std::vector<cv::DMatch>> putative_matches;
            matcher->knnMatch(descriptors_img_1, descriptors_img_2, putative_matches, k, cv::Mat());
            for(auto& match: putative_matches)
            {
                int counter = 0;
                for(int j=1; j < k; j++)
                {
                    if(match[0].distance < minRatio * match[j].distance)
                    {
                        counter ++;
                    }
                }
                if(counter == k-1)
                    this->matches.push_back(match[0]);
            }

        }
        std::cout << "#Matches " << this->matches.size() << std::endl;
        return this->matches;
    }
};

#endif /*__FEATURE_EXTRACTION__*/