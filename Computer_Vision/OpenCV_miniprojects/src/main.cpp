#include"Feature_Extraction_and_Matching.hpp"
#include"Diabetic_Classification.hpp"

int main(int argc, char* argv[])
{
    /******************************************************************************************/
    
    //Feature Detection(i.e find key points) -> Feature Description(i.e find description of the provided keypoints) -> Features matchers(either using BF--i.e finding lowest distance between matcher-- or FLANN)
    // cv::Mat img1 = cv::imread("../../images/img1gray.png");
    // cv::Mat img2 = cv::imread("../../images/img2gray.png");

    // std::string detectorType = "AKAZE";
    // std::string windowName = "AKAZE detector";
    // Feature_Extract_Match fem1 = Feature_Extract_Match(windowName);
    // Feature_Extract_Match fem2 = Feature_Extract_Match(windowName);

    // std::vector<cv::KeyPoint> kp1 = fem1.feature_detection(img1, detectorType);
    // std::vector<cv::KeyPoint> kp2 = fem2.feature_detection(img2, detectorType);

    // cv::Mat descriptor_1 = fem1.features_extractor(img1, detectorType);
    // cv::Mat descriptor_2 = fem2.features_extractor(img2, detectorType);

    // std::vector<cv::DMatch> matches = fem1.feature_matcher(descriptor_1, descriptor_2, "FLANN", "flann");
    // cv::Mat matchImg = img1.clone();
    // cv::drawMatches(img1, kp1, img2, kp2, matches, matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // cv::imshow("Matched KeyPoints", matchImg);
    // cv::waitKey();
    // cv::destroyAllWindows();

    /******************************************************************************************/
    std::string file_path = "../../datasets/diabetic_classification.txt";
    cv::Mat X;
    cv::Mat y;
    load_csv(file_path, X, y, ',');
    Diabetic_Classification model = Diabetic_Classification(X, y);
    model.normalize_and_split_data();
    std::cout << "Logistic Regression\n";
    model.LogisticRegression(false);
    std::cout << "SVM\n";
    model.SVM(false);




    
    
    return 0;
}