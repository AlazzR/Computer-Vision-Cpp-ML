#include<iostream>
#include<opencv4/opencv2/opencv.hpp>
#include"histogram_utils.hpp"
#include"smoothing.hpp"
#include"sharpening.hpp"

int main(int argc, char* argv[])
{
    /* Histogram Equalization */
        // const cv::Mat img_1 = cv::imread("../datasets/1_spatial_histEq.tif", CV_8UC1);
        // const cv::Mat img_2 = cv::imread("../datasets/2_spatial_histEq.tif", CV_8UC1);
        // const cv::Mat img_3 = cv::imread("../datasets/3_spatial_histEq.tif", CV_8UC1);
        // const cv::Mat img_4 = cv::imread("../datasets/4_spatial_histEq.tif", CV_8UC1);

        // std::cout << img_1.size << std::endl; 
        // //cv::imshow("Origina_Image", img_1);
        // //cv::waitKey(0);
        // //compute_histogram(); 
        // histogram_equalization(img_1, "1_spatial_histEq");
        // histogram_equalization(img_2, "2_spatial_histEq");
        // histogram_equalization(img_3, "3_spatial_histEq");
        // histogram_equalization(img_4, "4_spatial_histEq");

        // openCV_histEqualization(img_1, "1_spatial_histEq");
        // openCV_histEqualization(img_2, "2_spatial_histEq");
        // openCV_histEqualization(img_3, "3_spatial_histEq");
        // openCV_histEqualization(img_4, "4_spatial_histEq");
    
    /* Smoothing Kernels (LPF) */
        //const cv::Mat img_5 = cv::imread("../datasets/blurring_effect.tif", CV_8UC1);
        //cv::imshow("original_img", img_5);
        //cv::waitKey(0);
        //smoothing_image_with_gaussian(img_5, "filtering_LPF_with_3.5", 1.0f, 3.5f, -1);

    /* Sharpening Kernels (HPF) */
        const cv::Mat img_6 = cv::imread("../datasets/blurry_moon.tif", CV_8UC1);
        sharpening_image(img_6, "blurred_moon");
        //cv::imshow("original_image", img_6);
        //cv::waitKey(0);
        

    return 0;
}