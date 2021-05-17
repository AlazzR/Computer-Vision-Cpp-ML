#ifndef __BLOB_DETECTION__
#define __BLOB_DETECTION__
#include<iostream>
#include<opencv4/opencv2/opencv.hpp>
#include<opencv4/opencv2/imgproc.hpp>
#include<cmath>
#include<vector>
#include<utility>
#include<array>

cv::Mat creating_LoF(const float sigma)
{
    int kernel_size = std::ceil( 6 * sigma );
    if(kernel_size%2 == 0)
        kernel_size -= 1;
    if(kernel_size == 0)
        kernel_size = 3;
    
    cv::Mat kernel = cv::Mat::zeros(cv::Size(kernel_size, kernel_size), CV_32FC1);
    std::cout << "Kernel size " << kernel.size() << std::endl;
    //equation from http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
    float* ptrRow;
    float sum = 0.0f;
    for(int row=0; row < kernel.rows; row++)
    {
        ptrRow = kernel.ptr<float>(row);
        for(int col=0; col < kernel.cols; col++)
        {
            float c = ( std::pow(row - kernel_size/2,2) + std::pow(col - kernel_size/2,2) - 2 * sigma * sigma)/ ( std::sqrt(2* M_PI * sigma * sigma ) * std::pow(sigma, 4));
            float exponent = -1.0f * ((std::pow(row - kernel_size/2,2) + std::pow(col - kernel_size/2,2))/ (2 * sigma * sigma));
            ptrRow[col] = std::pow(sigma, 2) * c * std::exp(exponent);//normalized sigma
            sum += ptrRow[col];
        }
    }

    // std::cout << kernel << std::endl;
    // std::cout << sum << std::endl;
    return kernel;
}

cv::Mat convolution(const cv::Mat& img, const cv::Mat& kernel, const float sigma)
{
    //I will assume that the kernel is the LoG kernel
    //i.e no need to rotate the matrix 180 because LoG is already symmetric
    int k = (kernel.rows - 1)/2;
    int square_img = img.rows >= img.cols? img.rows: img.cols;
    cv::Mat padded_img = cv::Mat(cv::Size(square_img + 2 * k, square_img + 2 * k), CV_32FC1);
    img.copyTo(padded_img(cv::Rect(k, k, img.cols, img.rows)));//notice flip in size

    const uchar* ptrImg;
    float* ptrPaddedImg;
    for(int row=k; row < padded_img.rows - k; row++)
    {
        ptrImg = img.ptr<uchar>(row - k);
        ptrPaddedImg = padded_img.ptr<float>(row);

        for(int col=k; col < padded_img.cols - k; col++)
        {
            float sum=0.0f;
            for(int i=-k; i <=k; i++)
            {
                for(int j=-k; j <=k; j++)
                {
                    sum += ptrImg[col - k + j] * kernel.at<float>(i + k, j + k);
                }
            }
            ptrPaddedImg[col] = sum;

        }
    }
    padded_img = padded_img(cv::Rect(k, k, img.cols, img.rows));
    cv::Mat unnormalized = padded_img.clone();
    cv::normalize(padded_img, padded_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("../spatial_filtering/results/blob_detection_" + std::to_string(sigma) + "_" + std::to_string(kernel.rows) + "x" + std::to_string(kernel.cols) + ".jpg", padded_img);
    //cv::imshow("filtered image", padded_img);
    //cv::waitKey(0);
    return unnormalized;
}

std::vector<cv::Mat> pyramid(const cv::Mat& img, const float initial_sigma, const float scale_factor, const int pyramid_depth)
{
    std::vector<cv::Mat> levels;
    float sigma = initial_sigma;
    for(int i=1; i<= pyramid_depth; i++)
    {
        std::cout <<  "level " << i << std::endl;
        cv::Mat kernel = creating_LoF(sigma);
        cv::Mat result = convolution(img, kernel, sigma);
        sigma = scale_factor * sigma;
        levels.push_back(result);
    }
    return levels;
}

std::vector<std::array<float, 3>> non_maximum_suppression(std::vector<cv::Mat>& levels, const float threshold, int window_size,  const float initial_sigma, const float scale_factor)
{
    std::vector<std::array<float, 3>> interest_points;
    if(window_size%2 == 0)
        window_size = window_size - 1;
    if(window_size == 0)
        window_size = 3;
    cv::Mat window = cv::Mat::zeros(window_size, window_size, CV_32FC1);
    int k = (window_size - 1)/2;
    std::cout << levels[0].size() << std::endl;
    //pad images in pyramid
    for(auto& level: levels)
        cv::copyMakeBorder(level, level, 2 * k, 2 * k, 2 * k, 2 * k, cv::BORDER_CONSTANT, 0);
    std::cout << levels[0].size() << std::endl;
    
    //non-overlapping windows
    for(int row=k; row < levels[0].rows - 2 * k; row+=window_size)
    {

        for(int col=k; col < levels[0].cols - 2 * k; col+=window_size)
        {
            int counter = 1;
            float putative_match[3] = {0, 0, 0};
            double max = 0.0f;  
            for(auto& img: levels)
            {
                window = img(cv::Rect(col, row, window_size, window_size));//notice the flip
                double max_v;
                cv::Point maxP;
                cv::minMaxLoc(window, (double*)0, &max_v, (cv::Point*) 0, &maxP);
                
                if(max <= max_v)
                {
                    max = max_v;
                    if(max_v >= threshold)
                    {
                        putative_match[0] = maxP.y + row;//row
                        putative_match[1] = maxP.x + col;//notice the flip // col
                        putative_match[2] = std::pow(scale_factor, counter) * initial_sigma * std::sqrt(2);//radius
                        //std::cout << "x: " << putative_match[1] << " y: " << putative_match[0] << " r: " << putative_match[2] << std::endl; 

                    }

                }
                counter++;
            }
            std::array<float, 3> tmp = { putative_match[0], putative_match[1], putative_match[2] } ;
            if(putative_match[0] != 0 && putative_match[1] != 0 && putative_match[2] != 0)
                interest_points.push_back(tmp);
        }
    }


    return interest_points;

}

void blob_detector(const cv::Mat& img, const float initial_sigma, const float scale_factor=1.3f, const int pyramid_depth=5, const float threshold=0.0f, int window_size = 3)
{
    int square = img.rows >= img.cols? img.rows: img.cols;
    cv::Mat grayscale = cv::Mat::zeros(cv::Size(square, square), CV_8UC1);
    img.copyTo(grayscale(cv::Rect(0, 0, img.cols, img.rows)));//notice the flip

    std::vector<cv::Mat> levels = pyramid(grayscale, initial_sigma, scale_factor, pyramid_depth);
    std::vector<std::array<float, 3>> interest_points = non_maximum_suppression(levels, threshold, window_size, initial_sigma, scale_factor);
    
    // cv::Mat imageWithColor = cv::Mat::zeros(cv::Size(square, square), CV_8UC3);
    // std::vector<cv::Mat> channels(3);
    // cv::split(imageWithColor, channels);
    // channels[0] = grayscale; channels[1] = grayscale; channels[2] = grayscale;
    std::cout << grayscale.channels() << std::endl;

    cv::cvtColor(grayscale, grayscale, cv::COLOR_GRAY2BGR);
    //opencv use BGR order for channels, I don't want to cvtColor
    for(auto& point: interest_points)
    {
        //std::cout << "x: " << point[1] << " y: " << point[0] << " r: " << point[2] << std::endl; 
        cv::circle(grayscale, cv::Point(point[1], point[0]), point[2], cv::Scalar(0, 0, 255));
    }
    // cv::resize(channels[0], channels[0], cv::Size(img.cols, img.rows));
    // cv::resize(channels[1], channels[1], cv::Size(img.cols, img.rows));
    // cv::resize(channels[2], channels[2], cv::Size(img.cols, img.rows));

    // cv::merge(channels, grayscale);
    grayscale = grayscale(cv::Rect(0, 0, img.cols, img.rows));//notice the flip
    std::cout << grayscale.channels() << std::endl;
    cv::imshow("image with blob", grayscale);
    cv::waitKey(0);
    cv::imwrite("../spatial_filtering/results/blob_detection_with_blobs.jpg", grayscale);

}

#endif /*__BLOB_DETECTION__*/