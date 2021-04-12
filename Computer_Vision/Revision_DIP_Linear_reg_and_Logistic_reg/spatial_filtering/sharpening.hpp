#ifndef __HPF__
#define __HPF__
#include<iostream>
#include<opencv4/opencv2/opencv.hpp>
#include<opencv4/opencv2/imgproc.hpp>
#include<opencv4/opencv2/core.hpp>//for flipping kernel
#include<string>

cv::Mat conv_3x3(const cv::Mat& img, const cv::Mat& kernel)
{
    int step = 1;
    cv::Mat paddedImage = cv::Mat::zeros(cv::Size(img.size().width + 2 * step, img.size().height + 2 * step), CV_8UC1);
    cv::Mat filteredImage = cv::Mat::zeros(paddedImage.size(), CV_32FC1);

    img.copyTo(paddedImage(cv::Rect(step, step, img.cols, img.rows)));
    float* ptrFilteredImg;
    for(int row=step; row < filteredImage.rows - step; row++)
    {
        ptrFilteredImg = filteredImage.ptr<float>(row);;
        for(int col=step; col < filteredImage.cols - step; col++)
        {
            float center = 0.0f;
            for(int i=-step; i <= step; i++)
            {
                for(int j=-step; j <= step; j++)
                {
                    center += paddedImage.at<uchar>(row + i, col + j) * kernel.at<float>(i + step, j + step);
                }
            }
            ptrFilteredImg[col] = center;

        }
    }
    //sobel and laplacian sum is 0, i.e, they will introduce negative intensities
    double min, max;
    cv::minMaxLoc(filteredImage, &min, &max, NULL, NULL);
    std::cout << std::endl;
    std::cout << "min" << min << " max " << max << std::endl;
    //std::cout << std::endl << kernel << std::endl;
    cv::normalize(filteredImage, filteredImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::imshow("grad", filteredImage);
    //cv::waitKey(0);
    cv::Mat output(filteredImage(cv::Rect(step, step, img.cols, img.rows)));

    return output;
}

void sharpening_image(const cv::Mat& img, const std::string file_name)
{
    std::cout << "Image size " << img.size() << std::endl;
    int max_dim = std::max<int>(img.size().width, img.size().height);
    cv::Mat square_img = cv::Mat::zeros(cv::Size(max_dim, max_dim), CV_8UC1);
    img.copyTo(square_img(cv::Rect(0, 0, img.cols, img.rows)));

    cv::Mat sobel_x = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
    sobel_x = (cv::Mat_<float>(3, 3) <<  -1.0f, 0.0f, 1.0f,
                                        -2.0f, 0.0f, 2.0f, 
                                        -1.0f, 0.0f, 1.0f);

    cv::Mat sobel_y = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
    sobel_y = (cv::Mat_<float>(3, 3) <<  -1.0f, -2.0f, -1.0f,
                                        0, 0, 0, 
                                        1.0f, 2.0f, 1.0f);

    cv::Mat laplacian = (cv::Mat_<float>(3, 3) << 0.0f, 1.0f, 0.0f,
                                                  1.0f, -4.0f, 1.0f,
                                                  0.0f, 1.0f, 0.0f);
    cv::Mat dst;
    cv::flip(sobel_x, dst, -1);
    sobel_x = dst;
    cv::flip(sobel_y, dst, -1);
    sobel_y = dst;

    std::cout << "Sobel in x-direction after flipping\n" << sobel_x << std::endl;
    std::cout << "Sobel in y-direction after flipping\n" << sobel_y << std::endl;


    std::cout << sobel_x;
    cv::Mat grad_x = conv_3x3(square_img, sobel_x);
    cv::Mat grad_y = conv_3x3(square_img, sobel_y);
    cv::Mat secondOrder_gradient = conv_3x3(square_img, laplacian);
    //taking the original size of the image
    grad_x = grad_x(cv::Rect(0, 0, img.cols, img.rows));
    grad_y = grad_y(cv::Rect(0, 0, img.cols, img.rows));
    secondOrder_gradient = secondOrder_gradient(cv::Rect(0, 0, img.cols, img.rows));


    cv::imwrite("../spatial_filtering/results/" + file_name + "_original.jpg", img);
    cv::imwrite("../spatial_filtering/results/" + file_name + "_grad_x.jpg", grad_x);
    cv::imwrite("../spatial_filtering/results/" + file_name + "_grad_y.jpg", grad_y);
    cv::imwrite("../spatial_filtering/results/" + file_name + "_2nd_gradient.jpg", secondOrder_gradient);

    cv::Mat sharpenedByLaplacian = img - 0.5 * secondOrder_gradient;
    cv::normalize(sharpenedByLaplacian, sharpenedByLaplacian, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::equalizeHist(sharpenedByLaplacian, sharpenedByLaplacian);
    //cv::imshow("dfdd", sharpenedByLaplacian);
    //cv::waitKey(0);
    cv::imwrite("../spatial_filtering/results/" + file_name + "_sharpended_by_laplacian.jpg", sharpenedByLaplacian);


}
#endif /*__HPF__*/