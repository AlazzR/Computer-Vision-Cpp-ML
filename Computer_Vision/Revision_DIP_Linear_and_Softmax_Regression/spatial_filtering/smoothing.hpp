#ifndef __LPF__
#define __LPF__
#include<iostream>
#include<string>
#include<opencv4/opencv2/opencv.hpp>

cv::Mat gaussian_kernel(float sigma=1.0f, float K=1.0f, int size = -1)
{
    int kernel_size = ceil(6 * sigma);
    if(kernel_size%2 == 0 )
        kernel_size = kernel_size - 1;//need to have odd size kernel.
    if(size != -1)
    {
        kernel_size = size;
        if(kernel_size%2 == 0)   
            kernel_size = kernel_size - 1;
    }
    if(kernel_size <= 0)
        kernel_size = 3;

    cv::Mat kernel = cv::Mat::zeros(cv::Size(kernel_size, kernel_size), CV_32FC1);
    std::cout << kernel.size() << std::endl;
    float* ptrKernel;
    float num;
    float den;
    for(int row=0; row < kernel.rows; row++)
    {
        ptrKernel = kernel.ptr<float>(row);
        for(int col=0; col < kernel.cols; col++)
        {
            num = -1.0f * ( std::pow( row - int(kernel_size/2), 2) + std::pow( col - int(kernel_size/2), 2));
            den = 2 * sigma * sigma;
            ptrKernel[col] = K * std::exp(num/den);
        }
    }
    double sum = cv::sum(kernel)[0];
    kernel = (1.0f/sum) * kernel;//in order to ensure not adding energy to the signal
    //std::cout << kernel << std::endl;
    return kernel;
}

void smoothing_image_with_gaussian(const cv::Mat& img, const std::string file_name, float K=1.0f, float sigma=1.0f, int size=3)
{
    //due to that gaussian is symmetric, we don't need to rotate the kernel and we can directly use convolution as a mere covolutions
    cv::Mat kernel = gaussian_kernel(sigma, K, size);
    //padding image

    int step = (int) (((kernel.rows)-1)/2);
    cv::Mat paddedImage = cv::Mat::zeros(cv::Size(img.size().width + 2 * step, img.size().height + 2 * step), CV_8UC1);
    cv::Mat filteredImage = cv::Mat::zeros(cv::Size(img.size().width + 2 * step, img.size().height + 2 * step), CV_32FC1);
    img.copyTo( paddedImage(cv::Rect(step, step, img.cols, img.rows)) );//notice the flip of rows and cols.

    //cv::imshow("Padded Image", paddedImage);
    //cv::waitKey(0);

    float* ptrFilteredImage;
    for(int row=step; row < filteredImage.rows - step; row++)
    {
        ptrFilteredImage = filteredImage.ptr<float>(row);
        for(int col=step; col < filteredImage.cols - step; col++)
        {
            float center = 0.0f;
            for(int i=-step; i <= step; i++)
            {
                for(int j=-step; j <= step; j++)
                    center += paddedImage.at<uchar>(row + i, col + j) * kernel.at<float>(i + step, j + step);//convolution step
            }
            ptrFilteredImage[col] = center;
        }
    }
    cv::normalize(filteredImage, filteredImage, 0, 255, cv::NORM_MINMAX);
    cv::Mat output(filteredImage(cv::Rect(step, step, img.cols, img.rows)));
    output.convertTo(output, CV_8UC1);

    //cv::imshow("Result", output);
    //cv::waitKey(0);
    cv::imwrite("../spatial_filtering/results/" + file_name + "_lpf_original.jpg", img);
    cv::imwrite("../spatial_filtering/results/" + file_name + "_lpf_transformed.jpg", output);


}


#endif /*__LPF__*/