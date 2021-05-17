#ifndef __HISTOGRAM_EQUALIZATION__
#define __HISTOGRAM_EQUALIZATION__
#include<iostream>
#include<opencv4/opencv2/opencv.hpp>
#include<opencv4/opencv2/imgproc.hpp>//for calcHist, 
#include<cmath>
#include<string>
#include<memory>
//computing histogram
std::unique_ptr<float[]> compute_histogram(const cv::Mat& array)
{
    cv::Mat arr = cv::Mat::zeros(array.size(), CV_8UC1);
    array.convertTo(arr, CV_8UC1);

    //Testing method first
    // cv::Mat array = cv::Mat::zeros(cv::Size(5, 5), CV_8UC1);
    // array =   (cv::Mat_<u_char>(5, 5) <<    0, 1, 2, 3, 4,
    //                                         0, 1, 2, 3, 4,
    //                                         0, 1, 2, 3, 4,
    //                                         0, 1, 2, 3, 4,
    //                                         0, 1, 2, 3, 4);

    std::unique_ptr<float[]> hist(new float[256]);
    for(int ind=0; ind < 256; ind++)
        hist[ind] = 0.0f;
    u_char* ptrRow;

    for(int row=0; row < arr.rows; row++)
    {   
        ptrRow = arr.ptr<u_char>(row);
        for(int col=0; col < arr.cols; col++)
            hist[ptrRow[col]] += 1.0f;
    }

    //print hist values
    //for(int lvl=0; lvl < 5; lvl++)
    //    std::cout << "lvl: " << lvl << " " << hist[lvl] << std::endl;

    return hist;
}


//Plotting hisograma and doing equalization
std::unique_ptr<float[]> histogram_plotting(std::unique_ptr<float[]> hist, const std::string file_name, double max, double min)
{
    std::unique_ptr<float[]> hist_for_plotting(new float[256]);
    cv::Mat histogram = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
    //scale histogram to be [0, 255] for plotting
    for(int i=0; i < 256; i++)
    {
        if(max != min)
            hist_for_plotting[i] = (hist[i] - min ) * (255 - 0)/(max - min) + 0;
    }

    //order in OpenCV is major-row order
    for(int col=0; col < histogram.cols; col++)
    {

        for(int row= histogram.rows - 1; row >= histogram.rows - (int) (hist_for_plotting[col]); row--)
        {
            histogram.at<uchar>(row, col) = 255;
        }
    }

    //cv::imshow("Histogram " + file_name, histogram);
    //cv::waitKey(0);
    cv::imwrite("../spatial_filtering/results/" + file_name + "_hist.jpg", histogram);
    return hist;
}

void histogram_equalization(const cv::Mat& img, const std::string file_name)
{
    std::unique_ptr<float[]> hist(compute_histogram(img));
    int width = img.cols;
    int height = img.rows;
    //scaling the hist to range[0, 1]
    double max = 0.0;
    double min = 1.0;
    float normalizeHist = 0.0f;
    for(int i=0; i < 256; i++)
    {
        hist[i] = hist[i]/(width * height);
        normalizeHist += hist[i];
        if(max <= hist[i])
            max = (double)hist[i];
        if(min > hist[i])
            min = (double)hist[i];
    }

    std::cout << "Checking if the histogram is a valid pdf: " << normalizeHist << std::endl; 
    hist = histogram_plotting(std::move(hist), file_name, max, min);
    //histogram equalization
    cv::Mat histogramEqualized = img.clone();
    histogramEqualized.convertTo(histogramEqualized, CV_32FC1);
    float* ptrImg;
    for(int row=0; row < histogramEqualized.rows; row++)
    {
        ptrImg = histogramEqualized.ptr<float>(row);
        for(int col=0; col < histogramEqualized.cols; col++)
        {
            float cdf = 0.0;
            for(int s=0; s < (int)ptrImg[col]; s++)
            {
                cdf += hist[s];
            }
            //std::cout << cdf << std::endl;
            ptrImg[col] = (256 - 1) * cdf;
        }
    }

    hist.release();
    hist = compute_histogram(histogramEqualized);
    //cv::minMaxLoc(histogramEqualized, &min, &max, NULL, NULL);
    max = 0.0;
    min = 1.0;
    for(int i=0; i < 256; i++)
    {
        hist[i] = hist[i]/(width * height);
        if(max <= hist[i])
            max = hist[i];
        if(min > hist[i])
            min = hist[i];
    }


    histogram_plotting(std::move(hist), file_name + "_after_equalization", max, min);
    histogramEqualized.convertTo(histogramEqualized, CV_8UC1);
    //cv::imshow("Transformed_image", histogramEqualized);
    //cv::waitKey(0);
    cv::imwrite("../spatial_filtering/results/" + file_name + "_original_image.jpg", img);
    cv::imwrite("../spatial_filtering/results/" + file_name + "_transformed_image.jpg", histogramEqualized);




}

void openCV_histEqualization(const cv::Mat& img, const std::string file_name)
{
    float range[] = {0, 256};// upper boundary is exclusive
    const float* histRange = { range };
    bool uniform=true, accumulate=false;
    cv::Mat hist;
    int histSize = 256;
    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = round( (double) hist_w/histSize );
    
    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    
    for( int i = 1; i < histSize; i++ )
    {
        cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - round(hist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - round(hist.at<float>(i)) ),
              cv::Scalar( 255, 0, 0), 2, 8, 0  );

    }
    cv::imwrite("../spatial_filtering/results/OpenCV_Methods_Results/" + file_name + "_hist.jpg", histImage);
    cv::Mat imageEqualized = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::equalizeHist(img, imageEqualized);
    cv::normalize(imageEqualized, imageEqualized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("../spatial_filtering/results/OpenCV_Methods_Results/" + file_name + "_transformed_image.jpg", imageEqualized);

} 

#endif /*__HISTOGRAM_EQUALIZATION__*/