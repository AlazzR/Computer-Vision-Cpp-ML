#ifndef __Linear_regression__
#define __Linear_regression__
#include<iostream>
#include<Eigen/Dense>
#include<utility>
#include<string>
#include"generate_data.hpp"
#include<random>
#include <Eigen/Cholesky>
#include<cmath>

Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> shuffling_data(size_t n)
{
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> shuffled(n);
    shuffled.setIdentity();
    std::random_shuffle(shuffled.indices().data(), shuffled.indices().data() + shuffled.indices().size());
    return shuffled;
}

float compute_mse(const Eigen::VectorXf& pred, const Eigen::VectorXf& y)
{
    float value = (1.0f/pred.rows()) * ((y - pred).transpose() * (y-pred)).sum();
    return value;

}
Eigen::VectorXf gradient_descent(const Eigen::MatrixXf& X, const Eigen::MatrixXf& y, float learning_rate, size_t num_rows)
{
    //I already assume that X is padded with 1's
    Eigen::VectorXf B = Eigen::VectorXf::Random(X.cols());
    B(0) = 0.0f;
    size_t blocks_size = (size_t) (X.rows()/num_rows);
    //in case if the num_rows was larger than the number of rows of X
    if(blocks_size == 0)
    {
        blocks_size = 1;
        num_rows = X.rows();
    }
    int counter = 0;
    Eigen::MatrixXf block_of_X = Eigen::MatrixXf::Zero(num_rows, X.cols());
    Eigen::VectorXf block_of_y = Eigen::VectorXf::Zero(num_rows);
    for(int iter = 0; iter < 100; iter++)
    {
        counter = 0;
        block_of_X = Eigen::MatrixXf(num_rows, X.cols());
        block_of_y = Eigen::VectorXf(num_rows);

        for(int block=0; block < blocks_size; block++)
        {
            //taking a slice from X, it appears that I don't have Eigen::seq
            for(int row=0; row < num_rows; row++)
            {
                block_of_X.row(row) = X.row(row + counter);
                block_of_y.row(row) = y.row(row + counter);
            }
            //updating B
            //std::cout << block_of_y << std::endl;
            // std::cout << "x\t" << block_of_X <<"\n" << std::endl;
            // std::cout << "xT\t" << block_of_X.transpose() <<"\n" << std::endl;

            B = B - learning_rate * 1.0f/(block_of_X.rows()) * block_of_X.transpose() * (block_of_X * B - block_of_y);
            Eigen::VectorXf pred = Eigen::VectorXf(block_of_X.rows());
            pred = block_of_X * B;
            float error = compute_mse(pred, block_of_y);
            std::cout << "Step: " << block + iter * blocks_size  << " error: " << error << std::endl;


            counter += num_rows;
        } 
        if(counter < X.rows())
        {
            int rest_of_rows = X.rows() - counter;
            block_of_X = Eigen::MatrixXf(rest_of_rows, X.cols());
            block_of_y = Eigen::VectorXf(rest_of_rows);

            for(int row=0; row < rest_of_rows; row++)
            {
                block_of_X.row(row) = X.row(row + counter);
                block_of_y.row(row) = y.row(row + counter);
            }
            //updating B
            //std::cout << "rows: " << block_of_X.rows() << std::endl;
            B = B - learning_rate * block_of_X.transpose() * ( block_of_X * B - block_of_y );
            Eigen::VectorXf pred = Eigen::VectorXf(rest_of_rows);
            pred = block_of_X * B;
            float error = compute_mse(pred, block_of_y);
            std::cout << "Step: " << blocks_size + 1 + + iter * blocks_size << " error: " << error << std::endl;

        }
    }
   
    //std::cout << "B\n " << B << std::endl; 

    return B;
}
float estimate_variance(const Eigen::MatrixXf& X, const Eigen::MatrixXf& y, const Eigen::VectorXf& B)
{
    //I will use the biased estimate in which I willn't divide by n-1
    float sigma = (1.0f/y.rows()) * ((y - X * B).transpose() * (y - X * B)).sum();
    return sigma;
}
void make_prediction(const Eigen::VectorXf& B, const Eigen::MatrixXf& x, float sigma, const float true_y);

//I will only deal with metric/conttinuous data
void Linear_regression_training(const std::string file_name, float lr, int num_rows)
{
    std::pair<Eigen::MatrixXf, Eigen::MatrixXf> data = load_csv_for_LinearR(file_name);
    Eigen::MatrixXf X = data.first;
    Eigen::MatrixXf X_with_pad = Eigen::MatrixXf(X.rows(), X.cols() + 1 );
    //pad X for the intercept term
    Eigen::VectorXf b0 = Eigen::VectorXf::Ones(X.rows());
    //shifting columns
    X_with_pad.col(0) = b0;
    for(int i=1; i < X_with_pad.cols(); i++)
    {
        X_with_pad.col(i) = X.col(i - 1);
    }
    Eigen::MatrixXf y = data.second;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> shuffled_indeces(shuffling_data(X.rows()));
    //shuffling data
    X_with_pad = shuffled_indeces * X_with_pad;
    y = shuffled_indeces * y;
    //standard normalize X
    // float* means = new float[X_with_pad.cols() - 1];
    // float* std_ = new float[X_with_pad.cols() - 1];
    // Eigen::VectorXf tmp = Eigen::VectorXf::Ones(X_with_pad.rows());

    // for(int i=1; i < X_with_pad.cols(); i++)
    // {
    //     means[i - 1] = X_with_pad.col(i).mean();
    //     X_with_pad.col(i) = X_with_pad.col(i) - tmp * means[i -1];
    //     std_[i - 1] = 1/X_with_pad.rows() * (X_with_pad.col(i).transpose() * X_with_pad.col(i)).sum();
    //     tmp = X_with_pad.col(i);
    //     std::cout << X_with_pad << std::endl;
    //     std::cout << "m " << means[i - 1] << " std: " << std_[i -1] << std::endl;
    //     tmp = tmp / std_[i-1];
    //     std::cout << tmp << std::endl;
    //     X_with_pad.col(i) = tmp;
    // }
    // float meany, std_y;
    // //standard normalize y
    // meany = y.mean();
    // y = y - tmp * meany;
    // std_y = 1/y.rows() * (y.transpose() * y).sum();
    // tmp = y;
    // y = tmp/std_y;
    //use meany, std_y, means and std_ for prediction stage.

    //Finding Value of B
    Eigen::VectorXf B = gradient_descent(X_with_pad, y, lr, num_rows);
    Eigen::VectorXf B_ols = (X_with_pad.transpose() * X_with_pad).inverse() * X_with_pad.transpose() * y;
    
    std::cout << "Mini-batch GD estimat of of B: \n" << B << std::endl;
    std::cout << "Error in prediction GD: " << compute_mse(X_with_pad * B, y) << std::endl;
    std::cout << "Sigma Estimate by GD: " << estimate_variance(X_with_pad, y, B) << std::endl;
    std::cout << "OLS of B: \n" << B_ols << std::endl;
    std::cout << "Error in prediction OLS: " << compute_mse(X_with_pad * B_ols, y) << std::endl;
    // std::cout << X_with_pad << std::endl;
    // std::cout << y << std::endl;
    make_prediction(B, X_with_pad.row(10).transpose(), estimate_variance(X_with_pad, y, B), y.row(10).sum());

}

void make_prediction(const Eigen::VectorXf& B, const Eigen::MatrixXf& x, float sigma, const float true_y)
{
    auto f = [](float pred, float true_v){

        return std::sqrt((pred - true_v) * (pred - true_v));
    };
    std::cout << "B: " << B.transpose().cols() << std::endl;
    std::cout << "x: " << x.rows() << std::endl;
    float y = (B.transpose() * x).sum();
    std::cout << "Prediciton: " << y << " true value: " << true_y << " rms: " << f(y, true_y) << std::endl;

}



#endif /*__Linear_regression__*/