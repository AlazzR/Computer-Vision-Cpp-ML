#ifndef __GENERATION_OF_DATA_STEP__
#define __GENERATION_OF_DATA_STEP__
#include<iostream>
#include<vector>
#include<fstream>
#include<random>
#include<string>
#include<Eigen/Dense>
#include<utility>// for pair

void generate_data(size_t n, size_t m)
{
    //n is #examples and m is #predictors
    float y;
    float intercept = 3.0f;
    float true_sigma = 5.0f;
    float a = 3.5;//multiply by the parameters
    float sign= -1;//alternate in the sign of the parameters of the model.
    float* parameters_true_values = new float[m];
    for(int i=0; i < m; i++)
    {
        parameters_true_values[i] = sign * a * (i + 1);
        sign *= -1;
    }
    std::default_random_engine generator;
    std::normal_distribution<float> normal(0.0f, true_sigma);//mean for noise is 0.0 and variance is 25
    std::vector<std::string> data;
    Eigen::VectorXf x = Eigen::VectorXf::Zero(n);
    for(int i=0; i < n; i++)
        x(i, 0) = normal(generator);
        
    for(int sample=0; sample < n; sample++)
    {
        float sum = 0.0f;
        std::string predictors_values = "";
        for(int b=1; b <= m; b++)
        {
            sum += parameters_true_values[b-1] * x(sample, 0);
            predictors_values += std::to_string(x(sample, 0)) + ",";
        }
        y = intercept + sum + normal(generator);
        predictors_values += std::to_string(y) + "\n";
        data.push_back(predictors_values);
    }
    std::ofstream file("../machine_learning_basics/linear_regression_data.txt", std::ios::out);
    if(file.is_open())
    {
        for(auto& line: data)
            file << line;
        file.close();
    }
    else{
        std::cout << "File not found linear_regression_data.txt\n";
    }
    std::ofstream info_data("../machine_learning_basics/information_about_data.txt", std::ios::out);
    if(info_data.is_open())
    {
        info_data << "#samples: " << data.size() << "\n";
        info_data << "true_sigma: " << true_sigma << "\n";
        info_data << "Number of parameter without b0 " << m << std::endl;
        info_data << "intercept: " << intercept << "\n";
        info_data << "Parameters: \n";
        for(int i=1; i <= m; i++)
            info_data << "b" + std::to_string(i) << " " << parameters_true_values[i-1] << "\n";
        info_data.close();
    }
    else{
        std::cout << "File not found information_about_data.txt\n";
    }
    delete[] parameters_true_values;
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> load_csv_for_LinearR(const std::string file_name)
{
    std::ifstream data("../machine_learning_basics/" + file_name, std::ios::in);
    std::string line;
    std::vector<float> y;
    std::vector<std::vector<float>> X;
    if(data.is_open())
    {
        //reading line by line
        while(std::getline(data, line))
        {
            std::vector<float> d;
            char delimiter = ',';
            size_t pos_old = 0;
            size_t pos = 0;
            //reading values in the line
            while((pos = line.substr(pos_old, line.length() - pos_old).find(delimiter)) != std::string::npos)
            {
                d.push_back(std::stof(line.substr(pos_old, pos - pos_old)));
                pos_old  += pos + 1;
            }
            y.push_back(std::stof(line.substr(pos_old, line.length() - pos_old)));
            X.push_back(d);
        }
        data.close();

    }
    else{
        std::cout << "This file doesn't exist " << file_name << std::endl;
        throw "THis file doesn't exist in the directory";
    }
    const int rows = y.size();
    const int cols = X[0].size();
    //printing y
    // for(auto& val: y)
    //     std::cout << val << std::endl;
    //printing X
    // for(auto& row: X)
    // {
    //     for(auto& col: row)
    //         std::cout << col << ",";
    //     std::cout << std::endl;
    // }
    float* ptrY = y.data();
    Eigen::MatrixXf X_object(rows, cols);
    //copying X, next time I need to stop using vector<vector> and to only use vector<float*> due to the memory hassle that is caused by vector
    for(int row=0; row< rows; row++)
        for(int col=0; col < cols; col++)
            X_object(row, col) = X[row][col];
    Eigen::MatrixXf y_object =  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ptrY, rows, 1);

    // std::cout << X_object << std::endl;
    //std::cout << y_object << std::endl;
    std::pair<Eigen::MatrixXf, Eigen::MatrixXf> result(X_object, y_object);
    return result;
}

#endif /*__GENERATION_OF_DATA_STEP__*/