#ifndef __DIABETIC_CLASSIFICATION__
#define __DIABETIC_CLASSIFICATION__
#include<opencv2/opencv.hpp>
#include<opencv2/ml.hpp>
#include<Eigen/Dense>
#include<fstream>
#include<string>
#include<vector>

void load_csv(std::string file_path, cv::Mat& X, cv::Mat& y, char delimiter)
{
    std::ifstream input = std::ifstream(file_path, std::ios::in);
    std::string line;
    if(input.is_open())
    {
        while(std::getline(input, line))
        {
            size_t pos = 0;
            size_t pos_old = 0;
            std::vector<float> row;
            float output;
            std::cout << line << std::endl;
            while((pos = line.substr(pos_old, line.length() - pos_old).find(delimiter)) != std::string::npos)
            {
                row.push_back(std::atof(line.substr(pos_old, pos - pos_old).c_str()));
                pos_old += pos + 1;
            }
            X.push_back(row);//[X] = 1 x (8 * 768)
            y.push_back(std::stof(line.substr(pos_old, line.length() - pos_old)));//[y] = 1 x768
        }

        input.close();
    }
    else
    {
        std::cout << "This file doesn't exist " << file_path << std::endl;
        throw "THis file doesn't exist in the directory";
    }
}

class Diabetic_Classification
{
    private:
        cv::Mat X;
        cv::Mat y;
        cv::Mat X_train_normalized;
        cv::Mat X_test_normalized;
        cv::Mat y_train;
        cv::Mat y_test;
        cv::Mat means;
        cv::Mat stddev;

    public:
        Diabetic_Classification(const cv::Mat& X, const cv::Mat& y)
        {  
            this->X = X.clone();
            this->y = y.clone();
            this->X = this->X.reshape(1, 768);//8x768, response 1x768
            this->X.convertTo(this->X, CV_32F);
            std::cout << this->X.size() << std::endl;
            std::cout << this->y.size() << std::endl;
        }

        void normalize_and_split_data(float split_train_to_test_ratio =0.8f)
        {
            cv::Ptr<cv::ml::TrainData> training_data = cv::ml::TrainData::create(this->X, cv::ml::ROW_SAMPLE, this->y);
            training_data->setTrainTestSplitRatio(split_train_to_test_ratio);//it will shuffle it
            this->X_train_normalized = training_data->getTrainSamples();
            this->X_test_normalized = training_data->getTestSamples();
            this->y_train = training_data->getTrainResponses();
            this->y_test = training_data->getTestResponses();

            //getting means
            for(int i=0; i < this->X_train_normalized.cols; i++)
            {
                cv::Mat mean; cv::Mat sigma;
                cv::meanStdDev(this->X_train_normalized.col(i), mean, sigma);
                this->X_train_normalized.col(i) = (this->X_train_normalized.col(i) - mean)/sigma;
                this->X_test_normalized.col(i) = (this->X_test_normalized.col(i) - mean)/sigma;
                this->means.push_back(mean);
                this->stddev.push_back(sigma);
            }
            std::cout << "COL mean: " << cv::mean(this->X_train_normalized.col(0)) << std::endl;
            // std::cout << training_data->getResponses().rows << std::endl;
            // std::cout << training_data->getTrainSamples().size() << std::endl;
            std::cout << this->y_train.size() << std::endl;
            std::cout << this->X_train_normalized.size() << std::endl;
        }

        void LogisticRegression(bool boolTrain=true)
        {
            this->y_train.convertTo(this->y_train, CV_32F);
            this->y_test.convertTo(this->y_test, CV_32F);
            if(boolTrain)
            {
                cv::Ptr<cv::ml::LogisticRegression> lr = cv::ml::LogisticRegression::create();
                lr->setLearningRate(0.01);
                lr->setIterations(1000);
                lr->setRegularization(cv::ml::LogisticRegression::REG_L2);
                lr->setTrainMethod(cv::ml::LogisticRegression::BATCH);
                lr->setMiniBatchSize(264);

                lr->train(this->X_train_normalized, cv::ml::ROW_SAMPLE, this->y_train);
                lr->save("../../logistic_regression.xml");
                std::cout << "X_train confusion matrix\n";
                this->confusion_matrix<cv::ml::LogisticRegression, float>("../../logistic_regression.xml", this->X_train_normalized, this->y_train);

            }
            else{
                std::cout << "X_test confusion matrix\n";
                this->confusion_matrix<cv::ml::LogisticRegression, float>("../../logistic_regression.xml", this->X_test_normalized, this->y_test);
            }

        }

        void SVM(bool boolTrain=true)
        {
            this->y_train.convertTo(this->y_train, CV_32S);
            this->y_test.convertTo(this->y_test, CV_32S);
            if(boolTrain)
            {
                for(int i=0; i < this->y_train.rows; i++)
                {
                    if(this->y_train.at<signed int>(i, 0) == 0)
                        this->y_train.at<signed int>(i, 0) = -1;
                }
                cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
                svm->setType(cv::ml::SVM::C_SVC);
                svm->setKernel(cv::ml::SVM::RBF);
                svm->setC(10);
                svm->setGamma(0.8);
                // svm->setNu(0.1);
                //svm->setDegree(10);
                svm->train(this->X_train_normalized, cv::ml::ROW_SAMPLE, this->y_train);

                svm->save("../../svm.xml");
                std::cout << "X_train confusion matrix\n";
                this->confusion_matrix<cv::ml::SVM, signed int>("../../svm.xml", this->X_train_normalized, this->y_train);

            }
            else{
                for(int i=0; i < this->y_test.rows; i++)
                {
                    if(this->y_test.at<signed int>(i, 0) == 0)
                        this->y_test.at<signed int>(i, 0) = -1;
                }
                std::cout << "X_test confusion matrix\n";
                this->confusion_matrix<cv::ml::SVM, signed int>("../../svm.xml", this->X_test_normalized, this->y_test);
            }

        }

        template<typename T, typename S>
        void confusion_matrix(std::string model_path, cv::Mat& X, cv::Mat& y, int k=2)
        {
            std::vector<float> predicted;
            std::vector<float> true_value;
            cv::Ptr<T> model = cv::Algorithm::load<T>(model_path);
            std::cout << model << std::endl;
            for(int i=0; i < X.rows; i++)
            {
                predicted.push_back(model->predict(X.row(i)));
                true_value.push_back(y.at<S>(i, 0));
            }

            cv::Mat confusion_matrix = cv::Mat::zeros(cv::Size(k, k), CV_32F);
            std::cout << predicted.size() << std::endl;

            for(int row=0; row < predicted.size(); row++)
            {
                //std::cout << "True: " << true_value[row] << std::endl;
                //std::cout << "Predicted: " << predicted[row] << std::endl;

                int t_row = true_value[row];
                int t_col = predicted[row];
                //to deal with svm classifications
                if(t_row == -1)
                    t_row = 0;
                if(t_col == -1)
                    t_col = 0;
                confusion_matrix.at<float>(t_row , t_col) += 1.0f;
            }
            std::cout << "confusion_matrix: \n" << confusion_matrix << std::endl;
            std::cout << "Accuracy: "<<cv::sum(confusion_matrix.diag())[0]/cv::sum(confusion_matrix)[0] << std::endl;

        }

};


#endif /*__DIABETIC_CLASSIFICATION__*/