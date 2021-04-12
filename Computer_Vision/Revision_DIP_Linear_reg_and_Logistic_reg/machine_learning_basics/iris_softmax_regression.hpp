#ifndef __IRIS_SOFTMAX__
#define __IRIS_SOFTMAX__
#include"reading_data.hpp"
#include<Eigen/Dense>
#include<map>
#include<random>
#include<Eigen/Core>
#include<vector>
 
class IRIS_SOFTMAX{
    private:
        size_t n;
        size_t m;
        Eigen::MatrixXf X_train;
        Eigen::MatrixXd y_train;
        Eigen::MatrixXf X_test;
        Eigen::MatrixXd y_test;
        std::vector<float> means;
        std::vector<float> std_deviation;
        Eigen::MatrixXf W;
        std::map<std::string, int> iris_type;
    
    public:
        IRIS_SOFTMAX(const std::string path, size_t num_parameters, const char delimiter, bool bContain_output, bool bHeaderExist, const size_t columnSkip, float train_test_ratio);
        void load_output(const std::string path, size_t num_parameters, const char delimiter, bool bContain_output, bool bHeaderExist, const size_t columnSkip);
        void fit(const float learning_rate=0.2f, int num_iter=100);
        size_t predict_class(const Eigen::VectorXf, bool, double);
        void create_confusion_matrix(bool bTrainTest);
        void predict_by_sample_number(int);

        ~IRIS_SOFTMAX();
};


IRIS_SOFTMAX::IRIS_SOFTMAX(const std::string path, size_t num_parameters, const char delimiter, bool bContain_output, bool bHeaderExist, const size_t columnSkip, float test_train_ratio=0.2f)
{
    std::pair<int, std::vector<std::string*>> complete_data = reading_anyfile(path, num_parameters, delimiter, bContain_output, bHeaderExist, columnSkip);
    this->n = (size_t)(complete_data.first * (1.0f - test_train_ratio));
    this->m = num_parameters;
    this->X_train = Eigen::MatrixXf(this->n, this->m + 1);
    this->X_test = Eigen::MatrixXf(complete_data.first - this->n, this->m + 1);
    this->y_train = Eigen::MatrixXd(this->n, 1);
    this->y_test = Eigen::MatrixXd(complete_data.first - this->n, 1);
    int output_exist = bContain_output == true? 1: 0;
    int unique_keys = 0;
    //shuffling data and getting unique classes
    Eigen::MatrixXf complete_X(complete_data.first, num_parameters + 1);
    Eigen::MatrixXd complete_y(complete_data.first, 1);
    int row = 0;

    for(auto& line: complete_data.second)
    {
        
        complete_X(row, 0) = 1.0f;//in order to have bias term
        for(int column=0; column <= num_parameters - output_exist; column++)
        {
            //std::cout << line[column] << "\t";
            complete_X(row, column + 1) = std::stof(line[column]);
        }
        if(output_exist == 1)
        {
            std::string key = line[num_parameters];
            if(this->iris_type.count(key) == 0)
            {
                //key not found
                this->iris_type.insert(std::pair<std::string, int>(key, unique_keys++));
            }
            complete_y(row, 0) = this->iris_type.at(key);
            //std::cout << complete_y(row, 0) << std::endl;

        }
        row++;
    }
    //initializing the W Matrix
    this->W = Eigen::MatrixXf::Zero(unique_keys, this->m + 1);
    this->W.row(W.rows() - 1).setZero();//We are using the last class as the reference 

    //due to that I already know the dataset, I don't need to implement a new method for loading the output
    if(output_exist == 0)
        this->load_output(path, 1, '!', false, false, 0);

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> shuffle_ind(complete_X.rows());
    shuffle_ind.setIdentity();
    std::random_shuffle(shuffle_ind.indices().data(), shuffle_ind.indices().data() + shuffle_ind.indices().size());
    complete_X = shuffle_ind * complete_X;
    complete_y = shuffle_ind * complete_y;
    //getting training data
    this->X_train = complete_X.block(0, 0, this->n, num_parameters + 1);

    this->y_train = complete_y.block(0, 0, this->n, 1);
    //getting test data
    this->X_test = complete_X.block(this->n, 0, complete_y.rows() - this->n, this->m + 1);
    this->y_test = complete_y.block(this->n, 0, complete_y.rows() - this->n, 1);
    //normalize X_train and X_test to be of mean 0 and std 1
    for(int i=1; i < this->X_train.cols(); i++)
    {
        this->means.push_back(this->X_train.col(i).mean());
        int N = this->X_train.rows();
        this->X_train.col(i) = this->X_train.col(i).array() - means[i - 1];
        this->X_test.col(i) = this->X_test.col(i).array() - means[i - 1];
        float value = std::sqrt((this->X_train.col(i).transpose() * this->X_train.col(i)).mean());

        this->std_deviation.push_back(value);
        this->X_train.col(i) = this->X_train.col(i).array() * 1.0f/this->std_deviation[i-1];
        this->X_test.col(i) = this->X_test.col(i).array() * 1.0f/this->std_deviation[i-1];
        
        std::cout << "Normalized col " << i <<" X_train: mean " << this->X_train.col(i).mean() << " std " << std::sqrt((this->X_train.col(i).transpose() * this->X_train.col(i)).mean()) << std::endl;

        std::cout << "Normalized col " << i << " X_test: mean " << this->X_test.col(i).mean() << " std " << std::sqrt((this->X_test.col(i).transpose() *  this->X_test.col(i)).mean()) << std::endl;

    }
    //std::cout << X_train << std::endl;
    // std::cout << X_test.rows() << std::endl;
    // std::cout << y_train << std::endl;
    // std::cout << W << std::endl;


    //deleting the original complete data
    for(auto&line : complete_data.second)
        delete[] line;
}
void IRIS_SOFTMAX::fit(const float learning_rate, int num_iter)
{
    auto f = [](const double element){return std::exp(element);};
    for(int iter=0; iter < num_iter; iter++)
    {
        Eigen::MatrixXf w_Xt = (this->W  * this->X_train.transpose()).array().exp();//kxn
        Eigen::MatrixXf oneHotEncoder = Eigen::MatrixXf::Zero(this->y_train.rows(), this->W.rows());//nxk
        Eigen::MatrixXf den = w_Xt.colwise().sum().array().inverse().matrix().asDiagonal();//1xn
        //std::cout << den.col(0) << std::endl;
        //std::cout << den.asDiagonal().inverse().diagonal()(0) << std::endl;
        //std::cout << den.asDiagonal().inverse().diagonal()(0) * w_Xt.transpose().row(0) << std::endl;

        w_Xt = (  den * w_Xt.transpose());//nxk scaling columns
        for(int row=0; row < this->y_train.rows(); row++)
        {
            oneHotEncoder(row, this->y_train(row, 0)) = 1.0f;

            for(int cls=0; cls < this->iris_type.size(); cls++)
            {
                if(this->y_train(row, 0) != cls)
                    w_Xt(row, cls) = 0.0f;
            }
        }
        //w_Xt = w_Xt.rowwise().maxCoeff();//getting only the column that correspond to each class which generate the highest coef, nx1
        w_Xt = w_Xt - oneHotEncoder;//nxk
        this->W = this->W - (learning_rate/this->y_train.rows()) * w_Xt.transpose() * this->X_train;
        
        //Wk sholuld be zeroed out but I don't know why it is not working.
        //this->W.row(this->W.rows() - 1).setZero();
        std::cout << "W\n" << this->W << std::endl;


    }
    
    
    std::cout << "W\n" << this->W << std::endl;
}


void IRIS_SOFTMAX::load_output(const std::string path, size_t num_parameters, const char delimiter, bool bContain_output, bool bHeaderExist, const size_t columnSkip)
{
    //No need for it because I already know that the output exist in the dataset
}

size_t IRIS_SOFTMAX::predict_class(const Eigen::VectorXf x, bool verbose=false, double true_value=0)
{

    Eigen::VectorXf x_new = x;
    const int num_classes = this->iris_type.size();
    if(x_new.rows() == 1)
        x_new = x_new.transpose();
    size_t cls = 0;
    float* all_classes = new float[num_classes];
    float sum = (this->W * x_new).array().exp().array().sum();//sum will never be zero, unless w^tx is less than 0 which contradict the rules of dot product.
    float max = 0.0f;
    float prob_sum = 0.0f; 
    for(int row=0; row < num_classes; row++)
    {

        all_classes[row] = std::exp(this->W.row(row)* x_new)/sum;
        prob_sum += all_classes[row];
        if(max <= all_classes[row])
        {
            cls = row;
            max = all_classes[row];
        }    
    }
    if(verbose)
    {
        int counter =0;
        std::string class_value;
        for(auto& row: this->iris_type)
        {
            if(true_value == row.second)
            {
                class_value = row.first;
                break;
            }
        }
        std::cout << "True class " << class_value << std::endl;
        for(auto& row: this->iris_type)
        {
            std::cout << row.first << " with propability " << all_classes[counter++] << "\t";
        }
        std::cout << std::endl;
    }
    delete[] all_classes;
    return cls;
}

void IRIS_SOFTMAX::create_confusion_matrix(bool bTrainTest=true)
{
    Eigen::MatrixXf X = this->X_train;
    Eigen::MatrixXd y = this->y_train;
    const size_t num_classes = this->iris_type.size();

    if(!bTrainTest)
    {
        X = this->X_test;
        y = this->y_test;
    }    
    std::map<double, std::vector<int>> confusion_matrix;
    for(int row=0; row < X.rows(); row++)
    {
        //std::cout << X.row(row) << std::endl;
        size_t cls = this->predict_class(X.row(row));
        //std::cout << cls << "\t" << y(row, 0) << std::endl;
        if(confusion_matrix.count(y(row, 0)) == 0)
        {
            std::vector<int> tmp = {0, 0, 0};
            confusion_matrix.insert( std::pair<double, std::vector<int>>(y(row, 0), tmp) );
        }
        confusion_matrix[y(row, 0)][(int)cls] += 1;
    }
    std::cout << "Predicted Class by Softmax model\n" << "\t";
    for(auto& key: this->iris_type)
        std::cout << key.first << "\t";
    std::cout << std::endl;
    for(auto& row: confusion_matrix)
    {
        std::cout << "\t";
        for(auto& col: row.second)
            std::cout << col << "\t\t";
        std::cout << std::endl;
    }
}

void IRIS_SOFTMAX::predict_by_sample_number(int sample)
{
    if(sample > this->X_train.rows())
    {
        std::cout << "The sample index is bigger the the number rows in X_train, choose an index that is less than " << this->X_train.rows() << std::endl;
        sample = this->X_train.rows() - 1;
    }
    std::cout << "Prediction for sample# " << sample <<std::endl;
    std::cout << "X:" << this->X_train.row(sample) << std::endl;
    this->predict_class(this->X_train.row(sample), true, this->y_train(sample, 0));
}

IRIS_SOFTMAX::~IRIS_SOFTMAX()
{

}



#endif /*__IRIS_SOFTMAX__*/