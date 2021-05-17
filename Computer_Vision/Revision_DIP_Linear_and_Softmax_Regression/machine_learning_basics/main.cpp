#include"generate_data.hpp"
#include"Linear_regression_Model.hpp"
#include"reading_data.hpp"
#include"iris_softmax_regression.hpp"

/*
    There is something wrong in the softmax regression, it doesn't behave properly. Because the last row of the W matrix should be zero, but, in my project when I zeroed out this row, I get really bad prediction for the Kth class.
*/
int main(int argc, char* argv[])
{
    /* Generating data to do training on for the linear regression model */
    generate_data(10000, 1);
    //Linear_regression_training("linear_regression_data.txt", 0.001, 32);
    //reading_anyfile("../datasets/Iris.csv", 4, ',', true, true, 1);
    IRIS_SOFTMAX iris = IRIS_SOFTMAX("../datasets/Iris.csv", 4, ',', true, true, 1);
    iris.fit(1.0f, 100);
    iris.create_confusion_matrix();
    iris.predict_by_sample_number(3);
    iris.create_confusion_matrix(false);

    
    return 0;
}