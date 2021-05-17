#ifndef __READING_DATA__
#define __READING_DATA__
#include<iostream>
#include<fstream>
#include<string>
#include<experimental/filesystem>
#include<unistd.h>
#include<utility>// for pair
/*
    This method will be used for reading files and hopefully it work with every file that I am going to deal with :)
*/

std::pair<int, std::vector<std::string*>> reading_anyfile(const std::string path, size_t num_parameters, const char delimiter, bool bContain_output, bool bHeaderExist, const size_t columnSkip)
{
    std::ifstream file = std::ifstream(path, std::ios::in);
    std::string line;
    std::vector<std::string*> data;
    int output_exist = bContain_output == true? 1: 0;

    if(!file.is_open())
        throw std::runtime_error("This file " + path + " doesn't exist in the current directory" + std::experimental::filesystem::current_path().string() + "\n");
    size_t n = 0;
    while(std::getline(file, line))
    {
        if(n == 0 && bHeaderExist)
        {
            //skip header
            n++;
            continue;
        }
        size_t pos = 0;
        size_t pos_old = 0;
        std::string* value = new std::string[num_parameters + output_exist];
        int counter = 0;
        size_t row_column_skip = columnSkip;
        while((pos = line.substr(pos_old, line.length() - pos_old).find(delimiter
        )) != std::string::npos)
        {
            if(row_column_skip == 0)
            {
                value[counter] = line.substr(pos_old, pos);
                counter++;
            }
            else{
                row_column_skip--;
            }
            pos_old += pos + 1;

        }
        value[counter] = line.substr(pos_old, line.length() - pos_old);
        data.push_back(value);
        n++;
    }
    //printing Details
    // if(n != 0 )
    // {
    //     for(auto& obs: data)
    //     {
    //         for(int i=0; i < num_parameters + output_exist; i++)
    //             std::cout << i << " "<<obs[i] << "\t";
    //         std::cout << std::endl;
    //         delete[] obs;
    //     }
    // }


    file.close();
    std::pair<int, std::vector<std::string*>> output;
    if(bHeaderExist)
        n--;
    output.first = n;
    output.second = data;

    return output;
}



#endif /*__READING_DATA__*/