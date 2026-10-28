// Constructors for dense-to-CSR conversion and existing CSR arrays.
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen>
#include "config.h"
#include "../include/config.h"
#include <map>
#include <vector>
#include "Matrix.h"
using namespace std;

CSR::CSR(const Defined::RowMajorMatrixXf &adj)
{

    cout << "Converting the dense adjacency matrix to CSR." << endl;

    if (adj.rows() != adj.cols()) {
        throw std::invalid_argument("Adjacency matrix must be square");
    }
    int adjRows = adj.rows();
    int adjCols = adj.cols();
    int count = 0;
    // Record each row start while appending its nonzero entries.
    for (int i = 0; i < adjRows; i++)
    {
        this->RowOffset.push_back(count);
        for (int j = 0; j < adjCols; j++)
        {
            if(adj(i, j) != 0){
                this->ColInd.push_back(j);
                this->Value.push_back(adj(i, j));
                count++;
            }
        }
    }
    this->RowOffset.push_back(count);

    // for (int i = 0; i < RowOffset.size(); ++i) {
    //     std::cout << RowOffset[i] << " ";
    // }
    // std::cout << " \n";
    // for (int i = 0; i < ColInd.size(); ++i) {
    //     std::cout << ColInd[i] << " ";
    // }
    // std::cout << " \n";
    // for (int i = 0; i < Value.size(); ++i) {
    //     std::cout << Value[i] << " ";
    // }
    // std::cout << " \n";
}

CSR::CSR(vector<int> RowOffset, vector<int> ColInd, vector<float> Value){
    this->RowOffset = RowOffset;
    this->ColInd = ColInd;
    this->Value = Value;
}
