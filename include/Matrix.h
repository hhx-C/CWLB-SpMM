#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen>
#include "config.h"
#include "../include/config.h"
#include <map>
#include <vector>
using namespace std;

// Standard CSR storage: row offsets index the nonzero column/value arrays.
class CSR
{
public:
    vector<int> RowOffset;
    vector<int> ColInd;
    vector<float> Value;

    CSR(const Defined::RowMajorMatrixXf &adj);
    CSR(vector<int> RowOffset, vector<int> ColInd, vector<float> Value);
    CSR(){}
    ~CSR(){};
};

// Degree-reordered CSR with the row mapping needed for output restoration.
class CSR_Order
{
public:
    vector<int> Origin_local;  // Reordered row -> original graph-node ID.
    vector<int> RowOffset;
    vector<int> ColInd;
    vector<float> Value;
};
