#include <float.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <vector>
#include <dirent.h>
#include <chrono>
// #include <opencv2/opencv.hpp>

std::vector<std::string> getFilenames(const std::string& folderPath) {
    std::vector<std::string> filenames;
    DIR* dir = opendir(folderPath.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_REG) { // Regular file
                filenames.emplace_back(entry->d_name);
            }
        }
        closedir(dir);
        std::sort(filenames.begin(), filenames.end());
    } else {
        std::cerr << "Error opening directory: " << folderPath << std::endl;
    }
    return filenames;
}

void loadWeights(const std::string fileLocation, float* weights,int weightsSize, float* bias, int biasSize){
    std::ifstream file(fileLocation);
    if(file.is_open()){
        for(int i=0; i<weightsSize; i++){
            file >> weights[i];
        }
        for(int i=0; i<biasSize; i++){
            file >> bias[i];
        }
    }
    file.close();
}

__global__ void convolutionCuda(float* input, float* output, float* weights, float* bias, 
                int input_dim, int kernel_dim, int kernel_depth, int num_kernels){
    
    int output_dim = input_dim - kernel_dim + 1;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int kernel = blockIdx.x;
    float sum = 0.0f;
    for(int i=0;i<kernel_depth;i++){
        for(int j=0;j<kernel_dim;j++){
            for(int k=0;k<kernel_dim;k++){
                int weight_index = kernel*kernel_depth*kernel_dim*kernel_dim + i*kernel_dim*kernel_dim + j*kernel_dim + k;
                int input_index = i*input_dim*input_dim + (row+j)*input_dim + (col+k);
                sum += weights[weight_index]*input[input_index];
            }
        }
    }
    output[kernel*output_dim*output_dim + row*output_dim + col] = sum + bias[kernel];
}

__global__ void maxPoolingCuda(float* input, float* output, int input_dim, int num_kernals, int kernel_dim){
    int output_dim = input_dim / kernel_dim;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int kernel = blockIdx.x;
    float mx = -FLT_MAX;
    for(int i=0;i<kernel_dim;i++){
        for(int j=0;j<kernel_dim;j++){
            int row_index = row*kernel_dim + i;
            int col_index = col*kernel_dim + j;
            mx = fmaxf(mx, input[kernel*input_dim*input_dim + row_index*input_dim + col_index]);
        }
    }
    output[kernel*output_dim*output_dim + row*output_dim + col] = mx;
}

__global__ void reluCuda(float* input, int size){
    int index = threadIdx.x;
    input[index] = max(0.0f, input[index]);
}

__global__ void tanhCuda(float* input, int size){
    int index = threadIdx.x;
    input[index] = tanh(input[index]);
}

void softmax(float* input, int size){
    float sum = 0;
    for(int i=0;i<size;i++){
        input[i] = exp(input[i]);
        sum += input[i];
    }
    for(int i=0;i<size;i++){
        input[i] = input[i]/sum;
    }
}

void sigmoid(float* input, int size){
    for(int i=0;i<size;i++){
        input[i] = 1/(1+exp(-input[i]));
    }
}

int main(int argc, char** argv){
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    // initialize the weights and biases
    int image_dim = 28;
    int kernel_dim = 5;
    int pool_dim = 2;
    int conv_1_num_kernels = 20;
    int conv_2_num_kernels = 50;
    int fc_1_num_neurons = 500;
    int fc_2_num_neurons = 10;
    int conv_1_weights_size = 20*25;
    int conv_1_bias_size = 20;
    int conv_2_weights_size = 50*5*5*20;
    int conv_2_bias_size = 50;
    int fc_1_weights_size = 500*4*4*50;
    int fc_1_bias_size = 500;
    int fc_2_weights_size = 10*500;
    int fc_2_bias_size = 10;
    float* conv_1_weights = new float[conv_1_weights_size];
    float* conv_1_bias = new float[conv_1_bias_size];
    float* conv_2_weights = new float[conv_2_weights_size];
    float* conv_2_bias = new float[conv_2_bias_size];
    float* fc_1_weights = new float[fc_1_weights_size];
    float* fc_1_bias = new float[fc_1_bias_size];
    float* fc_2_weights = new float[fc_2_weights_size];
    float* fc_2_bias = new float[fc_2_bias_size];
    loadWeights("weights/conv1.txt",conv_1_weights,conv_1_weights_size,conv_1_bias,conv_1_bias_size);
    loadWeights("weights/conv2.txt",conv_2_weights,conv_2_weights_size,conv_2_bias,conv_2_bias_size);
    loadWeights("weights/fc1.txt",fc_1_weights,fc_1_weights_size,fc_1_bias,fc_1_bias_size);
    loadWeights("weights/fc2.txt",fc_2_weights,fc_2_weights_size,fc_2_bias,fc_2_bias_size);

    float* conv_1_weights_d, *conv_2_weights_d, *fc_1_weights_d, *fc_2_weights_d;
    float *conv_1_bias_d, *conv_2_bias_d, *fc_1_bias_d, *fc_2_bias_d;
    cudaMalloc(&conv_1_weights_d, conv_1_weights_size*sizeof(float));
    cudaMalloc(&conv_2_weights_d, conv_2_weights_size*sizeof(float));
    cudaMalloc(&fc_1_weights_d, fc_1_weights_size*sizeof(float));
    cudaMalloc(&fc_2_weights_d, fc_2_weights_size*sizeof(float));
    cudaMalloc(&conv_1_bias_d, conv_1_bias_size*sizeof(float));
    cudaMalloc(&conv_2_bias_d, conv_2_bias_size*sizeof(float));
    cudaMalloc(&fc_1_bias_d, fc_1_bias_size*sizeof(float));
    cudaMalloc(&fc_2_bias_d, fc_2_bias_size*sizeof(float));
    cudaMemcpy(conv_1_weights_d, conv_1_weights, conv_1_weights_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv_2_weights_d, conv_2_weights, conv_2_weights_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc_1_weights_d, fc_1_weights, fc_1_weights_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc_2_weights_d, fc_2_weights, fc_2_weights_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv_1_bias_d, conv_1_bias, conv_1_bias_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv_2_bias_d, conv_2_bias, conv_2_bias_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc_1_bias_d, fc_1_bias, fc_1_bias_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(fc_2_bias_d, fc_2_bias, fc_2_bias_size*sizeof(float), cudaMemcpyHostToDevice);


    // get all the filenames in the folder in sorted order
    std::vector<std::string> filenames = getFilenames("pre-proc-img");

    // int counter = 0;
    // iterate over all images in the folder
    for(int i=0;i<filenames.size();i++){

        // read the image and convert it to 1D array
        std::string imageLocation = "pre-proc-img/" + filenames[i];
        float* image_1d = new float[image_dim*image_dim];
        float* image_bias = nullptr;
        loadWeights(imageLocation, image_1d, image_dim*image_dim, image_bias, 0);

        float* image_1d_d;
        cudaMalloc(&image_1d_d, image_dim*image_dim*sizeof(float));
        cudaMemcpy(image_1d_d, image_1d, image_dim*image_dim*sizeof(float), cudaMemcpyHostToDevice);

        // Convolution 1
        int conv_1_output_dim = image_dim - kernel_dim + 1;
        int conv_1_kernel_depth = 1;
        float* conv_1_output_d;
        cudaMalloc(&conv_1_output_d, conv_1_num_kernels*conv_1_output_dim*conv_1_output_dim*sizeof(float));
        convolutionCuda<<<conv_1_num_kernels,dim3(conv_1_output_dim,conv_1_output_dim)>>>
            (image_1d_d, conv_1_output_d, conv_1_weights_d, conv_1_bias_d, image_dim, kernel_dim, conv_1_kernel_depth, conv_1_num_kernels);
        
        // // Max Pooling 1
        int pool_1_output_dim = conv_1_output_dim / pool_dim;
        float* pool_1_output_d;
        cudaMalloc(&pool_1_output_d, conv_1_num_kernels*pool_1_output_dim*pool_1_output_dim*sizeof(float));
        maxPoolingCuda<<<conv_1_num_kernels,dim3(pool_1_output_dim,pool_1_output_dim)>>>
            (conv_1_output_d, pool_1_output_d, conv_1_output_dim, conv_1_num_kernels, pool_dim);
        
        // Convolution 2
        int conv_2_output_dim = pool_1_output_dim - kernel_dim + 1;
        int conv_2_kernel_depth = conv_1_num_kernels;
        float* conv_2_output_d;
        cudaMalloc(&conv_2_output_d, conv_2_num_kernels*conv_2_output_dim*conv_2_output_dim*sizeof(float));
        convolutionCuda<<<conv_2_num_kernels,dim3(conv_2_output_dim,conv_2_output_dim)>>>
            (pool_1_output_d, conv_2_output_d, conv_2_weights_d, conv_2_bias_d, pool_1_output_dim, kernel_dim, conv_2_kernel_depth, conv_2_num_kernels);

        // Max Pooling 2
        int pool_2_output_dim = conv_2_output_dim / pool_dim;
        float* pool_2_output_d;
        cudaMalloc(&pool_2_output_d, conv_2_num_kernels*pool_2_output_dim*pool_2_output_dim*sizeof(float));
        maxPoolingCuda<<<conv_2_num_kernels,dim3(pool_2_output_dim,pool_2_output_dim)>>>
            (conv_2_output_d, pool_2_output_d, conv_2_output_dim, conv_2_num_kernels, pool_dim);

        // Fully Connected 1
        float* fc_1_output_d;
        cudaMalloc(&fc_1_output_d, fc_1_num_neurons*sizeof(float));
        int fc_kernel_dim = 4;
        int fc_1_kernel_depth = conv_2_num_kernels;
        convolutionCuda<<<fc_1_num_neurons,dim3(1,1)>>>
            (pool_2_output_d, fc_1_output_d, fc_1_weights_d, fc_1_bias_d, pool_2_output_dim, fc_kernel_dim, fc_1_kernel_depth, fc_1_num_neurons);
        reluCuda<<<1,fc_1_num_neurons>>>(fc_1_output_d, fc_1_num_neurons);

        // Fully Connected 2
        float* fc_2_output_d;
        cudaMalloc(&fc_2_output_d, fc_2_num_neurons*sizeof(float));
        int fc_2_kernel_depth = fc_1_num_neurons;
        int fc_2_kernel_dim = 1;
        convolutionCuda<<<fc_2_num_neurons,dim3(1,1)>>>
            (fc_1_output_d, fc_2_output_d, fc_2_weights_d, fc_2_bias_d, 1, fc_2_kernel_dim, fc_2_kernel_depth, fc_2_num_neurons);
        
        // Softmax
        float* fc_2_output = new float[fc_2_num_neurons];
        cudaMemcpy(fc_2_output, fc_2_output_d, fc_2_num_neurons*sizeof(float), cudaMemcpyDeviceToHost);
        softmax(fc_2_output, fc_2_num_neurons);

        // Calculate the accuracy
        // int predicted_class = 0;
        // float max_prob = -std::numeric_limits<float>::infinity();
        // for(int j=0;j<fc_2_num_neurons;j++){
        //     if(fc_2_output[j] > max_prob){
        //         max_prob = fc_2_output[j];
        //         predicted_class = j;
        //     }
        // }
        // std::string actual_class = filenames[i].substr(filenames[i].size()-5,1);
        // if(predicted_class == std::stoi(actual_class)){
        //     counter++;
        // }
        // else{
        //     // std::cout << "Predicted class: " << predicted_class << " Actual class: " << actual_class << " Filename: " << filenames[i] << std::endl;
        // }

        std::string probabilityDirecory = "output";
        DIR* dir = opendir(probabilityDirecory.c_str());
        std::string probabilityFile = probabilityDirecory + "/" + filenames[i].substr(0,filenames[i].size()-4) + ".txt";
        std::ofstream file(probabilityFile);
        // write top 5 probabilities and their classes
        std::vector<std::pair<float,int>> prob;
        for(int j=0;j<fc_2_num_neurons;j++){
            prob.push_back(std::make_pair(fc_2_output[j],j));
        }
        std::sort(prob.begin(), prob.end(), std::greater<std::pair<float,int>>());
        for(int j=0;j<5;j++){
            file << prob[j].first*100 << " class " << prob[j].second << std::endl;
        }
        file.close();
        closedir(dir);

        // free the memory
        delete[] image_1d;
        delete[] image_bias;
        cudaFree(image_1d_d);
        cudaFree(conv_1_output_d);
        cudaFree(pool_1_output_d);
        cudaFree(conv_2_output_d);
        cudaFree(pool_2_output_d);
        cudaFree(fc_1_output_d);
        cudaFree(fc_2_output_d);
        delete[] fc_2_output;
    }

    // print the accuracy
    // std::cout << "Accuracy: " << (float)counter/filenames.size() << std::endl;

    // free the memory
    delete[] conv_1_weights;
    delete[] conv_1_bias;
    delete[] conv_2_weights;
    delete[] conv_2_bias;
    delete[] fc_1_weights;
    delete[] fc_1_bias;
    delete[] fc_2_weights;
    delete[] fc_2_bias;
    cudaFree(conv_1_weights_d);
    cudaFree(conv_2_weights_d);
    cudaFree(fc_1_weights_d);
    cudaFree(fc_2_weights_d);
    cudaFree(conv_1_bias_d);
    cudaFree(conv_2_bias_d);
    cudaFree(fc_1_bias_d);
    cudaFree(fc_2_bias_d);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
    std::cout << "Time taken: " << time_span.count() << " seconds" << std::endl;
}