#include <float.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <dirent.h>
#include <chrono>
#include <string>
// #include <opencv2/opencv.hpp>


__global__ void relu(float* input, float* output, int N, int M){
    int i = threadIdx.y;
    int j = threadIdx.x;
    output[i*N + j] = fmaxf(0.0, input[i*N + j]);
}


__global__ void tanh(float* input, float* output, int N, int M){
    int i = threadIdx.y;
    int j = threadIdx.x;
    output[i*N + j] = tanh(input[i*N + j]);
}



__global__ void maxPooling(float* input, float* output, int N, int M, int output_size){
    int i = threadIdx.y;
    int j = threadIdx.x;
    float max = FLT_MIN;
    for (int k = 0; k < M; k++){
        for (int l = 0; l < M; l++){
            int ind = (i+k)*N + j+l;
            max = fmaxf(max, input[ind]);
        }
    }
    output[i*output_size + j] = max;
}


__global__ void avgPooling(float* input, float* output, int N, int M, int output_size){
    int i = threadIdx.y;
    int j = threadIdx.x;
    float sum = 0.0;
    for (int k = 0; k < M; k++){
        for (int l = 0; l < M; l++){
            sum += input[(i+k)*N + j+l];
        }
    }
    output[i*output_size + j] = (float)(sum/(M*M));
}

__global__ void convolution(float* input, float* weights, float* output, int N, int M, int P){
    int output_size = N+2*P-M+1;
    // only one block
    int i = threadIdx.y;
    int j = threadIdx.x;
    float sum = 0.0f;
    for(int k = 0; k < M; k++){
        for (int l = 0; l < M; l++){
            sum += input[(i+k)*(N+2*P) + j+l] * weights[k*M + l];
        }
    }
    output[i*output_size + j] = sum;

}




__global__ void sigmoid(float* input, float* output, int N){
    int i = threadIdx.x;
    output[i] = 1/(1+exp(-input[i]));
}

void softmax(float* input, float* output, int N){
    float sum = 0.0;
    for (int i = 0; i < N; i++){
        output[i] = exp(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < N; i++){
        output[i] = output[i]/sum;
    }
}


int main(int argc, char *argv[]){
    if (std::stoi(argv[1]) == 1){ 
        //convolution

        int N = std::stoi(argv[2]);
        int M = std::stoi(argv[3]);
        int P = std::stoi(argv[4]);
        // N, M matrices are float matrices
        float* input = new float[(N+2*P)*(N+2*P)];
        float* weights = new float[M*M];
        float* output = new float[(N+2*P-M+1)*(N+2*P-M+1)];


        int argvIndex = 5;

        for (int i = 0; i < (N+2*P)*(N+2*P); i++){
            if (i < P*(N+2*P)){
                input[i] = 0.0;
            }
            else if (i >= (N+P)*(N+2*P)){
                input[i] = 0.0;
            }
            else if (i % (N+2*P) < P){
                input[i] = 0.0;
            }
            else if (i % (N+2*P) >= N+P){
                input[i] = 0.0;
            }
            else{
                input[i] = std::stof(argv[argvIndex]);
                argvIndex++;
            }
        }

        for(int i = 0; i < M*M; i++){
            weights[i] = std::stof(argv[argvIndex]);
            argvIndex++;
        }

        float* input_d;
        float* weights_d;
        float* output_d;
        cudaMalloc(&input_d, (N+2*P)*(N+2*P)*sizeof(float));
        cudaMalloc(&weights_d, M*M*sizeof(float));
        cudaMalloc(&output_d, (N+2*P-M+1)*(N+2*P-M+1)*sizeof(float));
        cudaMemcpy(input_d, input, (N+2*P)*(N+2*P)*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(weights_d, weights, M*M*sizeof(float), cudaMemcpyHostToDevice);
        convolution<<<1, dim3(N+2*P-M+1, N+2*P-M+1)>>>(input_d, weights_d, output_d, N, M, P);

        cudaMemcpy(output, output_d, (N+2*P-M+1)*(N+2*P-M+1)*sizeof(float), cudaMemcpyDeviceToHost);

        //print as matrix
        for (int i = 0; i < N+2*P-M+1; i++){
            for (int j = 0; j < N+2*P-M+1; j++){
                std::cout << output[i*(N+2*P-M+1) + j] << " ";
            }
            std::cout << std::endl;
        }
        cudaFree(input_d);
        cudaFree(weights_d);
        cudaFree(output_d);
        delete[] input;
        delete[] output;
        delete[] weights;
    }

    if(std::stoi(argv[1]) == 2){
        //activation 0 = relu, 1 = tanh
        int activation = std::stoi(argv[2]);
        int N = std::stoi(argv[3]);
        int M = std::stoi(argv[4]);
        float* input = new float[N*M];
        float* output = new float[N*M];
        for (int i = 0; i < N*M; i++){
            input[i] = std::stof(argv[i+5]);
        }

        float* input_d;
        float* output_d;
        cudaMalloc(&input_d, N*M*sizeof(float));
        cudaMalloc(&output_d, N*M*sizeof(float));

        cudaMemcpy(input_d, input, N*M*sizeof(float), cudaMemcpyHostToDevice);


        if (activation == 0){
            relu<<<1, dim3(N,M)>>>(input_d, output_d, N, M);
        }
        else if (activation == 1){
            tanh<<<1, dim3(N,M)>>>(input_d, output_d, N, M);
        }

        cudaMemcpy(output, output_d, N*M*sizeof(float), cudaMemcpyDeviceToHost);
        //print as matrix
        for (int i = 0; i < N; i++){
            for (int j = 0; j < M; j++){
                std::cout << output[i*M + j] << " ";
            }
            std::cout << std::endl;
        }
        cudaFree(input_d);
        cudaFree(output_d);

        delete[] input;
        delete[] output;
    }

    if(std::stoi(argv[1]) == 3){
        int pool_func = std::stoi(argv[2]);
        int M = std::stoi(argv[3]);
        int N = std::stoi(argv[4]);
        float* input = new float[N*N];
        for(int i = 0; i < N*N; i++){
            input[i] = std::stof(argv[i+5]);
        }
        // ceil of N/M
        int output_size = N-M+1;
        float* output = new float[output_size*output_size];

        float* input_d;
        float* output_d;
        cudaMalloc(&input_d, N*N*sizeof(float));
        cudaMalloc(&output_d, output_size*output_size*sizeof(float));
        cudaMemcpy(input_d, input, N*N*sizeof(float), cudaMemcpyHostToDevice);
        if (pool_func == 0){
            maxPooling<<<1, dim3(output_size, output_size)>>>(input_d, output_d, N, M, output_size);
        }
        else if (pool_func == 1){
            avgPooling<<<1, dim3(output_size, output_size)>>>(input_d, output_d, N, M, output_size);
        }
        cudaMemcpy(output, output_d, output_size*output_size*sizeof(float), cudaMemcpyDeviceToHost);
        //print the output matrix
        for (int i = 0; i < output_size; i++){
            for (int j = 0; j < output_size; j++){
                std::cout << output[i*output_size + j] << " ";
            }
            std::cout << std::endl;
        }
        cudaFree(input_d);
        cudaFree(output_d);

        delete[] input;
        delete[] output;
        
    }
    if(std::stoi(argv[1]) == 4){
        float* input = new float[argc - 3];
        int function = std::stoi(argv[2]);
        for (int i = 0; i < argc - 3; i++){
            input[i] = std::stof(argv[i+3]);
        }

        int N = argc - 3;
        float* output = new float[N];
        // 0 = sigmoid, 1 = softmax
        if (function == 0){
            float* input_d;
            float* output_d;
            cudaMalloc(&input_d, N*sizeof(float));
            cudaMalloc(&output_d, N*sizeof(float));
            cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
            sigmoid<<<1, N>>>(input_d, output_d, N);
            cudaMemcpy(output, output_d, N*sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(input_d);
            cudaFree(output_d);

        }
        else if (function == 1){
            softmax(input,output, N);
        }
        for (int i = 0; i < N; i++){
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;
        delete[] input;
        delete[] output;
    }
}