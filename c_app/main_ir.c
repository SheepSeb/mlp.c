#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../c_headers/weights_2_copy.h"
#include "flatten_2_img_copy.h"


float* allocate_matrix(int rows, int cols){
    float *matrix;

    matrix = (float *)calloc(rows * cols, sizeof(float *));
    return matrix;
}

void free_matrix(float *matrix){
    free(matrix);
}


void print_matrix(float *matrix, int r, int c){
    printf("Printing matrix\n");
    for(int i = 0; i < r; i++){
        for(int j = 0; j < c; j++){
            printf("%f , ", matrix[i * c + j]);
        }
        printf("\n");
    }
}


void matmul(float *input, float *output, float *weights, int r, int c, int s){
    int i,j,k;
    //c = 1
    //output = allocate_matrix(c, s);
    //input = allocate_matrix(r, c);
    //weights = allocate_matrix(s, r);
    for(i = 0; i < r; i++){
        for(j = 0; j < c; j++){
            for(k = 0; k < s; k++){
                output[j * s + k] += input[i * c + j] * weights[k * r + i];
            }
        }
    }
}

void add_bias(float *matrix, float *bias, int r, int c){
    int i,j,k;

    for(i = 0; i < r; i++){
        for(j = 0; j < c; j++){
            matrix[i*c+j] += bias[i];
        }
    }
}

void relu(float *matrix, int r){
    for(int i = 0; i< r; i++){
        matrix[i]= matrix[i]> 0 ? matrix[i] : 0;
    }
}

int softmax(float *matrix, int r){
    int i,j;
        // Do this in C output_fc3_log_softmax = output_fc3 - np.max(output_fc3)
    float max = matrix[0];
    for(i = 0; i < r; i++){
        for(j = 0; j < 1; j++){
            // printf("soft %f\n",matrix[i * 1 + j]);
            if(matrix[i * 1 + j] > max){
                max = matrix[i * 1 + j];
            }
        }
    }

    for(i = 0; i < r; i++){
        for(j = 0; j < 1; j++){
            matrix[i * 1 + j] -= max;
        }
    }

    // Do this in C output_fc3_log_softmax = output_fc3_log_softmax - np.log(np.sum(np.exp(output_fc3_log_softmax), axis=0))
    float sum = 0;
    for(i = 0; i < r; i++){
        for(j = 0; j < 1; j++){
            sum += exp(matrix[i * 1 + j]);
        }
    }

    float log_sum = log(sum);
    for(i = 0; i < r; i++){
        for(j = 0; j < 1; j++){
            matrix[i * 1 + j] -= log_sum;
        }
    }

    // Do this in C np.argmax(output_fc3_log_softmax)
    int max_index = 0;
    for(i = 0; i < r; i++){
        for(j = 0; j < 1; j++){
            if(matrix[i * 1 + j] > matrix[max_index * 1 + j]){
                max_index = i;
            }
        }
    }

    return max_index;
}


void forward(float *input, float *output, struct layer layers[NUM_LAYERS]){
    //printf("LAYER SIZE %f\n",input[14]);

    float *prec = allocate_matrix(layers[0].size_input, 1);
    memcpy(prec, input, layers[0].size_input * sizeof(float));

    for(int i = 0; i< NUM_LAYERS - 1; i++){
        float *aux = allocate_matrix(layers[i].size_output, 1);
       // print_matrix(prec, layers[i].size_input, 1);
        matmul(prec, aux, layers[i].weights, layers[i].size_input, 1, layers[i].size_output);

        free_matrix(prec);
        prec = aux;

        add_bias(prec,layers[i].bias,layers[i].size_output, 1); 
        relu(prec,layers[i].size_output);
        print_matrix(prec, layers[i].size_output, 1);
    }

    // last layer
    float *aux = allocate_matrix(layers[NUM_LAYERS-1].size_output, 1);
    matmul(prec,aux,layers[NUM_LAYERS-1].weights, layers[NUM_LAYERS-1].size_input, 1 ,layers[NUM_LAYERS-1].size_output);
   //printf("Out %f\n",aux[0]);
    free_matrix(prec);
    add_bias(aux,layers[NUM_LAYERS-1].bias,layers[NUM_LAYERS-1].size_output, 1);

   /// print_matrix(aux, layers[NUM_LAYERS-1].size_output, 1);
    memcpy(output, aux, layers[NUM_LAYERS-1].size_output * sizeof(float));
    free_matrix(aux);
}

void create_layer(struct layer *layer, int size_input, int size_output){
    layer->size_input = size_input;
    layer->size_output = size_output;
    layer->weights = allocate_matrix(size_output, size_input);
    layer->bias = allocate_matrix(size_output, 1);
}

void assign_weights_and_bias(struct layer *layer, const float *weights, const float *bias){
    memcpy(layer->weights, weights, layer->size_output * layer->size_input * sizeof(float));
    memcpy(layer->bias, bias, layer->size_output * sizeof(float));
}

void create_layers(struct layer *layers){
    create_layer(&layers[0], SIZE_L0, SIZE_L1);
    create_layer(&layers[1], SIZE_L1, SIZE_L2);
    create_layer(&layers[2], SIZE_L2, SIZE_L3);
}

void assign_layers(struct layer *layers){
    assign_weights_and_bias(&layers[0], fc1_weights, fc1_bias);
    assign_weights_and_bias(&layers[1], fc2_weights, fc2_bias);
    assign_weights_and_bias(&layers[2], fc3_weights, fc3_bias);
}

void free_layer(struct layer *layer){
    free_matrix(layer->weights);
    free_matrix(layer->bias);
}

void print_layers(struct layer *layers){
    for(int i = 0; i < NUM_LAYERS; i++){
        printf("Layer %d\n", i);
        printf("Size Input: %d\n", layers[i].size_input);
        printf("Size Output: %d\n", layers[i].size_output);
        printf("Weights: \n");
        for(int j = 0; j < layers[i].size_output; j++){
            for(int k = 0; k < layers[i].size_input; k++){
                printf("%f ", layers[i].weights[j * layers[i].size_input + k]);
            }
            printf("\n");
        }
        printf("Bias: \n");
        for(int j = 0; j < layers[i].size_output; j++){
            printf("%f ", layers[i].bias[j]);
        }
        printf("\n");
    }
}


int main(){

    // Create the layers
    struct layer layers[NUM_LAYERS];
    create_layers(layers);
    assign_layers(layers);
   // print_layers(layers);

    float *f_img = allocate_matrix(SIZE_L0, 1);
    memcpy(f_img, flatten_img, SIZE_L0 * sizeof(float));
   
    float *output = allocate_matrix(SIZE_L3, 1);
    forward(f_img, output, layers);
   // print_matrix(output, SIZE_L3, 1);
    int index = softmax(output, layers[NUM_LAYERS-1].size_output);

    printf("[Max index] Digit Predicted: %d\n", index);
    free_matrix(output);
}