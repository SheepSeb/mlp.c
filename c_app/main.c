#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "weights.h"
#include "flatten_img.h"

int main(){
    float output[50][1] = {0};

    int i,j,k;

    /** FC 1 LAYER **/

    for(i = 0; i < 50; i++){
        for(j = 0; j < 1; j++){
            for(k = 0; k < 28*28; k++){
                output[i][j] += fc1_weights[i][k] * flatten_img[k][j];
            }
        }
    }

    for(i = 0; i < 50; i++){
        for(j = 0; j < 1; j++){
            output[i][j] += fc1_bias[i][0];
        }
    }


    for(i = 0; i < 50; i++){
        for(j = 0; j < 1; j++){
            output[i][j] = output[i][j] > 0 ? output[i][j] : 0;
        }
    }



    /** FC 2 LAYER **/

    float output_fc2[50][1] = {0};

    for(i = 0; i < 50; i++){
        for(j = 0; j < 1; j++){
            for(k = 0; k < 50; k++){
                output_fc2[i][j] += fc2_weights[i][k] * output[k][j];
            }
        }
    }

    for(i = 0; i < 50; i++){
        for(j = 0; j < 1; j++){
            output_fc2[i][j] += fc2_bias[i][0];
        }
    }

    for(i = 0; i < 50; i++){
        for(j = 0; j < 1; j++){
            output_fc2[i][j] = output_fc2[i][j] > 0 ? output_fc2[i][j] : 0;
        }
    }


    float output_fc3[10][1] = {0};

    /** FC 3 LAYER **/

    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            for(k = 0; k < 50; k++){
                output_fc3[i][j] += fc3_weights[i][k] * output_fc2[k][j];
            }
        }
    }

    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            output_fc3[i][j] += fc3_bias[i][0];
        }
    }


    // Do this in C output_fc3_log_softmax = output_fc3 - np.max(output_fc3)
    float max = output_fc3[0][0];
    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            if(output_fc3[i][j] > max){
                max = output_fc3[i][j];
            }
        }
    }

    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            output_fc3[i][j] -= max;
        }
    }

    // Do this in C output_fc3_log_softmax = output_fc3_log_softmax - np.log(np.sum(np.exp(output_fc3_log_softmax), axis=0))
    float sum = 0;
    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            sum += exp(output_fc3[i][j]);
        }
    }

    float log_sum = log(sum);
    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            output_fc3[i][j] -= log_sum;
        }
    }

    // printf("Output FC3:\n");
    // for(i = 0; i < 10; i++){
    //     for(j = 0; j < 1; j++){
    //         printf("%f ", output_fc3[i][j]);
    //     }
    //     printf("\n");
    // }

    // Do this in C np.argmax(output_fc3_log_softmax)
    int max_index = 0;
    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            if(output_fc3[i][j] > output_fc3[max_index][j]){
                max_index = i;
            }
        }
    }
    
    printf("[Max index] Digit Predicted: %d\n", max_index);


}