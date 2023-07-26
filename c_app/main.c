#include <stdio.h>
#include <math.h>
#include "weights.h"
#include "flatten_img.h"

int main(){
    float output[50][1] = {0};
    float input[5][2] = {{1,2},{3,4},{5,6},{7,8},{9,10}};
    float weight[2][7] = {{1,2,3,4,5,6,7},{8,9,10,11,12,13,14}};

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


    for(i = 0; i < 50; i++){
        for(j = 0; j < 1; j++){
            for(k = 0; k < 50; k++){
                output[i][j] += fc2_weights[i][k] * output[k][j];
            }
        }
    }


    for(i = 0; i < 50; i++){
        for(j = 0; j < 1; j++){
            output[i][j] += fc2_bias[i][0];
        }
    }

    for(i = 0; i < 50; i++){
        for(j = 0; j < 1; j++){
            output[i][j] = output[i][j] > 0 ? output[i][j] : 0;
        }
    }

            printf("Output:\n");
    for(i = 0; i < 50; i++){
        for(j = 0; j < 1; j++){
            printf("%f\n", output[i][j]);
        }
    }

    /** FC 3 LAYER **/

    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            for(k = 0; k < 10; k++){
                output[i][j] += fc3_weights[i][k] * output[k][j];
            }
        }
    }

    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            output[i][j] += fc3_bias[i][0];
        }
    }


    // Do this in C output_fc3_log_softmax = output_fc3 - np.max(output_fc3)
    float max = output[0][0];
    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            if(output[i][j] > max){
                max = output[i][j];
            }
        }
    }

    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            output[i][j] -= max;
        }
    }

    // Do this in C output_fc3_log_softmax = output_fc3_log_softmax - np.log(np.sum(np.exp(output_fc3_log_softmax), axis=0))
    float sum = 0;
    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            sum += exp(output[i][j]);
        }
    }

    float log_sum = log(sum);
    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            output[i][j] -= log_sum;
        }
    }

    // Do this in C np.argmax(output_fc3_log_softmax)
    int max_index = 0;
    for(i = 0; i < 10; i++){
        for(j = 0; j < 1; j++){
            if(output[i][j] > output[max_index][j]){
                max_index = i;
            }
        }
    }


    
    printf("Max index: %d\n", max_index);


}