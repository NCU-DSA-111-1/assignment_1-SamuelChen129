//
//  main.cpp
//  NeuralNetwork
//
//  Created by Santiago Becerra on 9/15/19.
//  Copyright © 2019 Santiago Becerra. All rights reserved.
//
//

#include <stdio.h>
#include <list>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define MAXinputbits 10
#define numInputs 2
#define numHiddenNodes 2 
#define numOutputs  1
#define numTrainingSets 4

// Simple network that can learn XOR
// Feartures : sigmoid activation function, stochastic gradient descent, and mean square error fuction

// Potential improvements :
// Different activation functions
// Batch training
// Different error funnctions
// Arbitrary number of hidden layers
// Read training end test data from a file
// Add visualization of training
// Add recurrence? (maybe that should be a separate project)

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); } //sigmoid's deriative is (sigmoid)*(1-sigmoid)
double init_weight() { return ((double)rand())/((double)RAND_MAX); }
void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

//trained model
void trainedXORmodel(double *hiddenbias,double **hiddenweights,
                     double *outputbias,double **outputweights)                     
{
    double *hidden=(double*)malloc(sizeof(double)*numInputs);

    char get_input1[MAXinputbits];
    char get_input2[MAXinputbits];
    
    char *input1_ptr=(char*)malloc(sizeof(char)*MAXinputbits);
    char *input2_ptr=(char*)malloc(sizeof(char)*MAXinputbits);

    double *output=(double*)malloc(sizeof(double)*MAXinputbits);
    int inputbits=0;
    //getting inputs from the user
    printf("input1:           ");
    scanf("%s",&get_input1);
    printf("input2:           ");
    scanf("%s",&get_input2);
    input1_ptr=&get_input1[0];
    input2_ptr=&get_input2[0];
    for(int i=0;i<MAXinputbits;i++)
    {
        if(input1_ptr[i]=='\0'){break;}       
        inputbits+=1;
    }
    double **input=(double**)malloc(sizeof(double*)*inputbits);
    for(int i=0;i<inputbits;i++)
    {
        input[i]=(double*)malloc(sizeof(double)*numInputs);
    }
    for(int i=0;i<inputbits;i++){
        if(input1_ptr[i]=='0'){input[i][0]=0.0f;}
        if(input1_ptr[i]=='1'){input[i][0]=1.0f;}
    }
    for(int i=0;i<inputbits;i++){
        if(input2_ptr[i]=='0'){input[i][1]=0.0f;}
        if(input2_ptr[i]=='1'){input[i][1]=1.0f;}
    }
    printf("input1 xor input2:");
    //put the inputs to the model
    for (int x=0; x<inputbits; x++) {
            // Forward propagation
            for (int j=0; j<numHiddenNodes; j++) {
                double activation=hiddenbias[j];
                 for (int k=0; k<numInputs; k++) {
                    activation+=input[x][k]*hiddenweights[k][j];
                }
                hidden[j] = sigmoid(activation);
            }
            for (int j=0; j<numOutputs; j++) {
                double activation=outputbias[j];
                for (int k=0; k<numHiddenNodes; k++) {
                    activation+=hidden[k]*outputweights[k][j];
                }
                output[j] = sigmoid(activation);
                if(output[j]>=0.5) {printf("1");}
                else{printf("0");}                  
            }
    }
    
    free(hidden);
    free(input1_ptr);
    free(input2_ptr);
    free(output);
    free(input);
    for(int i=0;i<inputbits;i++)
    {
        free(input[i]);
    }
}

int main(int argc, const char * argv[]) {
    //creat a file called loss.txt, which is used to save the loss value during traing.
    FILE *fptr;
    fptr = fopen("loss.txt","w");


    
    const double lr = 0.1f; //lr==learning rate
    
    double *hiddenLayer=(double*)malloc(sizeof(double)*numHiddenNodes);
    double *outputLayer=(double*)malloc(sizeof(double)*numOutputs); // actual output
    
    double *hiddenLayerBias=(double*)malloc(sizeof(double)*numHiddenNodes);
    double *outputLayerBias=(double*)malloc(sizeof(double)*numOutputs);

    double **hiddenWeights=(double**)malloc(sizeof(double*)*numInputs);
    for(int i=0;i<numInputs;i++)
    {
        hiddenWeights[i]=(double*)malloc(sizeof(double)*numHiddenNodes);
    }

    double **outputWeights=(double**)malloc(sizeof(double*)*numHiddenNodes);
    for(int i=0;i<numHiddenNodes;i++)
    {
        outputWeights[i]=(double*)malloc(sizeof(double)*numOutputs);
    }
    
    double **training_inputs=(double**)malloc(sizeof(double*)*numTrainingSets);
    for(int i=0;i<numTrainingSets;i++)
    {
        training_inputs[i]=(double*)malloc(sizeof(double)*numInputs);
    }

    training_inputs[0][0] = 0.0f;
	training_inputs[0][1] = 0.0f;
    training_inputs[1][0] = 0.0f;
    training_inputs[1][1] = 1.0f;
    training_inputs[2][0] = 1.0f;
    training_inputs[2][1] = 0.0f;
    training_inputs[3][0] = 1.0f;
    training_inputs[3][1] = 1.0f;

    double **training_outputs=(double**)malloc(sizeof(double*)*numTrainingSets);
    for(int i=0;i<numTrainingSets;i++)
    {
        training_outputs[i]=(double*)malloc(sizeof(double)*numOutputs);
    }
    training_outputs[0][0] = 0.0f;
	training_outputs[1][0] = 1.0f;
    training_outputs[2][0] = 1.0f;
    training_outputs[3][0] = 0.0f;

    
    //initializing weights and bias
    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weight();
        }
    }
    for (int i=0; i<numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j=0; j<numOutputs; j++) {
            outputWeights[i][j] = init_weight();
        }
    }
    for (int i=0; i<numOutputs; i++) {
        outputLayerBias[i] = init_weight();
    }
    
    int *trainingSetOrder=(int*)malloc(sizeof(int)*numTrainingSets);
    trainingSetOrder[0]=0;
    trainingSetOrder[1]=1;
    trainingSetOrder[2]=2;
    trainingSetOrder[3]=3;

    //training
    for (int n=0; n < 10000; n++) {
        shuffle(trainingSetOrder,numTrainingSets);
        for (int x=0; x<numTrainingSets; x++) {
            
            int i = trainingSetOrder[x];
            
            // Forward propagation
            
            for (int j=0; j<numHiddenNodes; j++) {
                double activation=hiddenLayerBias[j];
                 for (int k=0; k<numInputs; k++) {
                    activation+=training_inputs[i][k]*hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }
            
            for (int j=0; j<numOutputs; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<numHiddenNodes; k++) {
                    activation+=hiddenLayer[k]*outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }
            
            double loss=fabs(outputLayer[0]-training_outputs[i][0]);

            printf("Input:%f %f    Output: %f   Expected Output: %f  loss: %f\n",training_inputs[i][0],training_inputs[i][1],outputLayer[0],training_outputs[i][0],loss);
            //saving loss values to the file
            fprintf(fptr,"n=%d loss=%f\n",n,loss);
           // Back-propagation
            
            double *deltaOutput=(double*)malloc(sizeof(double)*numOutputs); 
            for (int j=0; j<numOutputs; j++) {
                double errorOutput = (training_outputs[i][j]-outputLayer[j]);
                deltaOutput[j] = errorOutput*dSigmoid(outputLayer[j]);
            }
            
            double *deltaHidden=(double*)malloc(sizeof(double)*numHiddenNodes);
            for (int j=0; j<numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<numOutputs; k++) {
                    errorHidden+=deltaOutput[k]*outputWeights[j][k];
                }
                deltaHidden[j] = errorHidden*dSigmoid(hiddenLayer[j]);
            }
            
            for (int j=0; j<numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j]*lr;
                for (int k=0; k<numHiddenNodes; k++) {
                    outputWeights[k][j]+=hiddenLayer[k]*deltaOutput[j]*lr;
                }
            }
            
            for (int j=0; j<numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j]*lr;
                for(int k=0; k<numInputs; k++) {
                    hiddenWeights[k][j]+=training_inputs[i][k]*deltaHidden[j]*lr;
                }
            }
            free(deltaOutput);
            free(deltaHidden); 
        }
    }
    
    // Print weights and bias after training
    printf("Final Hidden Weights\n[ ");
    for (int j=0; j<numHiddenNodes; j++) {
        printf( "[ ");
        for(int k=0; k<numInputs; k++) {
            printf("%f ",hiddenWeights[k][j]);
        }
        printf("] ");
    }
    printf("]\n");
    
    printf("Final Hidden Biases\n[ ");
    for (int j=0; j<numHiddenNodes; j++) {
        printf("%f ",hiddenLayerBias[j]);

    }
    printf("]\n");
    printf("Final Output Weights");
    for (int j=0; j<numOutputs; j++) {
        printf("[ ");
        for (int k=0; k<numHiddenNodes; k++) {
            printf("%f ",outputWeights[k][j]);
        }
        printf("]\n");
    }
    printf("Final Output Biases\n[ ");
    for (int j=0; j<numOutputs; j++) {
        printf("%f ",outputLayerBias[j]);
        
    }
    printf("]\n");

    //put the final weights and bias to the trained model
    trainedXORmodel(hiddenLayerBias, hiddenWeights,
                    outputLayerBias, outputWeights);
    
    fclose(fptr);

    free(hiddenLayer);
    free(outputLayer);     
    free(hiddenLayerBias);
    free(outputLayerBias);
    free(hiddenWeights);
    for(int i=0;i<numInputs;i++)
    {
        free(hiddenWeights[i]);
    }
    free(outputWeights);
    for(int i=0;i<numHiddenNodes;i++)
    {
        free(outputWeights[i]);
    }
    free(training_inputs);
    for(int i=0;i<numTrainingSets;i++)
    {
        free(training_inputs[i]);
    }
    free(training_outputs);
    for(int i=0;i<numTrainingSets;i++)
    {
        free(training_outputs[i]);
    }
    free(trainingSetOrder);


    return 0;
}

