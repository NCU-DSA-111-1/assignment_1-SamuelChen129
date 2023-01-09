# Implementing XOR Gate by Simple Neural Network 
## After reading this README,you will know:
* how to compile the source code.
* how to run the execution file of the source code.
* the structure of the source code.
## how to compile
1. Open the terminal.
2. Open the folder to the source code.
   The file name of the source code is "108501569_assignment_1_code.cpp".
3. Enter the linux instrction below:
```
gcc -o 108501569_assignment_1_code 108501569_assignment_1_code.cpp
```
## how to run
1. After compiling ,enter the linux instrction below:
```
./108501569_assignment_1_code
```
2. Wait until the final weights and bias produced.
3. Enter an n-bits binary number as input1. (n<=10)
4. Enter an n-bits binary number as input2. 
(The length of input2 should be same as the length of input1)
5. Check that whether the "input1 xor input2" is correct. Also,<font color=orange>"loss.txt"</font> is produced at tha same time.
6. Run the <font color=orange>"loss.py"</font> to show the loss convergence:
```
python loss.py
```

## Structure
There are two main part of this program:
* Neural Network for training
* Trained model
### Neural Network 
The neural network is using _Forward propagation_ and _Back-propagation_ techniques.  
First,the <font color=orange>forward propagation</font> means propagating the training input data through the network like the picture below. 
![image](https://github.com/NCU-DSA-111-1/assignment_1-SamuelChen129/blob/main/nn.PNG)
There are a hidden layer and a output layer in this network.   
The hidden layer has two nodes,the output layer has one node.This means that there are two weight values and two bias values in the hidden layer and one weight value and one bias value in the output layer.   
The output value of the network is calculated by the values below,
* two input values
* two weight values in the hidden layer
* two bias values in the hidden layer
* one weight value in the output layer
* one bias value in the output layer

and these values are initialized with random floating numbers between 0 and 1,which means that the output could hardly be desired at first.
Here is the code of forward propagation:
```sh
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
```
Next,the <font color=orange>Back-propagation</font> is to adjust the weights and bias according to the loss function.  
Just like the picture below,   
![GITHUB](nn2.png)
the loss function will send out the loss,which is the difference between the actual output (from the forward propagation) and the desired output.  
We can use the change in the losses from the output and the last output,which is the so-called derivative of loss.  
By the derivative of loss,we can know we should increase or decrease how much in the weights and bias values.
Here is the code of back-propagation:  
```sh
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
```
After many times of the forward propagation and the back-propagation,we can get the weights and bias which can make the output closed to desired value.
And this procession is called "training".
### Tranied Model
The trained model is the forward propagation that uses the weights and bias values,which produced after training, to calculate the output.
And this output should be the result of the two inputs operated with the xor gate. 
The two input values used by the propagation are got from the user.
Here is the code of trained model:
```sh
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
```
## Reference
[Simple neural network implementation in C](https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547)


