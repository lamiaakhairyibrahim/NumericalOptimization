# Numerical Optimization
 - [Overview](#overview)
 - [Topics](#topics)


 ## Overview
  - For this practical work, we will have to develop a Python program that is able to implement the type of gradient descent variants in order to achieve the linear regression of a set of datasets.

## Topics
 - ### 1. Gradient Descent
     - ### *Step1*:
      Initialize parameters (theta_0 & theta_1) with random value or simply zero ,Also choose the Learning rate.

     - ### *Step2*:
      Use (theta_0 & theta_1) to predict the output h(x)= theta_0 + theta_1 * x.

     - ### *Step3*:
      Calculate Cost function 𝑱(theta_0,theta_1 ).

     - ### *Step4*:
      Calculate the gradient.

     - ### *Step5*:
      Update the parameters (simultaneously).

     - ### *Step6*:
      Repeat from 2 to 5 until converge to the minimum or achieve maximum iterations.
 - ### 2. Stochastic Gradient Descent
        Stochastic Gradient Descent (SGD) is an optimization algorithm used to minimize the cost function in machine learning and deep learning models. Unlike batch gradient descent, which processes the entire dataset at once, SGD updates the model parameters using one training example at a time, making it faster and more efficient, especially for large datasets.

    🔄 Steps of Stochastic Gradient Descent:
    - 1. ### Initialize Parameters

        Start by initializing the model parameters (weights theta) to zeros or small random values.

    - 2. ### Add Bias Term

        Add a column of ones to the input data to account for the bias (intercept) term.

    - 3. ### Repeat for a Number of Epochs

        Loop through the dataset multiple times (each loop is called an "epoch").

    - 4. ### For Each Sample in the Dataset:

        - Select one training example (xi, yi).

        -  Predict the output using current parameters:
        ```bach
             prediction = xi @ theta
        ```

        -  Compute the error:
        ```bach
            - error = prediction - yi
        ```

        -  Calculate the gradient of the cost function w.r.t. the parameters:
        ```bach
            - grad = xi.T @ error
        ```

        - Update the parameters using the learning rate lr:
        ```bach
            - theta = theta - lr * grad
        ```

    - 5. ### Track Loss (Optional)

        Compute and store the loss after each epoch to monitor the training process.

    - 6. ### Return the Final Parameters

        After all epochs, return the optimized parameters and the loss history.