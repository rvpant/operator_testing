## Some Things to Pay Attention to

below are 3 different variants of KAN I borrowed from 
https://github.com/SynodicMonth/ChebyKAN
https://github.com/SpaceLearner/JacobiKAN
https://github.com/Boris-73-TA/OrthogPolyKANs

I slightly modified them to make them a bit more easy to use, feel free to make additional changes to it.
Since all codes are designed for tasks such as MNIST, a few things you may want to include in your testing:
* Activation functions being used.
* degree of polynomial
* whether layerNorm is needed for our tasks
*  more....

I would suggest writing an iterative script for testing different hyperparameters so that you can have it running in the background and no need to manually change the setting every few minutes. 

I only did minimal tests on these models so in case they dont work let me know.
