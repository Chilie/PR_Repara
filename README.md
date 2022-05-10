# Code for Deep Learning for Phase Retrieval
## Authors, institution, location
Below is the information for the authors.
 + *Author*       Ji Li
 + *Institution*  Department of Mathematics, National University of Singapore
 + *Location*    21 Lower Kent Ridge Rd, Singapore 119077  
 -------
## Main idea
The solution is built on the fully connected network to represent the unknown signal. Then the optimization is to learning the network and the nonconvex optimization is with better convergence behaviors. Such global convergence property for learning deep networks has been investigated in the deep learning community. 

## Existing issues
The solution works well for recovering a non-negative signal. It is found that our algorithm framework failed to recover a general signal with negative elements.

It is our future work to make our solution work well for recovering a general signal.