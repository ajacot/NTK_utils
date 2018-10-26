# NTK_utils
The file network.py contains an alternative definition of the usual Linear and ConvNd layers using the parametrization described in the article https://arxiv.org/abs/1806.07572. This parametrization gives a consistent scaling behaviour as one increases the width of the network (the number of neurons in the hidden layers). As a result of the reparametrization, we suggest a learning rate around 1.0.

Furthermore network.py contains a module LinearNet which defines a fully-connected network given a list of the number of neurons in each layer. From this module, one can directly calculate the Neural Tangent Kernel, and the activation kernels as described in the article.
