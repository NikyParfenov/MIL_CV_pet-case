# MIL CV pet-case

**CASE DESCRIPTION:**  
- Build and train autoencoder model on the one of datasets: CIFAR10, CIFAR100. Train classification-model relying on latent representation of trained autoencoder.
 
**RESULTS:**  
1. MIL_test_case.ipynb - Manual NN for autoencoder and classification model (Hid-dim: 32, Acc: ~28%).
2. MIL_test_case_VGG11.ipynb - VGG11-based autoencoder and classification model (Hid_dim: 16, Acc: ~46%).

There 2 types of networks (backbones) have been used: manual and VGG. Manual one was very simple and couldn't build precise autoencoder and classificator. VGG showed the better result and the further research should be held eather with VGG (vary of latent dimention) or with more complex network.
