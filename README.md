# MIL_test_case
*Computer Vision test case of MIL*

**ЗАДАНИЕ:**  
- Написать и обучить модель-автокодировщик на датасете на выбор: CIFAR10, CIFAR100. Обучить модель-классификатор на латентных представлениях обученного автокодировщика.
 
**РЕШЕНИЕ:**  
1. MIL_test_case.ipynb - Manual NN for autoencoder and classification model (Hid-dim: 32, Acc: ~28%).
2. MIL_test_case_VGG11.ipynb - VGG11-based autoencoder and classification model (Hid_dim: 8, Acc: ~44%).
3. MIL_test_case_VGG16.ipynb - VGG16-based autoencoder and classification model (Hid_dim: 8, Acc: ~40%).

The increse of feature-model size from VGG11 to VGG16 doesn't improve the model quiality. So, to increase the classification quiality hidden-dimension and classification block should be researched more carefully.

