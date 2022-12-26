import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(dataset, model):
    x = torch.Tensor(dataset.data.transpose(0, 3, 1, 2))
    y = torch.Tensor(dataset.targets)

    _, predicted = torch.max(model(x).data, 1)
    cm = confusion_matrix(y, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot( values_format ='.3g')
    plt.show()

def plot_decoded_pics(dataset, model, chosen_dim=''):
    figure = plt.figure(figsize=(18, 4))
    cols, rows = 8, 2
    for i in range(1, cols + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        prediction = model(img[None, :, :, :]).detach().numpy()[0]
        prediction[prediction > 1.] = 1.
        
        figure.add_subplot(rows, cols, i)
        plt.title(dataset.classes[label])
        plt.xticks([]), plt.yticks([])
        plt.imshow(np.transpose(img, axes=(1, 2, 0)))

        figure.add_subplot(rows, cols, i+8)
        plt.title(dataset.classes[label])
        plt.xticks([]), plt.yticks([])
        plt.imshow(np.transpose(prediction, axes=(1, 2, 0)))
    
    plt.suptitle(f'Hidden dimension size {chosen_dim}', color='yellow')
    plt.show()
