import os
import gc
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummary import summary


class ModelRun:
    def __init__(self, model, device=torch.device('cpu'), show_model=True):
        self.device = device
        self.model = model
        self.history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss'])
        if show_model:
            summary(self.model, (3, 32, 32), device=self.device.type)

        
    def _model_run(self, data_loader, stage, model_type, criterion, optimizer):
        '''
        Function of the model train/valid stage single epoch
        '''
        loss_cum = 0.
        # Vars for classification
        accuracy = 0
        total = 0
        accuracy_metric = None
        for inputs, labels in data_loader:
            
            if model_type == 'autoencoder':
                inputs = inputs.to(self.device)
                labels = inputs.to(self.device)
            elif model_type == 'classification':
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss_cum += loss.item()
                         
            if stage=='train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if (stage=='eval') and (model_type == 'classification'):
                y_pred_softmax = torch.log_softmax(self.model(inputs), dim=1)
                _, predicted = torch.max(y_pred_softmax.data, dim=1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
                accuracy_metric = (100 * accuracy / total)

            del inputs, labels
            gc.collect()
            torch.cuda.empty_cache()

        return loss_cum / len(data_loader), accuracy_metric
                             
    
    def model_train(self, train_loader, valid_loader, epochs=1, lr=0.001, classification=False, epoch_freq=5, file_suffix=''):
        '''
        Function of the model training run
        '''
        if classification:
            model_type = 'classification'
            criterion = torch.nn.CrossEntropyLoss()
        else: # autoencoder
            model_type = 'autoencoder'
            criterion = torch.nn.MSELoss()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        min_loss = float('inf')
        
        # Training process
        print(f'Start {model_type} model training...')
        for epoch in range(epochs):  
            for stage in ['train', 'eval']:
                # TRAIN
                if stage=='train':
                    self.model.train()
                    loss_train, accuracy = self._model_run(train_loader, stage, model_type, criterion, optimizer)
                # EVALUATION
                elif stage=='eval':
                    self.model.eval()
                    with torch.no_grad():
                        loss_valid, accuracy = self._model_run(valid_loader, stage, model_type, criterion, optimizer)

            self.history.loc[len(self.history)] = [epoch, loss_train, loss_valid]

            # Save the best model (relying on valid loss)
            if loss_valid < min_loss:
                min_loss = loss_valid
                self._save_results('Models', model_type, file_suffix)

            if (epoch % epoch_freq == 0):
                if accuracy is None:
                    add_accuracy = ''
                else:
                    add_accuracy = f', valid Accuracy: {accuracy}%'
                print(f"Epoch: {epoch}, train_loss: {loss_train}, valid_loss: {loss_valid}{add_accuracy}")

        self._save_results('History', model_type, file_suffix)
        print(f'Training of {model_type} model is finished!')
        self._plot_history()
    
    
    def _save_results(self, save_object, model_type, file_suffix):
        os.makedirs(f'./{save_object}', exist_ok=True)
        if save_object == 'Models':
            torch.save(self.model.state_dict(), f'./Models/{model_type}_model_hidden_dim{file_suffix}.pth')
        elif save_object == 'History':
            self.history.to_csv(f'./History/{model_type}_history_hidden_dim{file_suffix}.csv')

    def _plot_history(self):
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=self.history.melt(id_vars='epoch', 
                                            value_vars=['train_loss', 'valid_loss'], 
                                            var_name='loss_type',
                                            value_name='loss_value'),
                    x='epoch', 
                    y='loss_value',
                    hue='loss_type',
                    palette='Paired',
                    )
        plt.title('Training history')
        plt.show()
