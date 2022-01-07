import torch, time, sys
import numpy as np
import wandb

def test_model(model, dataloaders, dataset_sizes, criterion, optimizer, nsamples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    since = time.time()
    
    result_images, result_pred = list(), list()
    outputs_list = list()
    labels_list = list()
    predicted_out = np.empty(nsamples)
    
    for phase in ['val']:
        model.eval()

        running_loss = 0.0
        running_corrects = 0
        
        for i,(inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                preds_num, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(dataloaders[phase]), loss.item() * inputs.size(0)), end="")
            sys.stdout.flush()
            
            if i < nsamples:
                result_images.append(inputs)
                result_pred.append(preds[0].detach().cpu().numpy())
                labels_list.append(labels)
                outputs_list.append(outputs)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
    print()
    print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    print()
    
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
    return result_images, result_pred, labels_list, outputs_list
    
    
    
    