import os
import gc
import numpy as np
import pickle as pkl
import torch
from model_file import device, model, loss_fn, optimizer, data_folder

num_epochs = 20000 #####################
batch_size = 9000 ###


def save_checkpoint(model_name, outdir, state, t):
    print('save_checkpoint')
    outname = 'checkpoint_'+str(t)+'.pt'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fpath = os.path.join(outdir, outname) 
    torch.save(state, fpath)
    return model_name, outdir, outname, str(t)


if __name__ == "__main__":
    data_split = None
    with open(os.path.join(data_folder, 'data_split.pkl'), 'rb') as fileObject2:
        data_split = pkl.load(fileObject2)
    data_train, data_valid, _ = data_split

    model_name = str(model.__class__.__name__)
    outdir = './' + model_name + '_checkpoints'

    train_losses, valid_losses = [], []
    historical_min_loss, checkpoint, best_epoch = None, None, None

    train_dataset_X, train_dataset_y = data_train
    validation_dataset_X, validation_dataset_y = data_valid

    for epoch in range(num_epochs):
        condition = epoch<3 or (epoch+1)%10==0
        
        if condition:
            print(str(epoch+1), end=' ')
            
        ## Training phase
        running_loss = .0
        model.train()

        for b in range(0, len(train_dataset_X), batch_size):
            inpt = train_dataset_X[b:b+batch_size, :, :]
            target = train_dataset_y[b:b+batch_size]    
            
            X_batch = torch.tensor(inpt, dtype=torch.float32).to(device=device)
            y_batch = torch.tensor(target, dtype=torch.float32).to(device=device)
            
            model.init_hidden(X_batch.size(0))
        #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
        #    lstm_out.contiguous().view(x_batch.size(0),-1)
            
            y_batch_pred = model(X_batch)
            loss = loss_fn(y_batch_pred.view(-1), y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss

        train_loss = running_loss/len(train_dataset_X)
        train_losses.append(train_loss.data.cpu().numpy())

        if condition:
            print(f'train_loss {train_loss}', end=' ')
            
        ## Validation phase ## 
        running_loss = .0
        model.eval()

        with torch.no_grad():
            
            for b in range(0, len(validation_dataset_X), batch_size):
                inpt = validation_dataset_X[b:b+batch_size, :, :]
                target = validation_dataset_y[b:b+batch_size]    

                X_batch = torch.tensor(inpt, dtype=torch.float32).to(device=device)
                y_batch = torch.tensor(target, dtype=torch.float32).to(device=device)

                model.init_hidden(X_batch.size(0))
            #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
            #    lstm_out.contiguous().view(x_batch.size(0),-1)

                y_batch_pred = model(X_batch)
                loss = loss_fn(y_batch_pred.view(-1), y_batch)
                
                running_loss += loss

            valid_loss = running_loss/len(validation_dataset_X)
            valid_losses.append(valid_loss.data.cpu().numpy())

            if condition:
                print(f'validation_loss {valid_loss}')
                
            if not historical_min_loss or valid_loss < historical_min_loss:
                historical_min_loss = valid_loss
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                best_epoch = epoch
        gc.collect()


    _, outdir, outname, best_epoch = save_checkpoint(model_name, outdir, checkpoint, best_epoch)
    print('Training completed!', 'best_epoch=', best_epoch)