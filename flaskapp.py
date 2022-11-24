import os
import numpy as np
import pickle as pkl
import torch
from numpy import array
import random
from model_file import device, model, loss_fn, optimizer, data_folder
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request

app = Flask(__name__)

#just setting to some working values to start for the app so it has data on boot first 2 predictions will be wrong
lux1 = 100
ph1 = 7.73
rd1 = 4.1
lux2 = 0
ph2 = 7.73
rd2 = 4.1

def load_checkpoint(fpath, model, optimizer):
    print('load checkpoint...')
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def make_prediction():

    print("-----------  evaluating... ------------")
    lux1 = globals()['lux1'] 
    lux2 = globals()['lux2']
    ph1 = globals()['ph1'] 
    ph2 = globals()['ph2']
    rd1 = globals()['rd1'] 
    rd2 = globals()['rd2']
    
    print(str(lux1) + " , " + str(ph1) + " , " + str(rd1) )
    print(str(lux2) + " , " + str(ph2) + " , " + str(rd2) )
    #max and min data to scale to 0-100 for light and max and min values from training dataset for PH
    scalerdata = [[8.514179, 0], [7.721082, 100]]
    scaler = MinMaxScaler()
    scaler.fit(scalerdata)
    #print(scaler.data_max_)
    
    dv_x = [[ph1, lux1],[ph2, lux2]]
    dv_x = scaler.transform(dv_x)
    #print(str(dv_x))
    
    dv_x = [[dv_x[0][0],dv_x[0][1],rd1],[dv_x[1][0],dv_x[1][1],rd2]]  #append RD onto input data arrary couldnt get concat to work lol noob
    
    #print(str(dv_x))

    data_split = None
    with open(os.path.join(data_folder, 'data_split.pkl'), 'rb') as fileObject2:
        data_split = pkl.load(fileObject2)
    _, _, data_test = data_split

    model = globals()['model']
    optimizer = globals()['optimizer']
    loss_fn = globals()['loss_fn']
    model_name = str(model.__class__.__name__)
    outdir = './' + model_name + '_checkpoints'
    outname = 'checkpoint_19998.pt'
    file_path = os.path.join(outdir, outname)
    model, optimizer = load_checkpoint(file_path, model, optimizer)
        
    model.eval()
        
    with torch.no_grad():
        

        # print(dv_y.shape, type(dv_y))
        dv_x = np.expand_dims(dv_x, axis=0)
        # dv_y = np.expand_dims(dv_y, axis=0)
        # print(dv.shape, type(dv))
        #print('Input:', dv_x)
        
        X_test = torch.tensor(dv_x, dtype=torch.float).to(device=device)
        # Y_test = torch.tensor(dv_y, dtype=torch.float).to(device=device)

        model.init_hidden(X_test.size(0))
        # lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
        # lstm_out.contiguous().view(x_batch.size(0),-1)

        y_pred = model(X_test)
        print('Predicted output:', y_pred.reshape(y_pred.shape[0]).data.cpu().numpy().tolist())
        # print('Actual output:', Y_test.data.cpu().numpy().tolist())
        est_rd = y_pred.reshape(y_pred.shape[0]).data.cpu().numpy().tolist()
    return est_rd

@app.route('/')
def home():
    #read data sent from mindsphere in get argument
    print("-----------  newrequest... ------------")
    newph = request.args.get('ph', default = '', type = float)
    newlux = request.args.get('lux', default = 0, type = float)
    newrd = request.args.get('rd', default = 0, type = float)
    print("mindsphere sent lux:" + str(newlux))
    print("mindsphere sent ph:" + str(newph))
    print("mindsphere sent rd:" +str(newrd))

    #update cache of stored 2 values for prediction
    globals()['lux1'] = globals()['lux2']
    globals()['ph1'] = globals()['ph2']
    globals()['rd1'] = globals()['rd2']
    globals()['lux2'] = newlux 
    globals()['ph2'] = newph
    globals()['rd2'] = newrd

    response = str(make_prediction()[0])
    print("prediction = " + str(response))
    print("-----------  finished... ------------")
    return response

if __name__ == "__main__":
    
    #flask stuff
    port = int(os.environ.get('PORT', 2052))
    app.run(debug=True, host='0.0.0.0', port=port)
