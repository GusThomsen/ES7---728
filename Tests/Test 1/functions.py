import torch
from torch import nn
import HeaterLossFunc as HLF
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

device = "cuda"
def ImportHelperFunctions():
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    import sklearn
    from sklearn.model_selection import train_test_split
    import requests
    from pathlib import Path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #Import helper functions
    if Path("helper_function.py").is_file():
        print("helper_functions.py already exists, skipping download")
    else:
        print("Download helper_functions.py")
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
        with open("helper_functions.py", "wb") as f:
            f.write(request.content)

    from helper_functions import plot_predictions, plot_decision_boundary
    
    
    
    

def CreateModel(hiddenLayerNum=1, inputNum =1, outputNum=1, hiddenUnitNum = 8, nonLinearFunction = nn.ReLU()):
    class Model(nn.Module):
        def __init__(self,  hidden_units = hiddenUnitNum):
            super().__init__()
            hidden_layers_temp = []

            k = 1
            k1 = 1
            for i in range(hiddenLayerNum):
                if hiddenLayerNum != 0:
                    #if i%10 == 0:
                     #   k1 = k1*2
                    hidden_layers_temp.append(nn.Linear(in_features = int(hidden_units/k),
                    out_features = int(hidden_units/k1)))
                    hidden_layers_temp.append(nonLinearFunction)
                    #k = k1
            self.layer_In = nn.Linear(in_features=inputNum, out_features=hidden_units)
            self.layer_Out = nn.Linear(in_features = int(hidden_units/k1), out_features = outputNum)
            self.linear_layer_stack = nn.Linear(in_features=inputNum, out_features=outputNum)
            self.activation = nonLinearFunction
            if hiddenLayerNum == 0:
                self.linear_layer_stack = nn.Linear(in_features=inputNum, out_features=outputNum)
                
            
            else:
                self.linear_layer_stack = nn.Sequential(*hidden_layers_temp)
                #self.linear_layer_stack = self.layer_Out(nonLinearFunction(self.layer_In()))
        
            
            
        def forward(self, x):
            
            x = self.activation(self.layer_In(x))
            
            if hiddenLayerNum != 0:
                x = self.linear_layer_stack(x)
                
            x = self.layer_Out(x)
            return x
    return Model()
    
    
    
def accuracy_fn(y_true, y_pred):
    
    correct = torch.zeros(len(y_true))
    for i in range(len(y_true)):
        if torch.abs(y_true[i]-y_pred[i]) <= 0.01:
            correct[i] = 1
        else:
            correct[i] = 0
    correct = correct.sum().item()
    acc = (correct/len(y_pred))*100
    return acc
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

loss_values = []
test_loss_values = []
epoch_values = []
learning_rates = []
def TrainModel(model = CreateModel(), epochs=1000, X_train = 0, X_test = 0, y_train = 0, y_test = 0, loss_fn = nn.L1Loss(), lr = 0.01, isClassification = False):
    
    schedulernum = 10000
    optimizer = torch.optim.Adam(params=model.parameters(),
                            lr = lr,weight_decay=0.01)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    loss = 1000000
    epoch = 0
    if isClassification == True:
        loss_fn = nn.BCEWithLogitsLoss()
    while loss > 0.01:
        epoch = epoch +1
    #for epoch in range(epochs):
        ### Training
        if epoch >= epochs:
            break
        model.train()
    
        y_pred = torch.zeros_like(y_train)
        # 1. Forward pass
        if isClassification == True:
            y_logits = model(X_train).squeeze().to("cuda")
            y_pred = torch.round(torch.sigmoid(y_logits)).to("cuda")
            # 2. Calculate loss/acc
            loss = loss_fn(y_logits, y_train)
        else:
            #print("Hej")
            y_pred = model(X_train)
            # 2. Calculate loss/acc
            loss = loss_fn(y_pred, y_train)
        #acc = accuracy_fn(y_true=y_train,
                      #y_pred=y_pred)
    
        # optimizer zero grad
        optimizer.zero_grad()
    
        # 4. Loss backwards
        loss.backward()
    
        # 5. Optimizer
        optimizer.step()
    
        ### Testing
        model.eval()
        with torch.inference_mode():
            y_pred = model(X_train)
            test_pred = 0
            # 1. Forward pass
            if isClassification == True:
                
                test_logits = model(X_test).squeeze()
                test_pred = torch.round(torch.sigmoid(test_logits))

                # 2. Calculate loss and accuracy
                test_loss = loss_fn(test_logits,
                            y_test)
        
            else:
                test_pred = torch.tensor(0).to(device)
                test_loss = torch.tensor(0).to(device)
                test_pred = model(X_test)

                # 2. Calculate loss and accuracy
                test_loss = loss_fn(test_pred,y_test)
                
            #test_acc = accuracy_fn(y_true=y_test,
                              # y_pred=test_pred)
            if epoch %100 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {0:.2f}% | Test loss: {test_loss:.5f}, Test acc: {0:.2f}% | Learning Rate: {get_lr(optimizer):.5f}")
                loss_values.append(loss.to("cpu"))
                test_loss_values.append(test_loss.to("cpu"))
                epoch_values.append(epoch)
                learning_rates.append(get_lr(optimizer))
            if epoch %schedulernum == 0:
                scheduler1.step()
                schedulernum = schedulernum*3
            if epoch % 950 == 0:
                # 3. Save the models state_dict
                MODEL_PATH = Path("models")
                MODEL_NAME = "TestModel2_Coords.pth"
                MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
                print(f"Saving model to: {MODEL_SAVE_PATH}")
                torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH) # Can also just save the model
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {0:.2f}% | Test loss: {test_loss:.5f}, Test acc: {0:.2f}% | Learning Rate: {get_lr(optimizer):.5f}")
    return y_pred, test_pred, loss_values, test_loss_values,epoch_values,learning_rates




def TrainModel2(model = CreateModel(), epochs=1000, X_train = 0, X_test = 0, y_train = 0, y_test = 0, loss_fn = nn.L1Loss(), lr = 0.01, isClassification = False):
    
    schedulernum = 10000
    optimizer = torch.optim.Adam(params=model.parameters(),
                            lr = lr,weight_decay=0.01)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    loss = 1000000
    epoch = 0
    if isClassification == True:
        loss_fn = nn.BCEWithLogitsLoss()
    while loss > 0.01:
        epoch = epoch +1
    #for epoch in range(epochs):
        ### Training
        if epoch >= epochs:
            break
        model.train()
    
        y_pred = torch.zeros_like(y_train)
        # 1. Forward pass
        if isClassification == True:
            y_logits = model(X_train).squeeze().to("cuda")
            y_pred = torch.round(torch.sigmoid(y_logits)).to("cuda")
            # 2. Calculate loss/acc
            loss = loss_fn(y_logits, y_train)
        else:
            #print("Hej")
            y_pred = model(X_train)
            # 2. Calculate loss/acc
            loss = loss_fn(y_pred, y_train)
        #acc = accuracy_fn(y_true=y_train,
                      #y_pred=y_pred)
    
        # optimizer zero grad
        optimizer.zero_grad()
    
        # 4. Loss backwards
        loss.backward()
    
        # 5. Optimizer
        optimizer.step()
    
        ### Testing
        model.eval()
        with torch.inference_mode():
            y_pred = model(X_train)
            test_pred = 0
            # 1. Forward pass
            if isClassification == True:
                
                test_logits = model(X_test).squeeze()
                test_pred = torch.round(torch.sigmoid(test_logits))

                # 2. Calculate loss and accuracy
                test_loss = loss_fn(test_logits,
                            y_test)
        
        
                
            #test_acc = accuracy_fn(y_true=y_test,
                              # y_pred=test_pred)
            if epoch %100 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {0:.2f}% | Learning Rate: {get_lr(optimizer):.5f}")
                loss_values.append(loss.to("cpu"))
                test_loss_values.append(0)
                epoch_values.append(epoch)
                learning_rates.append(get_lr(optimizer))
            if epoch %schedulernum == 0:
                scheduler1.step()
                schedulernum = schedulernum*3
            if epoch % 950 == 0:
                # 3. Save the models state_dict
                MODEL_PATH = Path("models")
                MODEL_NAME = "TestModel2_Coords.pth"
                MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
                print(f"Saving model to: {MODEL_SAVE_PATH}")
                torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH) # Can also just save the model
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {0:.2f}% | Learning Rate: {get_lr(optimizer):.5f}")
    return y_pred, test_pred, loss_values, test_loss_values,epoch_values,learning_rates







# Constants
h = 1000  # Convective heat transfer coefficient (W/m^2·K), adjust as needed
T_water = 300  # Temperature of the water (K), assumed constant



# Define physical parameters for heater
rho_h = 21.45e3     # Mass density [kg/m^3]
c_h = 133           # Specific heat capacity [J/kg K]
k_h = 77.8          # Thermal conductivity [W/m K]
'''
rho_h = 997     # Mass density [kg/m^3]
c_h = 418.4         # Specific heat capacity [J/kg K]
k_h = 0.6          # Thermal conductivity [W/m K]
'''

# Define heater parameters
d = 20e-6                           # Diamater of heater [m]
l = 0.13                            # lenght of heater [m]
resistivity = 10.6e-8               # [ohm m]

R = l*resistivity/((np.pi * d**2)/4)  # Resistance [ohm]
I = 75e-3                           # Current amplitude [A]
omega = 2*np.pi*1                   # Applied AC frequency [rad/s]
P = 1000 * R * I**2                 # Artificially inflated peak heat power [W]

T_ambient = 300  # Ambient temperature in K




def TrainModelPI(model = CreateModel(), epochs=5000, X_train = 0, X_test = 0, y_train = 0, y_test = 0, loss_fn = nn.L1Loss(), lr = 0.01, isClassification = False):
    schedulernum = 30000
    optimizer = torch.optim.Adam(params=model.parameters(),
                            lr = lr,weight_decay=0.01)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    loss = 1000
    epoch = 0
    
    y_pred = torch.zeros_like(y_train)
    if isClassification == True:
        loss_fn = nn.BCEWithLogitsLoss()
    #while loss > 0.001:
       # epoch = epoch +1
    for epoch in range(epochs):
        ### Training
        model.train()
    
        
        # 1. Forward pass
        if isClassification == True:
            y_logits = model(X_train).squeeze().to("cuda")
            y_pred = torch.round(torch.sigmoid(y_logits)).to("cuda")
            # 2. Calculate loss/acc
            loss = loss_fn(y_logits, y_train)
        else:
            # optimizer zero grad
            optimizer.zero_grad()
            #init_pred = model(torch.zeros(1).to("cuda"))
            #init_pred.to("cuda")
            #print("Hej")
            X_train.requires_grad_(True)
            y_pred = model(X_train)
            
            # 2. Calculate loss/acc
            
            loss = loss_fn(y_pred, y_train)

            
            
            T_t = torch.autograd.grad(y_pred, X_train,torch.ones_like(y_pred),create_graph=True,retain_graph=True)[0]         # dT/dt
            T_t = T_t.to("cuda")

            Q = JoulesHeating(X_train[:,0])
            #print(Q)
            Q = Q.to("cuda")
            
            # Heater's energy balance
            #residual = rho_h * c_h * T_t[:, 0] - k_h * (y_pred - T_ambient) - Q #- h * (y_pred - T_water)
            residual = -T_t[:, 0] + (k_h * (y_pred - T_ambient)+ Q )/(rho_h * c_h) #+ Q 
            PIloss = (1e-4)*torch.mean(residual**2)
            loss = loss+PIloss
            
        acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
  
    
        # 4. Loss backwards
        loss.backward()
    
        # 5. Optimizer
        optimizer.step()
    
        ### Testing
        model.eval()
        with torch.inference_mode():
            y_pred = model(X_train)
            test_pred = 0
            # 1. Forward pass
            if isClassification == True:
                
                test_logits = model(X_test).squeeze()
                test_pred = torch.round(torch.sigmoid(test_logits))

                # 2. Calculate loss and accuracy
                test_loss = loss_fn(test_logits,
                            y_test)
        
            else:
                test_pred = model(X_test)
                # 2. Calculate loss and accuracy
                test_loss = loss_fn(test_pred,y_test)
            test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
            if epoch %schedulernum == 0:
                scheduler1.step()
                schedulernum = schedulernum*3
                
            if epoch %100 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {0:.2f}% | Test loss: {test_loss:.5f}, Test acc: {0:.2f}% | Learning Rate: {get_lr(optimizer):.5f}")
                loss_values.append(loss.to("cpu"))
                test_loss_values.append(test_loss.to("cpu"))
                epoch_values.append(epoch)
                learning_rates.append(get_lr(optimizer))
                MODEL_PATH = Path("models")
                MODEL_NAME = "TestModelTEST.pth"
                MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
                print(f"Saving model to: {MODEL_SAVE_PATH}")
                torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH) # Can also just save the model
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    return y_pred, test_pred, loss_values, test_loss_values,epoch_values,learning_rates



def TrainModelPINEW(model = CreateModel(), epochs=5000, X_train = 0, X_test = 0, y_train = 0, y_test = 0, loss_fn = nn.L1Loss(), lr = 0.01, isClassification = False):
    schedulernum = 30000
    optimizer = torch.optim.Adam(params=model.parameters(),
                            lr = lr,weight_decay=0.01)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    loss = 1000
    epoch = 0
    
    y_pred = torch.zeros_like(y_train)
    if isClassification == True:
        loss_fn = nn.BCEWithLogitsLoss()
    #while loss > 0.001:
       # epoch = epoch +1
    for epoch in range(epochs):
        ### Training
        model.train()
    
            # optimizer zero grad
        optimizer.zero_grad()
            #init_pred = model(torch.zeros(1).to("cuda"))
            #init_pred.to("cuda")
            #print("Hej")
        X_train.requires_grad_(True)
        y_pred = model(X_train)
            
            # 2. Calculate loss/acc
            
        loss = loss_fn(y_pred, y_train)

            
            
        T_t = torch.autograd.grad(y_pred, X_train,torch.ones_like(y_pred),create_graph=True,retain_graph=True)[0]         # dT/dt
        T_t = T_t.to("cuda")
        Q = JoulesHeating(X_train)
        Q = Q.to("cuda")
        residual = rho_h * c_h *T_t-k_h*(y_pred-0) - Q
   
        loss = loss+(1e-14)*torch.mean(residual**2)
            
            
        acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
  
    
        # 4. Loss backwards
        loss.backward()
    
        # 5. Optimizer
        optimizer.step()
    
        ### Testing
        model.eval()
        with torch.inference_mode():
            y_pred = model(X_train)
            test_pred = 0
            # 1. Forward pass
            if isClassification == True:
                
                test_logits = model(X_test).squeeze()
                test_pred = torch.round(torch.sigmoid(test_logits))

                # 2. Calculate loss and accuracy
                test_loss = loss_fn(test_logits,
                            y_test)
        
            else:
                test_pred = model(X_test)
                # 2. Calculate loss and accuracy
                test_loss = loss_fn(test_pred,y_test)
            test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
            if epoch %schedulernum == 0:
                scheduler1.step()
                schedulernum = schedulernum*3
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
                MODEL_PATH = Path("models")
                MODEL_NAME = "TestModelTEST.pth"
                MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
                print(f"Saving model to: {MODEL_SAVE_PATH}")
                torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH) # Can also just save the model
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    return y_pred, test_pred






def JoulesHeating(time):
    #time = time.to("cpu")
    #time1 = time.detach()
    Q = P*(1 - torch.cos(2*omega*time))/(2*torch.pi*d*d*k_h)                  # Joules heating
    return Q




def HeaterLoss_Single(T,time):     
    # Loss function for heater, when no position
    T_t = torch.autograd.grad(T, time,torch.ones_like(T),create_graph=True)[0]         # dT/dt

    T_t = T_t.to("cuda")
    Q = JoulesHeating(time)
    Q = Q.to("cuda")
    residual = rho_h * c_h *T_t-k_h*(T-0) - Q
   
    loss = (1e-4)*torch.mean(residual**2)
    return loss

def plot_predictions(train_data = torch.tensor(0),
                     train_labels = torch.tensor(0),
                     test_data = torch.tensor(0),
                     test_labels = torch.tensor(0),
                     train_predictions = None,
                     test_predictions=None):
    
    
    train_data = train_data.to("cpu")
    train_labels = train_labels.to("cpu")
    test_data = test_data.to("cpu")
    test_labels = test_labels.to("cpu")

        
    if test_predictions is not None:
        test_predictions = test_predictions.to("cpu")
    if train_predictions is not None:
        train_predictions = train_predictions.to("cpu")
        train_predictions = torch.cat((train_predictions,test_predictions),0)
    #Plots Training data, test data and compares predictions.
    predictionX = torch.cat((train_data,test_data),0)
    plt.figure(figsize=(10,7))
    plt.subplot(1,3,1)
    plotX = torch.cat((train_data,test_data), 0)
    plotY = torch.cat((train_labels,test_labels), 0)
    plotX = plotX.detach().numpy()
    plotY=plotY.detach().numpy()
    predictionX = predictionX.detach().numpy()
    train_data = train_data.detach().numpy()
    test_data = test_data.detach().numpy()
    plt.plot(plotX, plotY, color="tab:blue", linewidth=3, alpha=1,label="Data")
    plt.legend(prop={"size": 14})
    if train_predictions is not None:
        #Plot them
        plt.subplot(1,3,2)    
        plt.plot(predictionX, train_predictions, color="tab:red", linewidth=2, alpha=0.8,label="Predictions")
        

     # Show the legend
    plt.legend(prop={"size": 14})
    
    
    plt.subplot(1,3,3)
     # Plot training data in blue
    plt.scatter(train_data, train_labels, c="tab:blue", s=20, label="Training data")
    
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="tab:green", s=20, label="Testing data")
    
    # Are there predictions?
    
 
        
    # Show the legend
    plt.legend(prop={"size": 14})
    
    file = "plots/nn_RealVsPredicted.png"
    plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
    
    
    
    
    
def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)
    
    
    
def SaveFigureForGIF(i,files):
    file = "plots/nn_%.8i.png"%(i+1)
    plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
    files.append(file)
    return files






def plot_results(model, plot=True, save=False, save_path='results.png'):

    x = torch.linspace(-1, 1, 100)
    y = torch.linspace(-1, 1, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)
    phi = model(xy)
    phi = phi.reshape(100, 100).detach().numpy()
    plt.contourf(X, Y, phi, 100, cmap=plt.cm.hot)
    C = plt.contour(X, Y, phi, 10, colors='black')
    plt.clabel(C, inline=True, fontsize=10)
    plt.xlabel('x')
    plt.ylabel('y')
    if plot:
        plt.pause(0.1)
    if save:
        plt.savefig(save_path)
        
        
        
def contour_at_timestamp(model,timestamp,timesteps,temperatures,plot=False,save=True,save_path = 'Contour_plot.png'):
    x = torch.linspace(-0.01, 0.01, 31)
    y = torch.linspace(-0.01, 0.01, 3)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    xy = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)
    time = torch.linspace(0,10,timesteps).unsqueeze(0).repeat(93,1)
    print(xy.shape)
    print(torch.transpose(temperatures,0,1).shape)
    print(time.shape)
    input = torch.cat((xy,torch.transpose(temperatures,0,1),time),dim=1)
    input = input.to(device)
    print(input.shape)
    Temp = model(input)
    print(Temp.shape)
    Temp = Temp.to('cpu')
    Temp = Temp.reshape(100, 93, -1).detach().numpy()

    num_frames = Temp.shape[2]
    frame_index = int((timestamp / 10)*num_frames - 1)

    fig, ax = plt.subplots(figsize = (8, 6))
    # print(Temp[:,:,frame_index])
    contour = ax.contour(X.numpy(), Y.numpy(), Temp[:, :, frame_index], levels = 100, cmap = plt.cm.hot)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Time: {timestamp:.2f}s')
    plt.colorbar(contour, ax=ax, label = 'Temperature')
    if plot:
        plt.pause(0.1)
    if save:
        plt.savefig(save_path)