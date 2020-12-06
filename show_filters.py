import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np

def main():
    modelname = input("Enter a model name: ")
    model = torch.load(modelname)
    for layer in model.modules():
        if type(layer) == nn.Conv2d:
            print(layer)
            weight_tensor = layer.weight.data.cpu()
            fig, ax = plt.subplots(nrows=6,ncols=6,figsize=(16,12))
            outputs = []
            for i in range(32):
                    npimg = np.array(weight_tensor[i].numpy(), np.float32)
                    #standardize the kernel
                    npimg = (npimg - np.mean(npimg)) / np.std(npimg)
                    npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
                    npimg = npimg.transpose((1, 2, 0))
                    outputs.append(npimg)
            
            for i, axi in enumerate(ax.flat):
                if i < 32:
                    axi.imshow(outputs[i])
                else:
                    axi.set_axis_off()
            plt.show()
            return  

if __name__ == '__main__':
    main()
    