import numpy as np
import matplotlib.pyplot as plt
import torch

def imshow(img):
    img = img     # unnormalize
    npimg = img.detach().numpy()
    tpimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(tpimg)
#    plt.show()
    plt.savefig("imshowfig.png")
    
def VisualizeImageGrayscale(image_3d):
    # Returns grayscale tensor normalized between 0 and 1
    vmin = torch.min(image_3d)
    image_2d = image_3d - vmin
    vmax = torch.max(image_2d)
    return (image_2d / vmax)
    
