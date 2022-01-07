import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import time
import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from test_model import test_model
import numpy as np
from skimage import img_as_ubyte
import cv2 
from functs import *
#%matplotlib inline
from captum.attr import (
    InputXGradient,
    Saliency,
    IntegratedGradients,
    GuidedGradCam,
    LayerGradCam,
    FeatureAblation,
    NoiseTunnel,
    GuidedBackprop,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For mutliple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def attribute_image_features(algorithm, input, **kwargs):
    model_ft.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=int(predicted_out[itr]),
                                              **kwargs
                                             )
    return tensor_attributions
    
class Args():
    def __init__(self):
        self.loadModel = 'pretrained'
        self.cuda = True
        self.epochs     = 30
        self.batch_size = 1
        self.lr         = 0.001
        self.nsamples = 5
        self.gray_maps = True
        self.squared_maps = True
        self.baseline = torch.tensor(np.float32(0.0*np.random.rand(1, 3, 32, 32)))
        self.set_size = 5
        self.heatmap = True
        
if __name__ == '__main__':
    args = Args()
    lr = args.lr
    cuda = args.cuda
    epochs = args.epochs
    model_path = args.loadModel
    batch_size = args.batch_size
    nsamples = args.nsamples
    gray_maps = args.gray_maps
    squared_maps = args.squared_maps
    baseline = args.baseline
    set_size = args.set_size
    heatmap = args.heatmap
    
    run_num = 16
    set = [79, 2100, 630, 1100, 1501] # Sample set of images
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Load Resnet18
    model_ft = models.resnet18(pretrained=True)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    data_dir = '/data/imagenet/imagenet/'
    num_workers = {'train' : 0,'val'   : 0}#,'test'  : 0}
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'val': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                      for x in ['train', 'val']}

    image_datasets['val'] = torch.utils.data.Subset(image_datasets['val'], set)
    
    remainder = int(1.0 * len(image_datasets['train'])) - int(0.05 * len(image_datasets['train'])) - int(0.95 * len(image_datasets['train']))
    cifar_figset, image_datasets['train'] = torch.utils.data.random_split(image_datasets['train'], 
                                                                [int(0.99 * len(image_datasets['train'])),
                                                                 int(0.01 * len(image_datasets['train'])) + remainder]
                                                                )
    
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=num_workers[x])
                      for x in ['train', 'val']}#, 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}#, 'test']}
    
    
    val_dataloader = dataloaders['train']
    test_dataloader = dataloaders['train']
    
    classes = np.loadtxt("imagenet-data/words.txt", delimiter="\n", dtype='str')
    classes_code = [i[0:9] for i in classes]
    classes = [i[10:] for i in classes]
    
    print('Using Pretrained Model')
        
    gbp = GuidedBackprop(model_ft)
    sa = Saliency(model_ft)
    gxi = InputXGradient(model_ft)
    i_g = IntegratedGradients(model_ft)
    gCAM = GuidedGradCam(model_ft, model_ft.conv1)
    gradCAM = LayerGradCam(model_ft, model_ft.conv1)
    layer_name = "conv1"
    #ablation = FeatureAblation(model_ft)
    
    nt = NoiseTunnel(sa)
    
    #Test
    result_images, predicted_out, labels_list, _ = test_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, nsamples)
    
    print(predicted_out)
    
    attribution_names = [
                      "Saliency", 
                      "Gradient⊙Input", 
                      "GBP", 
                      "IG", 
                      "SmoothGrad",
                      "Guided Grad-CAM",
                      "Grad-CAM",
                      #"Feature Ablation",
                      ]
    attribution_maps = list()
    for x in range(len(attribution_names)):
        attribution_maps.append(list())
    
    for itr in range(nsamples):
        print("Calculating saliency maps for sample ", int(itr + 1))
        data = result_images[itr].clone().detach()
        
        grad_x_image = attribute_image_features(gxi, data)
        
        vanilla_grad = attribute_image_features(sa, data, abs = squared_maps)
        
        gbp_attributions = gbp.attribute(data, target=int(predicted_out[itr]))
        gbp_attributions = gbp_attributions.cpu()
        
        integrated_grad = attribute_image_features(i_g, data, n_steps=50)
        
        smoothgrad = nt.attribute(data, nt_type='smoothgrad', nt_samples=15, target=int(predicted_out[itr]))
        
        gCAM_attributions = attribute_image_features(gCAM, data)
        gCAM_attributions = gCAM_attributions.cpu()
        
        gradCAM_attributions = attribute_image_features(gradCAM, data)
        gradCAM_attributions = gradCAM_attributions.cpu()
        
        #ablation_attributions = attribute_image_features(ablation, data)
        #ablation_attributions = ablation_attributions.cpu()
        
        if squared_maps:
            grad_x_image = abs(grad_x_image)# * grad_x_image
            gbp_attributions = abs(gbp_attributions)# * gbp_attributions
            integrated_grad = abs(integrated_grad)# * integrated_grad
            gCAM_attributions = abs(gCAM_attributions)# * gCAM_attributions
            gradCAM_attributions = abs(gradCAM_attributions)# * gradCAM_attributions
            #ablation_attributions = abs(ablation_attributions)
        
        attribution_maps[0].append(vanilla_grad.clone().detach())
        attribution_maps[1].append(grad_x_image.clone().detach())
        attribution_maps[2].append(gbp_attributions.clone().detach())
        attribution_maps[3].append(integrated_grad.clone().detach())
        attribution_maps[4].append(smoothgrad.clone().detach())
        attribution_maps[5].append(gCAM_attributions.clone().detach())
        attribution_maps[6].append(gradCAM_attributions.clone().detach())
        #attribution_maps[7].append(ablation_attributions.clone().detach())
    
    fig=plt.figure(figsize=(19, 9))
    length, width = nsamples, (len(attribution_maps) + 1)
    
    
    for i in range((len(result_images))):
        
        label_first_word = str(classes[int(labels_list[i])]).split(',')
        fig.add_subplot(length, width, (i * width) + 1)#.set_title(str(label_first_word[0]))
        imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(result_images[i][0].clone().detach().cpu())))
        plt.axis('off')
        
        plt.set_cmap('hot')
        for j in range(len(attribution_maps)):
            
            attribution_map = attribution_maps[j][i].clone().detach().cpu()
            
            if gray_maps:
                if str(attribution_names[j]) != "Grad-CAM":
                    attribution_map = attribution_map[0][0] + attribution_map[0][1] + attribution_map[0][2]
                else:
                    attribution_map = attribution_map[0][0]
            if i == 0:
                fig.add_subplot(length, width, (i * width) + j + 2).set_title(str(attribution_names[j]))
            else:
                fig.add_subplot(length, width, (i * width) + j + 2)
            
            ## Plot as Heatmap ##
            if heatmap == True:
                cv_image = img_as_ubyte(VisualizeImageGrayscale(attribution_map.clone().detach()))
                cv_image = cv2.applyColorMap(cv_image, cv2.COLORMAP_JET)
                img = torchvision.utils.make_grid(VisualizeImageGrayscale(torch.tensor(cv_image)))     # unnormalize
                npimg = img.detach().numpy()
                tpimg = npimg
                plt.imshow(tpimg)
                plt.savefig("imshowfig.png")
            
            ## Plot grayscale ##
            if heatmap == False:
                imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(attribution_map)))
            
            plt.axis('off')
            
    if (squared_maps):
        plt.savefig(str("saved_figs/attr_ImageNet_sq_"+ str(run_num) + "_" + str(layer_name) + ".pdf"))
        #plt.savefig(str("saved_figs/attr_ImageNet_sq_"+ str(run_num) + "_" + str(layer_name) + ".png"))
    else:
        plt.savefig(str("saved_figs/attr_ImageNet_"+ str(run_num) + "_" + str(layer_name) + ".pdf"))
        #plt.savefig(str("saved_figs/attr_ImageNet_"+ str(run_num) + "_" + str(layer_name) + ".png"))
    
