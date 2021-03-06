# Robust Explainability

A Tutorial on Gradient-Based Attribution Methods for Deep Neural Networks

Full paper can be found here: https://arxiv.org/abs/2107.11400

## Install

1. Clone repo.

``git clone https://github.com/nielseni6/ShiftSmoothedAttributions.git``

2. Make sure to have the following required libraries if you do not already.

>Python 3.8.5
> 
>PyTorch 1.4.0
> 
>torchvision 0.5.0
> 
>matplotlib 3.5.1
> 
>numpy 1.16.3
> 
>robustness 1.2.1.post2
> 
>scikit-image 0.19.1
> 
>opencv-python 4.5.5.62
>
>captum 0.4.1
>
>numpy 1.22.1


3. Pretrained models can be found here: https://drive.google.com/drive/u/0/folders/1KdJ0aK0rPjmowS8Swmzxf8hX6gU5gG2U 
and here: https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0

Add these files to the \model folder.

4. Run the following scripts to generate the figures which can be found in the paper:

Visualize_Saliency_ImageNet.py:
[fig2.pdf](https://github.com/nielseni6/Robust_Explainability_Experiments/files/7874143/fig2.pdf)
![fig2](https://user-images.githubusercontent.com/36169018/149972829-99b81130-e543-41a8-8cd8-6c3771c9c440.png)

Visualize_Target_Class.py:
[fig3.pdf](https://github.com/nielseni6/Robust_Explainability_Experiments/files/7874144/fig3.pdf)
![fig3](https://user-images.githubusercontent.com/36169018/149973069-db4db331-be18-4895-a3d0-6329f4de9980.png)

Visualize_Robust_Grad.py:
[fig4.pdf](https://github.com/nielseni6/Robust_Explainability_Experiments/files/7874145/fig4.pdf)
![fig4](https://user-images.githubusercontent.com/36169018/149973102-3f6ee0f3-80c4-4a0f-b7e8-6fd265ee34d7.png)

For questions contact nielseni6@rowan.edu to get more information on the paper
