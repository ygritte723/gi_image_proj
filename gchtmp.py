import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import resnet18, resnet50

if __name__ == '__main__':
    # load data
    img_s_path = 'z-line.png'
    img_q_path = 'z-line1.png'
    paths = [img_s_path, img_q_path]
    img_s = cv2.imread(img_s_path)
    img_q = cv2.imread(img_q_path)
    h_max = max(img_s.shape[1], img_q.shape[1])

    img_s = cv2.resize(img_s,(int(img_s.shape[0] * h_max / img_s.shape[1]), h_max), cv2.INTER_CUBIC)
    img_q = cv2.resize(img_q,(int(img_q.shape[0] * h_max / img_q.shape[1]), h_max), cv2.INTER_CUBIC)
    im = cv2.hconcat([img_s, img_q])
    cv2.imwrite('attn_images/'+'ori_ghattn.png', im)

    # Load model resnet18
    model = resnet18(pretrained=True)
    # Pick up layers for visualization
    target_layers = [model.layer4[-1]]
    #ScoreCAM, AblationCAM cannot be used
    # GradCAM, EigenCAM is good
    cams = [GradCAM, EigenCAM]
    
    i = 0
    for cams1 in cams:
        i = i+1
        cam = cams1(model=model, target_layers=target_layers, use_cuda=False)
        cam_images = []
        for path in paths:
            rgb_img = Image.open(path).convert('RGB')
            # Max min normalization
            rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
            # Create an input tensor image for your model
            input_tensor = torchvision.transforms.functional.to_tensor(rgb_img).unsqueeze(0).float()

            grayscale_cam = cam(input_tensor=input_tensor)

            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            cam_images.append(cam_image)
        #cam_output_path = str(i)+'_cam.jpg'
        #cv2.imwrite(cam_output_path, cam_image)
        img_s_att = cam_images[0]
        img_q_att = cam_images[1]
        h_max = max(img_s_att.shape[1], img_q_att.shape[1])

        img_s_att = cv2.resize(img_s_att,(int(img_s_att.shape[0] * h_max / img_s_att.shape[1]), h_max), cv2.INTER_AREA)
        img_q_att = cv2.resize(img_q_att,(int(img_q_att.shape[0] * h_max / img_q_att.shape[1]), h_max), cv2.INTER_AREA)
    
        # Concatenate the images
        #total_width = img_s_att.shape[0] + img_q_att.shape[0]
        #im = np.zeros((total_width, h_max,3), dtype=np.uint8)
        #im = Image.new('RGB', (total_width, h_max))
        #im.paste(Image.fromarray(img_s_att), (0, 0))
        #im.paste(Image.fromarray(img_q_att), (img_s_att.shape[0], 0))
        #print("im shape is ", im.shape)
        im = cv2.hconcat([img_s_att, img_q_att])
        cv2.imwrite('attn_images/'+str(i)+'_ghattn.png', im)
        #plt.imshow(im)
        #plt.savefig('attn_images/'+str(i)+'_ghattn.png')




    print(">>>>>>>>>>>>>>>>> finish")

