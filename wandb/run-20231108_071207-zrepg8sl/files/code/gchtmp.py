
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from common.utils import load_model, setup_run
from models.renet import RENet

padim = lambda x, h_max: np.concatenate((x, x.view(-1)[0].copy().expand(1, 3, h_max - x.shape[2], x.shape[3]) / 1e20),
                                        axis=0) if x.shape[2] < h_max else x



def attn_heatmap_cca(cam, img_s, img_q):
    h_max = int(np.max([img_s.shape[2], img_q.shape[2]]))
    targets = [ClassifierOutputTarget(281)]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=img_s, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(img_s, grayscale_cam, use_rgb=True)
    img_s_heatmap = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    
    grayscale_cam = cam(input_tensor=img_q, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(img_q, grayscale_cam, use_rgb=True)
    img_q_heatmap = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    
    im = np.concatenate((padim(img_s_heatmap, h_max), padim(img_q_heatmap, h_max)), axis=1)
    #print("im shape is ", im.shape)
    plt.imshow(im)
    plt.savefig('attn_cca.png')
    return plt
def attn_heatmap_sca(cam, img_s, img_q):
    h_max = int(np.max([img_s.shape[2], img_q.shape[2]]))
    targets = [ClassifierOutputTarget(281)]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=img_s, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam[0, :]
    img_s_heatmap = show_cam_on_image(img_s, grayscale_cam, use_rgb=True)
    
    grayscale_cam = cam(input_tensor=img_q, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(img_q, grayscale_cam, use_rgb=True)
    img_q_heatmap = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    im = np.concatenate((padim(img_s_heatmap, h_max), padim(img_q_heatmap, h_max)), axis=1)
    #print("im shape is ", im.shape)
    plt.imshow(im)
    plt.savefig('attn_sca.png')
    return plt

def attn_ori(img_s, img_q):
    # Resize images to the same height
    h_max = max(img_s.size[1], img_q.size[1])
    img_s = img_s.resize((int(img_s.size[0] * h_max / img_s.size[1]), h_max), Image.ANTIALIAS)
    img_q = img_q.resize((int(img_q.size[0] * h_max / img_q.size[1]), h_max), Image.ANTIALIAS)
    
    # Concatenate the images
    total_width = img_s.size[0] + img_q.size[0]
    im = Image.new('RGB', (total_width, h_max))
    im.paste(img_s, (0, 0))
    im.paste(img_q, (img_s.size[0], 0))
    #print("im shape is ", im.shape)
    plt.imshow(im)
    plt.savefig('attn_ori.png')
    return plt
    



image_size = 84
resize_size = 92

transform = transforms.Compose([
    transforms.Resize([resize_size, resize_size]),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                         np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

if __name__ == '__main__':
    # load data
    img_s_path = '/jet/home/lisun/work/xinliu/fewshot/Renet-MLTI/z-line_.jpg'
    img_q_path = '/jet/home/lisun/work/xinliu/fewshot/Renet-MLTI/z-line.png'
    img_s_bs = transform(Image.open(img_s_path).convert('RGB'))
    img_q_bs = transform(Image.open(img_q_path).convert('RGB'))
    attn_ori(Image.open(img_s_path).convert('RGB'), Image.open(img_q_path).convert('RGB'))
    img_s = img_s_bs.cuda()
    img_q = img_q_bs.cuda()
    #print(img_s.size())
    #img_s = img_s.permute(1,2,0)
    #img_q = img_q.permute(1,2,0)
    #(3,84,84) -> (1,3,84,84)
    img_s = img_s.unsqueeze(0).repeat(1, 1, 1, 1)
    #print(img_s.size())
    
    img_q = img_q.unsqueeze(0).repeat(1, 1, 1, 1)

    # load model for extract feature and attn
    args = setup_run(arg_mode='test')
    ''' define model '''
    model = RENet(args).cuda()
    pre_path = '/jet/home/lisun/work/xinliu/fewshot/Renet-MLTI/checkpoints/isic/5shot-2way/test222/max_acc.pth'
    model = load_model(model, pre_path)
    #model = nn.DataParallel(model, device_ids=args.device_ids)
    #model.module.mode = 'encoder'
    # ([1, 640, 5, 5])
    
    scacam = GradCAM(model=model, target_layers=model.scr_module, use_cuda=True)
    
    
    
    #model.module.mode = 'ccaother'    
    ccacam = GradCAM(model=model, target_layers=model.cca_module,use_cuda=True)




    
    attn_heatmap_sca(scacam, img_s_bs, img_q_bs)
    attn_heatmap_cca(ccacam, img_s_bs, img_q_bs)
    #attn_heatmap(feature_img_s, feature_img_q, attn_s, attn_q)

    print(">>>>>>>>>>>>>>>>> finish")

