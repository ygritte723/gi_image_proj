import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from common.utils import load_model, setup_run
from models.renet import RENet

padim = lambda x, h_max: np.concatenate((x, x.view(-1)[0].copy().expand(1, 3, h_max - x.shape[2], x.shape[3]) / 1e20),
                                        axis=0) if x.shape[2] < h_max else x



def attn_heatmap_cca(img_s, img_q, attn_s_qs, attn_q_qs):
    h_max = int(np.max([img_s.shape[2], img_q.shape[2]]))
    attn_s_qs_normalized = (attn_s_qs - attn_s_qs.min()) / (attn_s_qs.max() - attn_s_qs.min())
    attn_q_qs_normalized = (attn_q_qs - attn_q_qs.min()) / (attn_q_qs.max() - attn_q_qs.min())
    img_s_heatmap = show_heatmap_on_image(img_s, attn_s_qs_normalized)
    img_q_heatmap = show_heatmap_on_image(img_q, attn_q_qs_normalized)
    im = np.concatenate((padim(img_s_heatmap, h_max), padim(img_q_heatmap, h_max)), axis=1)
    #print("im shape is ", im.shape)
    plt.imshow(im)
    plt.savefig('attn_cca.png')
    return plt
def attn_heatmap_sca(img_s, img_q, attn_s_qs, attn_q_qs):
    h_max = int(np.max([img_s.shape[2], img_q.shape[2]]))
    attn_s_qs_normalized = (attn_s_qs - attn_s_qs.min()) / (attn_s_qs.max() - attn_s_qs.min())
    attn_q_qs_normalized = (attn_q_qs - attn_q_qs.min()) / (attn_q_qs.max() - attn_q_qs.min())
    img_s_heatmap = show_heatmap_on_image(img_s, attn_s_qs_normalized)
    img_q_heatmap = show_heatmap_on_image(img_q, attn_q_qs_normalized)
    im = np.concatenate((padim(img_s_heatmap, h_max), padim(img_q_heatmap, h_max)), axis=1)
    #print("im shape is ", im.shape)
    plt.imshow(im)
    plt.savefig('attn_sca.png')
    return plt

def show_heatmap_on_image(img, attn):
    heatmap = cv2.applyColorMap(np.uint8(255 * attn), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    #heatmap = np.transpose(heatmap, [2,0,1])
    img_ = np.float32(img) / 255
    attended_img = heatmap + np.float32(img_)
    attended_img = attended_img / np.max(attended_img)
    return np.uint8(255 * attended_img)


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
    img_s_path = '/jet/home/lisun/work/xinliu/images/fewshot/kvasirv2/dyed-lifted-polyps/0053d7cd-549c-48cd-b370-b4ad64a8098a.jpg'
    img_q_path = '/jet/home/lisun/work/xinliu/images/fewshot/kvasirv2/dyed-lifted-polyps/007d5aa7-7289-4bad-aa4a-5c3a259e9b19.jpg'
    img_s_bs = transform(Image.open(img_s_path).convert('RGB'))
    img_q_bs = transform(Image.open(img_q_path).convert('RGB'))
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
    model = nn.DataParallel(model, device_ids=args.device_ids)

    model.module.mode = 'encoder'
    # ([1, 640, 5, 5])
    feature_img_s = model(img_s)
    feature_img_q = model(img_q)

    model.module.mode = 'ccaother'
    attn_s, attn_q = model((feature_img_s, feature_img_q))
    attn_s, attn_q = torch.squeeze(attn_s), torch.squeeze(attn_q)
    attn_s = attn_s.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, 5, 5]
    attn_q = attn_q.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, 5, 5]

    # Resize
    attn_s = F.interpolate(attn_s, size=(84, 84), mode='bilinear', align_corners=False).squeeze()
    # Resize
    attn_q = F.interpolate(attn_q, size=(84, 84), mode='bilinear', align_corners=False).squeeze()
    # Convert back to original format
    
    feature_img_s = torch.squeeze(feature_img_s).mean(dim=0)
    feature_img_s = F.interpolate(feature_img_s.unsqueeze(0).unsqueeze(0), size=(84, 84), mode='bilinear', align_corners=False).squeeze()
    feature_img_q = torch.squeeze(feature_img_q).mean(dim=0)
    feature_img_q = F.interpolate(feature_img_q.unsqueeze(0).unsqueeze(0), size=(84, 84), mode='bilinear', align_corners=False).squeeze()

    # draw cam
    print("feat: ", feature_img_s.size())
    print("attn: ", attn_s.size())    
    feature_img_s = feature_img_s.cpu().detach().numpy()
    feature_img_q = feature_img_q.cpu().detach().numpy()
    attn_s = attn_s.cpu().detach().numpy()
    attn_q = attn_q.cpu().detach().numpy()
    
    #img_s_bs = torch.squeeze(feature_img_s) 
    #img_q_bs = torch.squeeze(feature_img_q)    
    img_s_bs = img_s_bs.permute(1,2,0)
    img_q_bs = img_q_bs.permute(1,2,0)
    print("img_size: ", img_s_bs.size())
    
    attn_heatmap_sca(img_s_bs, img_q_bs, feature_img_s, feature_img_q)
    attn_heatmap_cca(img_s_bs, img_q_bs, attn_s, attn_q)
    #attn_heatmap(feature_img_s, feature_img_q, attn_s, attn_q)

    print(">>>>>>>>>>>>>>>>> finish")

