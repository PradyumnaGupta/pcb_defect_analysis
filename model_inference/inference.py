import timm
import cv2
import numpy as np
from torch import nn
import albumentations
import numpy as np
import torch
from torchvision.ops import nms

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        # self.model=pretrainedmodels.__dict__['se_resnext101_32x4d']()
        self.model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
        self.relu=nn.ReLU()
        self.linear=nn.Linear(in_features=1000,out_features=6)
        
    def forward(self,images):
        step=self.model(images)
        step=self.relu(step)
        step=self.linear(step)
        return step

def extract_contours_from_image(image, hsv_lower=[0,150,50], hsv_upper=[10,255,255]):
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array(hsv_lower)
    hsv_upper = np.array(hsv_upper)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    offset = 25
    ROI_number = 0
    regions = []
    for c in cnts:
      x, y, w, h = cv2.boundingRect(c)
      regions.append([max(x-offset,0),max(y-offset,0),x+w+offset,y+h+offset])
    boxes = torch.Tensor(regions)
    inds = list(nms(boxes=boxes,scores=torch.Tensor([1]*len(regions)),iou_threshold=0.1).numpy())
    ans = []
    for i in inds :
      ans.append(regions[i])
    return ans

def subtract_image(image_path_1,image_path_2):
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)
    difference = cv2.subtract(image1, image2)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]
    image1[mask != 255] = [0, 0, 255]
    image2[mask != 255] = [0, 0, 255]
    return image2    

def setLabel(im, label, coord):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    baseline = 0

    text = cv2.getTextSize(label, fontface, scale, thickness)
    
    cv2.rectangle(im, coord,(coord[0]+text[0][0],coord[1]-text[0][1]) ,(0,255,0), -1)
    cv2.putText(im, label, coord, fontface, scale, (0,0,0), thickness, 8)
    return im

def model_inference(model,path_1,path_2):

  aug = albumentations.Compose(
    [
        albumentations.Normalize(always_apply=True),
        albumentations.PadIfNeeded(128,128)   
    ]
  )

  img1 = subtract_image(path_1,path_2)
  img2 = subtract_image(path_2,path_1)

  r1 = extract_contours_from_image(img1)
  r2 = extract_contours_from_image(img2)
  r1.extend(r2)
  
  classes = ["Open","Short","Mousebite","Spur","Copper","Pin_hole"]
  # model.cuda()

  for r in r1:
    region = img1[r[1]:r[3],r[0]:r[2]]
    region = cv2.resize(region,(128,128))
    region = aug(image = region)['image']
    region = np.transpose(region,(2,0,1))
    region = torch.Tensor(region).unsqueeze(0)
    output = model(region).argmax(-1).detach().cpu().item()
    img1 = setLabel(img1,classes[output],(r[0],r[1]))

  for r in r1:
    img1 = cv2.rectangle(img1,
                        (r[0],r[1]),
                        (r[2],r[3]),
                        [0,255,0],2)  
  return img1

def load_model():
    model = Net()
    model.load_state_dict(torch.load('saved_weights/effb4-0.9722.pth',map_location=torch.device('cpu')))
    model.eval()
    # model.cuda()
    return model

