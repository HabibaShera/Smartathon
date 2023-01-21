import torch
from torch import nn 
import torchvision.transforms as T




class LetterBox(nn.Module):
    """Resize image and padding for detection, instance segmentation, pose"""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        super().__init__()
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
    def _make_border(self,img,top,bot,left,right):
        h,w = self.new_shape
        img_shape = (3,h,w)
        new_img = torch.ones(img_shape,dtype=img.dtype).to(img.device)*114
        new_img[:,top:w-bot,left:h-right] = img
        return new_img

    def forward(self, image):
      
        img = torch.tensor(image).permute(2,0,1)
        shape = img.shape[1:]  # current shape [height, width]
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = torch.remainder(dw, self.stride), torch.remainder(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0, 0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            img = T.functional.resize(img, (new_unpad[1],new_unpad[0]))     
        top, bottom = torch.tensor(int(round(dh - 0.1))), torch.tensor(int(round(dh + 0.1)))
        left, right = torch.tensor(int(round(dw - 0.1))), torch.tensor(int(round(dw + 0.1)))
        img = self._make_border(img,top,bottom,left,right) # add border
        return img.unsqueeze(0)/255.0,torch.tensor(shape)

test = LetterBox()
def load_preprocess_img(img):
    im ,sh =test(img.copy())
    return im,sh

    

