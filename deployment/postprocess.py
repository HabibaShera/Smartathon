import numpy as np
import cv2
from PIL import Image
import io
import base64
INDEX_MAP = {1: 'CLUTTER_SIDEWALK',
 2: 'POTHOLES',
 3: 'CONSTRUCTION_ROAD',
 4: 'BAD_BILLBOARD',
 5: 'GRAFFITI',
 6: 'SAND_ON_ROAD',
 7: 'FADED_SIGNAGE',
 8: 'BROKEN_SIGNAGE',
 9: 'UNKEPT_FACADE',
 10: 'BADSTREET_LIGHT',
 0: 'GARBAGE'}

class YoloOutputs:
  def __init__(self,coords=None,scores=None,classes=None,label_map=INDEX_MAP):
    self.coords =coords
    self.scores = scores
    self.classes = label_map[classes]
  @classmethod
  def from_yolo(cls,yolo_outs):
    yolos= []
    for i in range(yolo_outs.shape[0]):
      coords= yolo_outs[i][:4].cpu().numpy().astype(int)
      score = yolo_outs[i][4].item()
      cat =int(yolo_outs[i][5].item())
      yolos.append(cls(coords,score,cat))
    return yolos
  


def to_numpy(yolos,scores=None):
    coords = np.stack([y.coords for y in yolos])
    if scores:
        classes = [y.classes+f":{(y.scores*100):.4f}%" for y in yolos ]
    else:
        classes = [y.classes for y in yolos ]    
    return coords,classes   

def infer_img(y,st_point,end_point,texts,
             thc=10,label_color=(255,0,0),bbox_color=None):
  font = cv2.FONT_HERSHEY_DUPLEX
  num_boxes = st_point.shape[0] if len(st_point.shape)>1 else 1
  st_point = st_point.reshape(-1,2)
  end_point = end_point.reshape(-1,2)
  p = (st_point+end_point)//2
  p[:,1] = st_point[:,1]
  for i in range(num_boxes):
    st = st_point[i]
    end = end_point[i]
    y = cv2.rectangle(y,st,end,color=bbox_color,thickness=thc)
    y = cv2.putText(y,texts[i],p[i],font,1,label_color,1,cv2.LINE_AA)
  return y  
  
def convert_string(img):    
    img = Image.fromarray(img.astype("uint8"))
    raw_bytes = io.BytesIO()
    img.save(raw_bytes,"JPEG")
    raw_bytes.seek(0)
    return  base64.b64encode(raw_bytes.read())
	  
          