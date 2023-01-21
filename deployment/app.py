from img_processor import load_preprocess_img
from flask  import Flask, jsonify, request
from PIL import Image
import torch
import numpy as np
from postprocess import infer_img,to_numpy,YoloOutputs,convert_string



app = Flask(__name__)


DEVICE = torch.device('cpu')
def load_warm():
   model = torch.jit.load('model\mainmodel.pt',map_location=DEVICE)   
   model(torch.zeros((1,3,640,640)).to(DEVICE),torch.tensor([640,640]).to(DEVICE))
   return model
model = load_warm()

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})


@app.route('/predict',methods=['POST'])
def predict():
    bbx_color = request.args.get('bbx_color',(0,0,255))
    label_color = request.args.get('label_color',(255,0,0))
    scores = request.args.get('scores',None)
    thc =request.args.get('th',10)
    fil = request.files.get('file',None)
    if fil is not None:
       img = Image.open(fil)
       img = np.array(img)
       im = load_preprocess_img(img)
       im =map(lambda x:x.to(DEVICE),im)
       pred = model(*im)[0]
       coords,classes= to_numpy(YoloOutputs.from_yolo(pred),scores)
       del pred
       y = infer_img(img,coords[:,:2],coords[:,2:],classes,thc=thc,label_color=label_color,bbox_color=bbx_color)
       y = convert_string(y)
       return jsonify({'body':str(y),'status':200})
    else:
        return jsonify({'msg':'nothing was provided','status':400})   


if __name__=='__main__':
    app.run(debug=True)



      
