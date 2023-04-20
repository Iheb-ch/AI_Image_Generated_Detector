import cv2
from utils import read_image,get_valid_augs
import torch
import torch.nn.functional as F
from model import CustomModel

CKPT = 'fold_0.pt'
Targets = ['Not AI' 'AI Generated']
def predict_one_image(path) :
    image = read_image(path)
    image = get_valid_augs()(image=image)['image']
    image = torch.tensor(image,dtype=torch.float)
    image = image.reshape((1,3,224,224))
    model = CustomModel()
    #loading ckpt
    model.load_state_dict(torch.load(CKPT,map_location=torch.device('cpu')))
    with torch.no_grad() :
        outputs = model(image)
        proba = F.sigmoid(outputs['label']).detach().numpy()[0]
    return {'Not AI' : 1-float(proba),'AI' : float(proba)}#(proba>0.5)*1

