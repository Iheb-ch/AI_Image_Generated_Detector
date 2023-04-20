import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def read_image(path) :
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def get_valid_augs() :
    return  A.Compose([
   A.Resize(height=224, width=224, always_apply=True, p=1),
   A.Normalize(
        mean = IMAGENET_DEFAULT_MEAN,
        std  = IMAGENET_DEFAULT_STD,
        max_pixel_value=255
            ),
   ToTensorV2(),
])