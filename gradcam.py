import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import pandas as pd
import os
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RemoveBottomPixels(object):
    def __init__(self, pixels_to_remove=20):
        self.pixels_to_remove = pixels_to_remove

    def __call__(self, img):
        width, height = img.size
        return img.crop((0, 0, width, height - self.pixels_to_remove))

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = RemoveBottomPixels(pixels_to_remove=20)(img) 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0)
    return img_t.to(DEVICE)

def resize_image_for_cam(img_path, target_size):
    img = Image.open(img_path).convert("RGB")
    img = RemoveBottomPixels(pixels_to_remove=20)(img)
    img = img.resize(target_size)
    return np.array(img) / 255.0

if __name__ == "__main__":
    model_path = "data/model/resnet50_aircraft_17.pt"
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()
    target_layer = model.stage4[-1]

    # 가장 많이 맞춘 클래스 식별
    most_correct_class = 85
    
    train_csv = "./csv/train.csv"
    df = pd.read_csv(train_csv)
    
    # 가장 잘 맞춘 클래스를 찾은 뒤에 [첫번째 행, 첫번째 열(이미지이름)]을 찾음
    target_img_path = df[df.iloc[:, 2] == most_correct_class].iloc[0, 0]
    image_path = os.path.join("./data/images", target_img_path)
    
    img_tensor = preprocess_image(image_path)
    rgb_img = resize_image_for_cam(image_path, (224, 224))

    cam_methods = {
        "GradCAM": GradCAM,
        "HiResCAM": HiResCAM,
        "ScoreCAM": ScoreCAM,
        "GradCAM++": GradCAMPlusPlus,
        "AblationCAM": AblationCAM,
        "XGradCAM": XGradCAM,
        "EigenCAM": EigenCAM,
        "FullGrad": FullGrad
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (name, cam_cls) in enumerate(cam_methods.items()):
        cam = cam_cls(model=model, target_layers=[target_layer])
        
        # 가장 많이 맞춘 클래스를 타겟으로 설정
        targets = [ClassifierOutputTarget(most_correct_class)]
        
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
        grayscale_cam_resized = cv2.resize(grayscale_cam, (224, 224))
        visualization = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True)
        
        axes[idx].imshow(visualization)
        axes[idx].set_title(name)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('results/heatmap/all_cams.jpg')
    plt.show()
    print("All CAM visualizations saved to 'results/all_cams.jpg'")

# cutmix 사용된 이미지 볼때만 주석해제

# import matplotlib.pyplot as plt
# import torch
# from torchvision import models, transforms
# from torchvision.transforms import Compose, Normalize, ToTensor
# from PIL import Image
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import numpy as np
# import pandas as pd
# import os
# import cv2
# import random

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class RemoveBottomPixels(object):
#     def __init__(self, pixels_to_remove=20):
#         self.pixels_to_remove = pixels_to_remove

#     def __call__(self, img):
#         width, height = img.size
#         return img.crop((0, 0, width, height - self.pixels_to_remove))

# def preprocess_image(img_path):
#     img = Image.open(img_path).convert("RGB")
#     img = RemoveBottomPixels(pixels_to_remove=20)(img) 
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     img_t = transform(img).unsqueeze(0)
#     return img_t.to(DEVICE)

# def resize_image_for_cam(img_path, target_size):
#     img = Image.open(img_path).convert("RGB")
#     img = RemoveBottomPixels(pixels_to_remove=20)(img)
#     img = img.resize(target_size)
#     return np.array(img) / 255.0

# def cutmix_image(image1, image2, alpha=1.0):
#     lam = np.random.beta(alpha, alpha)
#     bbx1, bby1, bbx2, bby2 = rand_bbox(image1.shape, lam)
#     image1[bbx1:bbx2, bby1:bby2, :] = image2[bbx1:bbx2, bby1:bby2, :]
#     return image1

# def rand_bbox(size, lam):
#     W = size[1]
#     H = size[0]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = int(W * cut_rat)
#     cut_h = int(H * cut_rat)

#     cx = np.random.randint(W)
#     cy = np.random.randint(H)

#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)

#     return bbx1, bby1, bbx2, bby2

# if __name__ == "__main__":
#     model_path = "data/model/resnet50_aircraft_17.pt"
#     model = torch.load(model_path, map_location=DEVICE)
#     model.eval()
#     target_layer = model.stage4[-1]

#     # 가장 많이 맞춘 클래스 식별
#     most_correct_class = 92
    
#     train_csv = "./csv/train.csv"
#     df = pd.read_csv(train_csv)
    
#     # 가장 잘 맞춘 클래스를 찾은 뒤에 [첫번째 행, 첫번째 열(이미지이름)]을 찾음
#     target_img_path1 = df[df.iloc[:, 2] == most_correct_class].iloc[0, 0]
#     image_path1 = os.path.join("./data/images", target_img_path1)

#     # CutMix용 두 번째 이미지 선택
#     target_img_path2 = df[df.iloc[:, 2] != most_correct_class].sample(n=1).iloc[0, 0]
#     image_path2 = os.path.join("./data/images", target_img_path2)

#     img1 = preprocess_image(image_path1)
#     img2 = preprocess_image(image_path2)
    
#     rgb_img1 = resize_image_for_cam(image_path1, (224, 224))
#     rgb_img2 = resize_image_for_cam(image_path2, (224, 224))
    
#     # CutMix 적용
#     cutmix_img = cutmix_image(rgb_img1, rgb_img2)
#     cutmix_img_pil = Image.fromarray((cutmix_img * 255).astype(np.uint8))
#     cutmix_img_tensor = transforms.ToTensor()(cutmix_img_pil).unsqueeze(0).to(DEVICE)
    
#     cam_methods = {
#         "GradCAM": GradCAM,
#         "HiResCAM": HiResCAM,
#         "ScoreCAM": ScoreCAM,
#         "GradCAM++": GradCAMPlusPlus,
#         "AblationCAM": AblationCAM,
#         "XGradCAM": XGradCAM,
#         "EigenCAM": EigenCAM,
#         "FullGrad": FullGrad
#     }

#     fig, axes = plt.subplots(2, 4, figsize=(20, 10))
#     axes = axes.flatten()

#     for idx, (name, cam_cls) in enumerate(cam_methods.items()):
#         cam = cam_cls(model=model, target_layers=[target_layer])
        
#         # 가장 많이 맞춘 클래스를 타겟으로 설정
#         targets = [ClassifierOutputTarget(most_correct_class)]
        
#         grayscale_cam = cam(input_tensor=cutmix_img_tensor, targets=targets)[0, :]
#         grayscale_cam_resized = cv2.resize(grayscale_cam, (224, 224))
#         visualization = show_cam_on_image(cutmix_img, grayscale_cam_resized, use_rgb=True)
        
#         axes[idx].imshow(visualization)
#         axes[idx].set_title(name)
#         axes[idx].axis('off')

#     plt.tight_layout()
#     plt.savefig('results/heatmap/all_cams_cutmix.jpg')
#     plt.show()
#     print("All CAM visualizations for CutMix saved to 'results/all_cams_cutmix.jpg'")