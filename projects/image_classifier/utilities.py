import numpy as np
import torch
from PIL import Image

def process_image(image):
    
    image.thumbnail((256,256))
    
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    width, height = image.size   # Get dimensions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = ((np_image / 255) - mean) / std
    
    np_image = np.moveaxis(np_image, -1, 0)
    torch_image = torch.from_numpy(np_image)
    
    return torch_image

def predict(image_path, model, device, topk=5):
    
    image = Image.open(image_path)
    image = process_image(image)
    image.unsqueeze_(0)
    image = image.float()
    image = image.to(device)
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        logps = model(image)

    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    top_p = top_p.cpu().tolist()[0]
    top_class = top_class.cpu().tolist()[0]
    top_class = [idx_to_class[k] for k in top_class]
    
    return top_p, top_class