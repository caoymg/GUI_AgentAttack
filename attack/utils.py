import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# Visual preprocessor
vis_processors_raw = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

def preprocess_image(image_path, device):
    image_rgb = Image.open(image_path).convert('RGB')
    image_tensor = vis_processors_raw(image_rgb).unsqueeze(0).to(device)
    return image_tensor

def save_tensor_as_image(tensor, path):
    image_rgb = (tensor.clone().squeeze().detach().cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
    pil_image = Image.fromarray(image_rgb)
    pil_image.save(path)
    return pil_image

def get_normalizer(mean, std):
    return transforms.Normalize(mean, std)
