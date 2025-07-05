import torch
from config import *
from attack.utils import preprocess_image, save_tensor_as_image, get_normalizer
from attack.model_loader import load_eva_clip
from attack.attack import clip_targeted_attack

if __name__ == '__main__':
    # Load model and tokenizer
    model, tokenizer = load_eva_clip(CLIP_TYPE, DEVICE)

    # Load and preprocess image
    image_tensor = preprocess_image(IMAGE_PATH, DEVICE)
    ref_image_tensor = preprocess_image(REF_IMAGE_PATH, DEVICE)

    # Normalize transform
    normalize = get_normalizer(MEAN, STD)

    # Run adversarial attack
    clip_targeted_attack(
        model, tokenizer, image_tensor, ref_image_tensor,
        TARGET_TEXT, CLEAN_TEXT,
        normalize, STEP_SIZE, EPS, STEPS,
        DEVICE, CLIP_TYPE
    )
