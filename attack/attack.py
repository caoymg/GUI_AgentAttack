import torch
import torch.nn.functional as F
from attack.utils import save_tensor_as_image

def clip_targeted_attack(
    model, tokenizer, image_tensor_raw, ref_image_tensor, target_text, clean_text,
    normalize, step_size, eps, steps, device, clip_type
):
    delta = torch.zeros_like(image_tensor_raw).to(device)
    tar_text_input = tokenizer(target_text).to(device)
    clean_text_input = tokenizer(clean_text).to(device)

    for step in range(steps):
        delta.data = torch.clamp(
            delta,
            -image_tensor_raw,
            1 - image_tensor_raw
        )
        delta.requires_grad = True

        adv_images = image_tensor_raw + delta

        with torch.cuda.amp.autocast():
            image_features = model.encode_image(normalize(adv_images))
            ref_image_features = model.encode_image(normalize(ref_image_tensor))
            tar_features = model.encode_text(tar_text_input)
            clean_features = model.encode_text(clean_text_input)

        image_features = F.normalize(image_features, dim=-1)
        ref_image_features = F.normalize(ref_image_features, dim=-1)
        tar_features = F.normalize(tar_features, dim=-1)
        clean_features = F.normalize(clean_features, dim=-1)

        tar_sim_it = torch.matmul(image_features.unsqueeze(1), tar_features.unsqueeze(-1)).squeeze()
        tar_sim_ii = torch.matmul(image_features.unsqueeze(1), ref_image_features.unsqueeze(-1)).squeeze()
        clean_sim = torch.matmul(image_features.unsqueeze(1), clean_features.unsqueeze(-1)).squeeze()

        loss = -tar_sim_ii - tar_sim_it + clean_sim

        if step % 10 == 0:
            print(f"step:[{step}], clean_sim:[{clean_sim}], tar_sim_ii:[{tar_sim_ii}], ltar_sim_it:[{tar_sim_it}], oss:[{loss}]")

        loss.backward()

        grad = delta.grad.detach()
        delta = torch.clamp(delta - step_size * torch.sign(grad), min=-eps, max=eps).detach().requires_grad_(True)

    adv_image_tensor = image_tensor_raw + delta.detach()
    adv_image_tensor = torch.clamp(adv_image_tensor, 0, 1)

    adv_path = f"adv_{clip_type}.jpeg"
    save_tensor_as_image(adv_image_tensor, adv_path)
