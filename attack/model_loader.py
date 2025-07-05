import open_clip

def load_eva_clip(clip_type, device):
    if clip_type == "e14":
        model, _, _ = open_clip.create_model_and_transforms(
            'EVA02-E-14', pretrained='laion2b_s4b_b115k')
        tokenizer = open_clip.get_tokenizer('EVA02-E-14')
    else:
        model, _, _ = open_clip.create_model_and_transforms(
            'EVA02-E-14-plus', pretrained='laion2b_s9b_b144k')
        tokenizer = open_clip.get_tokenizer('EVA02-E-14-plus')

    model.eval().to(device)
    return model, tokenizer
