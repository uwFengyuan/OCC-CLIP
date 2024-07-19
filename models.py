import torch
import torch.nn as nn
from promptlearner import TextEncoder, PromptLearner_CoOp
from clip import clip
import torch.nn.functional as F
import torchvision.transforms as tt
import numpy as np
import math

def CoOp(opt):
    if opt.backbone == 'RN50':
        backbone_name = 'RN50' #cfg.MODEL.BACKBONE.NAME
    elif opt.backbone == 'ViT':
        backbone_name = "ViT-B/16"

    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    clip_model = clip.build_model(state_dict or model.state_dict())


    print("Building custom CLIP")
    class_names = [opt.neg_name] + [opt.pos_name]
    # assert len(class_names) == opt.n_splitNG + opt.n_splitG, f"the number of classes is {opt.n_splitNG + opt.n_splitG}"
    
    if opt.model == 'CoOp':
        model = CustomCLIP_CoOp(opt, class_names, clip_model)


    transform = tt.Compose([tt.Resize((224, 224), interpolation=tt.InterpolationMode.BICUBIC),
            tt.CenterCrop((224, 224)),
            lambda image: image.convert("RGB"),
            tt.ToTensor(),
            tt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

    return model, transform


class CustomCLIP_CoOp(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_CoOp(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual # image encoder, ResNet
        self.text_encoder = TextEncoder(clip_model) # text encoder
        self.logit_scale = clip_model.logit_scale # tensor(4.6052)
        self.dtype = clip_model.dtype # torch.float32
        self.cfg = cfg

    def forward(self, image):
        if self.cfg.backbone == "ViT":
            image_features = self.image_encoder(image.type(self.dtype))[:, 0, :]
        else:
            image_features = self.image_encoder(image.type(self.dtype))
        # print(image_features.shape)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # print(image_features.shape)

        # image_feature + caption text??

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # print(text_features.shape)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        # if self.cfg.exp:
        #     torch.set_printoptions(precision=8)
        #     print(image_features[0][:5])

        return logits, image_features, text_features