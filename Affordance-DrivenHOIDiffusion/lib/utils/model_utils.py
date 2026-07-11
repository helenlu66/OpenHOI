import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.networks.cvae import SeqCVAE, CTCVAE
from lib.networks.texthom import TextHOM
from lib.networks.refiner import Refiner
from lib.networks.diffusion import Diffusion
from lib.networks.pointnet import PointNetfeat
from lib.device_utils import get_inference_device

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class _TransformersMpnet:
    """MPNet text encoder fallback when sentence-transformers is unavailable."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def encode(self, sentences, convert_to_tensor=True):
        if isinstance(sentences, str):
            sentences = [sentences]
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = F.normalize(summed / counts, p=2, dim=1)
        if convert_to_tensor:
            return embeddings
        return embeddings.cpu().numpy()
def _to_device(module):
    return module.to(get_inference_device())

def init_weights_to_zero(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def build_mpnet(args):
    print(f"build mpnet {args.mpnet.version}")
    if SentenceTransformer is not None:
        mpnet = SentenceTransformer(args.mpnet.version)
    else:
        print("sentence-transformers not found; using transformers MPNet fallback")
        mpnet = _TransformersMpnet(args.mpnet.version)
    mpnet = _to_device(mpnet)
    return mpnet
def build_refiner(args, test=False):
    args_refiner = args.refiner
    refiner = Refiner(**args_refiner)
    refiner = _to_device(refiner)
    print("build refiner")

    if args_refiner.zero_initialized:
        print("initialize refiner with zero")
        refiner.apply(init_weights_to_zero)

    if test:
        weight_path = args_refiner.weight_path
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
        print(f"load refiner, {weight_path}")
        checkpoints = torch.load(weight_path, map_location=get_inference_device())
        refiner.load_state_dict(checkpoints["model"])
        refiner.eval()
        for p in refiner.parameters():
            p.requires_grad = False
    return refiner

def build_model_and_diffusion(args, lhand_layer, rhand_layer, test=False):
    args_texthom = args.texthom
    args_diffusion = args.diffusion
    texthom = TextHOM(**args_texthom)
    diffusion = Diffusion(
        lhand_layer=lhand_layer, 
        rhand_layer=rhand_layer, 
        **args_diffusion
    )
    texthom = _to_device(texthom)
    diffusion = _to_device(diffusion)
    print("build texthom, diffusion")
    
    if test:
        weight_path = args_texthom.weight_path
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
        print(f"load texthom, {weight_path}")
        checkpoints = torch.load(weight_path, map_location=get_inference_device())
        texthom.load_state_dict(checkpoints["model"])
        texthom.eval()
        diffusion.eval()
        for p in texthom.parameters():
            p.requires_grad = False
    return texthom, diffusion

def build_seq_cvae(args, test=False):
    args_cvae = args.seq_cvae
    seq_cvae = SeqCVAE(**args_cvae)
    seq_cvae = _to_device(seq_cvae)
    print("build seq cvae")
    
    if test:
        weight_path = args_cvae.weight_path
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"    
        print(f"load seq cvae, {weight_path}")
        checkpoints = torch.load(weight_path, map_location=get_inference_device())
        seq_cvae.load_state_dict(checkpoints["model"])
        seq_cvae.eval()
        for p in seq_cvae.parameters():
            p.requires_grad = False
    return seq_cvae

def build_pointnetfeat(args, test=False):
    args_pointfeat = args.pointfeat
    point_encoder = PointNetfeat(**args_pointfeat)
    point_encoder = _to_device(point_encoder)
    print("build point encoder")
    
    if test:
        weight_path = args_pointfeat.weight_path
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
        print(f"load point encoder, {weight_path}")
        checkpoints = torch.load(weight_path, map_location=get_inference_device())
        point_encoder.load_state_dict(checkpoints["model"])
        point_encoder.eval()
        for p in point_encoder.parameters():
            p.requires_grad = False
    return point_encoder

def build_contact_estimator(args, test=False):
    args_contact = args.contact
    #print(1,args_contact)
    contact_estimator = CTCVAE(**args_contact)
    contact_estimator = _to_device(contact_estimator)
    print("build contact estimator")
    
    if test:
        weight_path = args_contact.weight_path
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
        print(f"load contact estimator, {weight_path}")
        checkpoints = torch.load(weight_path, map_location=get_inference_device())
        contact_estimator.load_state_dict(checkpoints["model"])
        contact_estimator.eval()
        for p in contact_estimator.parameters():
            p.requires_grad = False


    return contact_estimator