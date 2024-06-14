import math
import random
import sys
import os
import torch
import tqdm
import argparse
import torch.nn
import numpy as np
import re
import wandb

from PIL import Image
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel
from diffusers import UNet2DConditionModel
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline, LDMPipeline
from accelerate import Accelerator
sys.path.insert(0, '%s'%os.path.join(os.path.dirname(__file__), '../src/'))
from markup2im_constants import get_image_size, get_input_field, get_encoder_model_type, get_color_mode
from markup2im_models import create_image_decoder, encode_text


#TODO: make it neater
def process_args(args):
    parser = argparse.ArgumentParser(description="Compare the latent forms of two sets of images")
#     parser.add_argument('--dataset_name',
#                         type=str, default='yuntian-deng/im2latex-100k',
#                         help=('Specifies which dataset to use.'
#                         ))
    parser.add_argument('--image_dir',
                        type=str, required=True,
                        help=('Image directory.'
                        ))
    parser.add_argument('--color_mode',
                        type=str, default=None,
                        help=('Specifies grayscale (grayscale) or RGB (rgb). If set to None, will be inferred according to dataset_name.'
                        ))
    parser.add_argument('--dataset_name',
                        type=str, default='yuntian-deng/im2smiles-20k',
                        help=('Specifies which dataset to use.'
                        ))
    
    parser.add_argument('--encoder_model_type',
                        type=str, default=None,
                        help=('Specifies encoder model type. If set to None, will be inferred according to dataset_name.'
                        ))
    parser.add_argument('--image_height',
                        type=int, default=None,
                        help=('Specifies the height of images to generate. If set to None, will be inferred according to dataset_name.'
                        ))
    parameters = parser.parse_args(args)
    return parameters


def load_and_tokenize_text(encoder_model_type, smiles_formula):
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_type)
    text_encoder = AutoModel.from_pretrained(encoder_model_type).cuda()
    tokenized_formula = tokenizer(smiles_formula, truncation=True, max_length=1024)
    input_ids = torch.LongTensor(tokenized_formula['input_ids']).unsqueeze(0).cuda()
    attention_mask = torch.FloatTensor(tokenized_formula['attention_mask']).unsqueeze(0).cuda()
    encoded_text = encode_text(text_encoder, input_ids, attention_mask)
    return encoded_text, attention_mask

def process_image(image_path, transform):
    image = Image.open(image_path)
    image_tensor = transform(image)
    return image_tensor

def encode_images(image_decoder, image_tensor, encoded_text, attention_mask):
    latent = image_decoder.encode(
        sample=image_tensor.cuda(),
        encoder_hidden_states=encoded_text.squeeze(2).cuda(),
        attention_mask=attention_mask.cuda()
    )
    return latent

def calculate_kl_divergence(gold_latent, pred_latent):

    # Compute log softmax over the last dimension
    gold_latent = torch.nn.functional.log_softmax(gold_latent, dim=-1)
    pred_latent = torch.nn.functional.log_softmax(pred_latent, dim=-1)
    
    # Calculate KL divergence with batchmean reduction
    kl_loss = torch.nn.functional.kl_div(pred_latent, gold_latent, reduction='batchmean', log_target=True)

    return kl_loss


def log_results(idx, file_name, kl_loss, gold_image, pred_image, gold_latent, pred_latent):
    wandb.log({
        "index": idx,
        "path": file_name,
        "kl-div": kl_loss.item(),
        "true-image": wandb.Image(gold_image, caption=f"{file_name} actual"),
        "true-latent": gold_latent.cpu().detach().numpy(),
        "predicted-image": wandb.Image(pred_image, caption=f"{file_name} predicted"),
        "predicted-latent": pred_latent.cpu().detach().numpy()
    })

def generate_latent(dataset_name, image_dir, image_size, color_channels, encoder_model_type):
    gold_image_dir = os.path.join(image_dir, "images_gold")
    pred_image_dir = os.path.join(image_dir, "images_pred")
    gold_files = sorted(os.listdir(gold_image_dir))
    pred_files = sorted(os.listdir(pred_image_dir))
    matched_files = zip(gold_files, pred_files)
    
    text_encoder = AutoModel.from_pretrained(encoder_model_type).cuda()
    hidden_states = encode_text(text_encoder, torch.zeros(1, 1).long().cuda(), None)
    cross_attention_dim = hidden_states.shape[-1]
    image_decoder = create_image_decoder(image_size=image_size, color_channels=color_channels, cross_attention_dim=cross_attention_dim)
    image_decoder = image_decoder.cuda()
    
    dataset = load_dataset(dataset_name, split="test")
    FILE_PATH_TO_SMILES = {}
    iter = dataset.iter(batch_size=1)
    for i in iter:
        path = i["filename"][0]
        smile = i["smiles"][0]
        FILE_PATH_TO_SMILES[path] = smile
    
    transform = transforms.Compose([transforms.Lambda(lambda img: torch.from_numpy(np.array(img).astype(np.float32)).permute(2, 0, 1).unsqueeze(0))])
    
    idx = 0
    for gold_file, pred_file in matched_files:
        gold_file_path = os.path.join(gold_image_dir, gold_file)
        pred_file_path = os.path.join(pred_image_dir, pred_file)
        file_name = re.search(r"\/([^\/]*)$", gold_file_path).group(1)
        smiles_formula = FILE_PATH_TO_SMILES[file_name]
        
        encoded_text, attention_mask = load_and_tokenize_text(encoder_model_type, smiles_formula)
        
        gold_tensor = process_image(gold_file_path, transform)
        pred_tensor = process_image(pred_file_path, transform)
        
        gold_latent = encode_images(image_decoder, gold_tensor, encoded_text, attention_mask)
        pred_latent = encode_images(image_decoder, pred_tensor, encoded_text, attention_mask)
        
        kl_loss = calculate_kl_divergence(gold_latent, pred_latent)
        
        log_results(idx, file_name, kl_loss, gold_file_path, pred_file_path, gold_latent, pred_latent)
        idx += 1
        


def main(args):

    image_dir = args.image_dir
    dataset_name = args.dataset_name

    if (args.image_height is not None) and (args.image_width is not None):
        image_size = (args.image_height, args.image_width)
    else:
        print (f'Using default image size for dataset {dataset_name}')
        image_size = get_image_size(dataset_name)
        print (f'Default image size: {image_size}')
    if args.color_mode is not None:
        color_mode = args.color_mode
    else:
        print (f'Using default color mode for dataset {dataset_name}')
        color_mode = get_color_mode(dataset_name)
        print (f'Default color mode: {color_mode}')
    args.color_mode = color_mode 
    assert args.color_mode in ['grayscale', 'rgb']
    if args.color_mode == 'grayscale':
        args.color_channels = 1
    else:
        args.color_channels = 3
    color_channels = args.color_channels
    if args.encoder_model_type is not None:
        encoder_model_type = args.encoder_model_type
    else:
        print (f'Using default encoder model type for dataset {dataset_name}')
        encoder_model_type = get_encoder_model_type(dataset_name)
        print (f'Default encoder model type: {encoder_model_type}')
    wandb.init(
        project="odys-latent",
        config={
            "dataset":"im2smiles-20k"
        }
    )
    generate_latent(dataset_name, image_dir, image_size, color_channels, encoder_model_type)
    wandb.finish()
    



if __name__ == '__main__':
    args = process_args(sys.argv[1:])
    main(args)
