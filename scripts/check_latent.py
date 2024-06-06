import math
import random
import sys
import os
import torch
import tqdm
import argparse
import torch.nn
import numpy as np
import importlib

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

# def convert_latent
# def compare_latent():
#     """
#     Compares 2 images in their latent form and returns the KL divergence between them

#     :output: KL divergence between 2 images
#     """
def generate_latent(image_dir, image_size, color_channels, encoder_model_type):

    # if args.encoder_model_type is not None:
    #     encoder_model_type = args.encoder_model_type
    # else:
    #     print (f'Using default encoder model type for dataset {args.dataset_name}')
    #     encoder_model_type = get_encoder_model_type(args.dataset_name)
    #     print (f'Default encoder model type: {encoder_model_type}')
    # args.encoder_model_type = encoder_model_type

    gold_image_dir = os.path.join(image_dir, "images_gold")
    pred_image_dir = os.path.join(image_dir, "images_pred")
    gold_files = sorted(os.listdir(gold_image_dir))
    pred_files = sorted(os.listdir(pred_image_dir))
    matched_files = zip(gold_files, pred_files)

    text_encoder = AutoModel.from_pretrained(encoder_model_type).cuda()
    # forward a fake batch to figure out cross_attention_dim
    hidden_states = encode_text(text_encoder, torch.zeros(1,1).long().cuda(), None)
    cross_attention_dim = hidden_states.shape[-1]

    image_decoder = create_image_decoder(image_size=image_size, color_channels=color_channels, \
            cross_attention_dim=cross_attention_dim)
    image_decoder = image_decoder.cuda()
   
    # Iterate over matched files
    for gold_file, pred_file in matched_files:
        # Construct full paths
        gold_file_path = os.path.join(gold_image_dir, gold_file)
        pred_file_path = os.path.join(pred_image_dir, pred_file)
        print(dir(image_decoder))  # This will print all attributes and methods of the instance
        if 'encode' in dir(image_decoder):
            print("Encode method is available")
        else:
            print("Encode method is not available, check the class definition and imports.")
        print(image_decoder.encode(gold_file_path))
        
       
    # Add error handling if the number of files or file names do not match
    if len(gold_files) != len(pred_files):
        print("Warning: The number of files in gold and pred directories do not match.")


def main(args):

    image_dir = args.image_dir

    if (args.image_height is not None) and (args.image_width is not None):
        image_size = (args.image_height, args.image_width)
    else:
        print (f'Using default image size for dataset {args.dataset_name}')
        image_size = get_image_size(args.dataset_name)
        print (f'Default image size: {image_size}')
    if args.color_mode is not None:
        color_mode = args.color_mode
    else:
        print (f'Using default color mode for dataset {args.dataset_name}')
        color_mode = get_color_mode(args.dataset_name)
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
        print (f'Using default encoder model type for dataset {args.dataset_name}')
        encoder_model_type = get_encoder_model_type(args.dataset_name)
        print (f'Default encoder model type: {encoder_model_type}')
    generate_latent(image_dir, image_size, color_channels, encoder_model_type)
    



if __name__ == '__main__':
    args = process_args(sys.argv[1:])
    main(args)