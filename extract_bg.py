"""Remove background of images"""

import os
import sys

import click
import numpy as np
import PIL.Image
import torch

from matting.remove_background import BackgroundRemoval, GuidanceCalculator

#----------------------------------------------------------------------------

def error(msg):
    print(f'Error: {msg}')
    sys.exit(1)

#----------------------------------------------------------------------------

def extract_bg_img(input_img, outdir_img, model_name, background_removal, guidance_calculator):
    img = PIL.Image.open(input_img)
    img = np.array(img)
    img = background_removal(img, guidance_calculator(model_name, img))
    img = PIL.Image.fromarray(img, 'RGBA')
    img.save(outdir_img)

#----------------------------------------------------------------------------

def extract_bg_folder(input_folder, outdir, model_name, background_removal, guidance_calculator):
    for root, _, filenames in os.walk(input_folder, topdown=True):
        cur_outdir = outdir
        if root != input_folder:
            cur_outdir = os.path.join(cur_outdir, os.path.relpath(root, input_folder))
            os.makedirs(cur_outdir)
        for filename in filenames:
            input_img = os.path.join(os.path.join(input_folder, os.path.relpath(root, input_folder)), filename)
            outdir_img =  os.path.join(cur_outdir, filename)
            if os.path.splitext(filename)[1] in PIL.Image.EXTENSION:
                extract_bg_img(input_img, outdir_img, model_name, background_removal, guidance_calculator)

#----------------------------------------------------------------------------

@click.command()
@click.option('--input-path', help='path of input image or folder containing images', type=str, required=True, metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--model-name', help='Matting model name', type=click.Choice(['SGHM', 'MODNet', 'P3M-ViTAE', 'P3M', 'AIM', 'GFM', 'InstMatt', 'MGMatting']), required=True)
def extract_bg(
    input_path: str,
    outdir: str,
    model_name: str
):
    """Remove background of images"""

    PIL.Image.init() # type: ignore

    os.makedirs(outdir)

    device = torch.device('cuda')
    background_removal = BackgroundRemoval('SGHM', device=device)
    guidance_calculator = GuidanceCalculator(device=device)

    if os.path.isfile(input_path) and os.path.splitext(input_path)[1] in PIL.Image.EXTENSION:
        outdir_img = os.path.join(outdir, os.path.basename(input_path))
        extract_bg_img(input_path, outdir_img, model_name, background_removal, guidance_calculator)
    elif os.path.isdir(input_path):
        extract_bg_folder(input_path, outdir, model_name, background_removal, guidance_calculator)
    else:
        error(f'{input_path} is not an image or folder')
    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    extract_bg() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
