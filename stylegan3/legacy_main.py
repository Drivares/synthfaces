# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Converting legacy network pickle into the new format."""

import click
import pickle

from core import dnnlib
from core import legacy

#----------------------------------------------------------------------------

@click.command()
@click.option('--source', help='Input pickle', required=True, metavar='PATH')
@click.option('--dest', help='Output pickle', required=True, metavar='PATH')
@click.option('--force-fp16', help='Force the networks to use FP16', type=bool, default=False, metavar='BOOL', show_default=True)
def convert_network_pickle(source, dest, force_fp16):
    """Convert legacy network pickle into the native PyTorch format.

    The tool is able to load the main network configurations exported using the TensorFlow version of StyleGAN2 or StyleGAN2-ADA.
    It does not support e.g. StyleGAN2-ADA comparison methods, StyleGAN2 configs A-D, or StyleGAN1 networks.

    Example:

    \b
    python legacy.py \\
        --source=https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl \\
        --dest=stylegan2-cat-config-f.pkl
    """
    print(f'Loading "{source}"...')
    with dnnlib.util.open_url(source) as f:
        data = legacy.load_network_pkl(f, force_fp16=force_fp16)
    print(f'Saving "{dest}"...')
    with open(dest, 'wb') as f:
        pickle.dump(data, f)
    print('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_network_pickle() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
