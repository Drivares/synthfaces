import numpy as np
import torch
import PIL.Image
from typing import Tuple

#----------------------------------------------------------------------------

def classes2label(classes_idx, num_classes):
    label = np.zeros(np.sum(num_classes), dtype=np.float32)
    p = 0
    for class_idx, class_dim in zip(classes_idx, num_classes):
        label[p + class_idx] = 1
        p += class_dim
    return label

#----------------------------------------------------------------------------

def gen_image_from_z(G, z, c, truncation_psi=1, noise_mode='const', translate=(0,0), rotate=0):
    ws = G.mapping(z, c, truncation_psi)
    return gen_image_from_ws(G, ws, noise_mode, translate, rotate)

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def gen_image_from_ws(G, ws, noise_mode='const', translate=(0,0), rotate=0):
    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    return G.synthesis(ws, noise_mode=noise_mode)

#----------------------------------------------------------------------------

def calc_latent_edit(G, ws, source_c, target_c, alpha=None):
    source_idx = torch.sum(torch.mul(source_c, G.mapping.c2idx), dim=1, dtype=torch.long)
    target_idx = torch.sum(torch.mul(target_c, G.mapping.c2idx), dim=1, dtype=torch.long)
    latent_direction = G.mapping.w_class_avg[target_idx] - G.mapping.w_class_avg[source_idx]
    if alpha is None:
        edited_ws = ws + latent_direction
    else:
        edited_ws = ws + alpha * torch.nn.functional.normalize(latent_direction, dim=1)
    return edited_ws

#----------------------------------------------------------------------------

def save_image(img, path):
    channels = img.shape[2] if img.ndim == 3 else 1
    img = PIL.Image.fromarray(img, {1: 'L', 2: 'LA', 3: 'RGB', 4: 'RGBA' }[channels])
    img.save(path)