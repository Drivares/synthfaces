"""Simple GUI for generating synthetic faces."""

import os
import sys
import io
import gc
import click
import gdown
import torch
import numpy as np
import PySimpleGUI as sg
from PIL import Image
from easydict import EasyDict

from stylegan3.core import dnnlib
from stylegan3.core import legacy
from stylegan3.core.gen_utils import classes2label, gen_image_from_z, gen_image_from_ws, calc_latent_edit
from matting.remove_background import BackgroundRemoval, GuidanceCalculator

#----------------------------------------------------------------------------

# CONFIG
NETWORK_PKL = 'stylegan3/pretrained/stylegan2_ffhq_aging_256.pkl'
NETWORK_URL = 'https://drive.google.com/uc?id=1t7K1uSFt23uPQhBd84LgBzrcLwFr8SeY'
DEVICE = torch.device('cuda')
sg.theme('LightBlue3')

#----------------------------------------------------------------------------

def get_classes(values):
    gender_idx, age_idx = None, None
    gender2idx = {'male': 0, 'female': 1}
    age2idx = {'0-2': 0, '3-6': 1, '7-9': 2, '10-14': 3, '15-19': 4, '20-29': 5, '30-39': 6, '40-49': 7, '50-69': 8, '+70': 9}
    for gender, idx in gender2idx.items():
        if values[gender]:
            gender_idx = idx
            break
    for age, idx in age2idx.items():
        if values[age]:
            age_idx = idx
            break
    return gender_idx, age_idx

#----------------------------------------------------------------------------

def get_editing_mode(values):
    available_modes = ['condition', 'latent']
    for mode in available_modes:
        if values[mode]:
            return mode
    return None

#----------------------------------------------------------------------------

def get_matting_model(values):
    available_models = ['SGHM', 'MODNet', 'P3M-ViTAE', 'P3M', 'AIM', 'GFM', 'InstMatt', 'MGMatting']
    for model in available_models:
        if values[model]:
            return model
    return None

#----------------------------------------------------------------------------

def generated2numpy(img):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img[0].cpu().numpy()
    return img

#----------------------------------------------------------------------------

def generated2PIL(img):
    img = generated2numpy(img)
    img = Image.fromarray(img, 'RGB')
    return img

#----------------------------------------------------------------------------

def update_img(window, current_state):
    resolution = int(window['-RESOLUTION-'].get())
    bio = io.BytesIO()
    current_state.img.resize((resolution, resolution)).save(bio, format='PNG')
    window['-IMAGE-'].update(data=bio.getvalue())

#----------------------------------------------------------------------------

def update_truncation(window, increment):
    truncation_psi = float(window['-TRUNCATION_PSI-'].get())
    truncation_psi = round(truncation_psi + increment, 1)
    truncation_psi = max(truncation_psi, 0.0)
    truncation_psi = min(truncation_psi, 1.0)
    window['-TRUNCATION_PSI-'].update(f'{truncation_psi:.1f}')
    return False

#----------------------------------------------------------------------------

def update_resolution(window, increment):
    resolution = int(window['-RESOLUTION-'].get())
    resolution = resolution + increment
    resolution = max(resolution, 256)
    resolution = min(resolution, 1024)
    window['-RESOLUTION-'].update(str(resolution))
    return True

#----------------------------------------------------------------------------

def gen_img(G, window, current_state, values, full):
    current_state.source_classes_idx = get_classes(values)
    current_state.z = torch.from_numpy(np.random.RandomState().randn(1, G.z_dim)).to(DEVICE)
    c = torch.tensor(classes2label(current_state.source_classes_idx, G.mapping.num_classes), device=DEVICE).unsqueeze(0)
    if full:
        current_state.truncation_psi = float(window['-TRUNCATION_PSI-'].get())
    current_state.ws = G.mapping(current_state.z, c, truncation_psi=current_state.truncation_psi)
    img = gen_image_from_ws(G, current_state.ws)
    current_state.img = generated2PIL(img)
    return True

#----------------------------------------------------------------------------

def edit_img(G, current_state, values, full):
    if current_state.z is None:
        return False
    target_classes_idx = get_classes(values)
    if full:
        current_state.editing_mode = get_editing_mode(values)
    target_c = torch.tensor(classes2label(target_classes_idx, G.mapping.num_classes), device=DEVICE).unsqueeze(0)
    if current_state.editing_mode == 'condition':
        edited_ws = G.mapping(current_state.z, target_c, truncation_psi=current_state.truncation_psi)
    elif current_state.editing_mode == 'latent':
        source_c = torch.tensor(classes2label(current_state.source_classes_idx, G.mapping.num_classes), device=DEVICE).unsqueeze(0)
        source_ws = G.mapping(current_state.z, source_c, truncation_psi=current_state.truncation_psi)
        edited_ws = calc_latent_edit(G, source_ws, source_c, target_c)
    edited_img = gen_image_from_ws(G, edited_ws)
    current_state.ws = edited_ws
    current_state.img = generated2PIL(edited_img)
    return True

#----------------------------------------------------------------------------

def update_matting_model(current_state, values):
    matting_model = get_matting_model(values)
    if matting_model == None or matting_model == current_state.matting_model:
        return False
    current_state.matting_model = matting_model
    current_state.background_removal = BackgroundRemoval(matting_model, device=DEVICE)
    torch.cuda.empty_cache()
    gc.collect()
    return True

#----------------------------------------------------------------------------

def extract_bg(G, current_state, values, full):
    if current_state.z is None:
        return False
    if full:
        update_matting_model(current_state, values)
    img = gen_image_from_ws(G, current_state.ws)
    img = generated2numpy(img)
    img = current_state.background_removal(img, current_state.guidance_calculator(current_state.matting_model, img))
    current_state.img = Image.fromarray(img, 'RGBA')
    return True

#----------------------------------------------------------------------------

def replace_bg(G, current_state, values, full):
    if current_state.z is None:
        return False
    background_path = values['Reemplazar fondo']
    if os.path.splitext(background_path)[1] not in Image.EXTENSION:
        sg.popup_error(f'{background_path} no es una imagen')
        return False
    extract_bg(G, current_state, values, full)
    img, background = np.array(current_state.img), Image.open(background_path)
    if background.size != img.shape[:2]:
        background = background.resize(img.shape[:2])
    foreground, alpha, background = img[:,:,:3], img[:,:,3:] / 255.0, np.array(background)
    current_state.img = alpha * foreground + (1 - alpha) * background
    current_state.img = Image.fromarray(current_state.img.astype(np.uint8), 'RGB')
    return True

#----------------------------------------------------------------------------

def save_img(current_state, values):
    if current_state.z is None:
        return False
    current_state.img.save(values['Guardar imagen'])
    return False

#----------------------------------------------------------------------------

def init_window(current_state):

    buttons_column = [
        [
            sg.Stretch(),
            sg.Text('Género:'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Radio('Hombre', 'gender', key='male', default=True),
            sg.Radio('Mujer', 'gender', key='female'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Text('Edad:'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Column([
                [sg.Radio('0-2', 'age', key='0-2', default=True)],
                [sg.Radio('7-9', 'age', key='7-9')],
                [sg.Radio('15-19', 'age', key='15-19')],
                [sg.Radio('30-39', 'age', key='30-39')],
                [sg.Radio('50-69', 'age', key='50-69')],
            ]),
            sg.Column([
                [sg.Radio('3-6', 'age', key='3-6')],
                [sg.Radio('10-14', 'age', key='10-14')],
                [sg.Radio('20-29', 'age', key='20-29')],
                [sg.Radio('40-49', 'age', key='40-49')],
                [sg.Radio('+70', 'age', key='+70')],
            ]),
            sg.Stretch(),
        ],
        [
            sg.HorizontalSeparator(),
        ],
        [
            sg.Stretch(),
            sg.Button('Generar imagen'),
            sg.Stretch(),
        ],
        [
            sg.HorizontalSeparator(),
        ],
        [
            sg.Stretch(),
            sg.Button('Editar imagen'),
            sg.Stretch(),
        ],
        [
            sg.HorizontalSeparator(),
        ],
        [
            sg.Stretch(),
            sg.Button('Extraer fondo'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.FileBrowse('Reemplazar fondo', target='Reemplazar fondo', enable_events=True),
            sg.Stretch(),
        ],
        [
            sg.HorizontalSeparator(),
        ],
        [
            sg.Stretch(),
            sg.FileSaveAs('Guardar imagen', target='Guardar imagen', file_types=(('PNG', '.png'), ('JPG', '.jpg')), enable_events=True),
            sg.Stretch(),
        ],
    ]

    initial_resolution = 768
    initial_image = current_state.img
    bio = io.BytesIO()
    initial_image.resize((initial_resolution, initial_resolution)).save(bio, format="PNG")

    image_column = [
        [
            sg.Stretch(), 
            sg.Image(data=bio.getvalue(), key='-IMAGE-'),
            sg.Stretch(),
        ],
    ]

    layout = [
        [
            sg.Stretch(),
            sg.Text('Tamaño de la imagen:'),
            sg.Button('+', key='increase resolution'),
            sg.Text(str(initial_resolution), size=(4,1), key='-RESOLUTION-'),
            sg.Button('-', key='decrease resolution'),
        ],
        [
            sg.Column(buttons_column),
            sg.VerticalSeparator(),
            sg.Column(image_column)
        ],
    ]

    button_list = ['male', 'female', '0-2', '3-6', '7-9', '10-14', '15-19', '20-29', '30-39', '40-49', '50-69', '+70', 'Generar imagen', 'Editar imagen', 'Extraer fondo', 'Reemplazar fondo', 'Guardar imagen', 'increase resolution', 'decrease resolution']

    return sg.Window('Synthetic Face Generator', layout, font=('Arial', 16)), button_list

#----------------------------------------------------------------------------

def init_window_extended(current_state):

    buttons_column = [
        [
            sg.Stretch(),
            sg.Text('Género:'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Radio('Hombre', 'gender', key='male', default=True),
            sg.Radio('Mujer', 'gender', key='female'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Text('Edad:'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Column([
                [sg.Radio('0-2', 'age', key='0-2', default=True)],
                [sg.Radio('7-9', 'age', key='7-9')],
                [sg.Radio('15-19', 'age', key='15-19')],
                [sg.Radio('30-39', 'age', key='30-39')],
                [sg.Radio('50-69', 'age', key='50-69')],
            ]),
            sg.Column([
                [sg.Radio('3-6', 'age', key='3-6')],
                [sg.Radio('10-14', 'age', key='10-14')],
                [sg.Radio('20-29', 'age', key='20-29')],
                [sg.Radio('40-49', 'age', key='40-49')],
                [sg.Radio('+70', 'age', key='+70')],
            ]),
            sg.Stretch(),
        ],
        [
            sg.HorizontalSeparator(),
        ],
        [
            sg.Stretch(),
            sg.Button('Generar imagen'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Text('Truncación:'),
            sg.Button('+', key='increase truncation'),
            sg.Text(f'{current_state.truncation_psi:.1f}', size=(3,1), key='-TRUNCATION_PSI-'),
            sg.Button('-', key='decrease truncation'),
            sg.Stretch(),
        ],
        [
            sg.HorizontalSeparator(),
        ],
        [
            sg.Stretch(),
            sg.Button('Editar imagen'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Text('Modo:'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Radio('condición', 'editing', key='condition'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Radio('dirección latente', 'editing', key='latent', default=True),
            sg.Stretch(),
        ],
        [
            sg.HorizontalSeparator(),
        ],
        [
            sg.Stretch(),
            sg.Button('Extraer fondo'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.FileBrowse('Reemplazar fondo', target='Reemplazar fondo', enable_events=True),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Text('Modelo:'),
            sg.Stretch(),
        ],
        [
            sg.Stretch(),
            sg.Column([
                [sg.Radio('SGHM', 'matting_model', key='SGHM', default=True)],
                [sg.Radio('P3M', 'matting_model', key='P3M-ViTAE')],
                [sg.Radio('AIM', 'matting_model', key='AIM')],
                [sg.Radio('InstMatt', 'matting_model', key='InstMatt')]
            ]),
            sg.Column([
                [sg.Radio('MODNet', 'matting_model', key='MODNet')],
                [sg.Radio('P3M-ViTAE', 'matting_model', key='P3M')],
                [sg.Radio('GFM', 'matting_model', key='GFM')],
                [sg.Radio('MGMatting', 'matting_model', key='MGMatting')]
            ]),
            sg.Stretch(),
        ],
        [
            sg.HorizontalSeparator(),
        ],
        [
            sg.Stretch(),
            sg.FileSaveAs('Guardar imagen', target='Guardar imagen', file_types=(('PNG', '.png'), ('JPG', '.jpg')), enable_events=True),
            sg.Stretch(),
        ],
    ]

    initial_resolution = 768
    initial_image = current_state.img
    bio = io.BytesIO()
    initial_image.resize((initial_resolution, initial_resolution)).save(bio, format="PNG")

    image_column = [
        [
            sg.Stretch(), 
            sg.Image(data=bio.getvalue(), key='-IMAGE-'),
            sg.Stretch(),
        ],
    ]

    layout = [
        [
            sg.Stretch(),
            sg.Text('Tamaño de la imagen:'),
            sg.Button('+', key='increase resolution'),
            sg.Text(str(initial_resolution), size=(4,1), key='-RESOLUTION-'),
            sg.Button('-', key='decrease resolution'),
        ],
        [
            sg.Column(buttons_column),
            sg.VerticalSeparator(),
            sg.Column(image_column)
        ],
    ]

    button_list = ['male', 'female', '0-2', '3-6', '7-9', '10-14', '15-19', '20-29', '30-39', '40-49', '50-69', '+70', 'increase truncation', 'decrease truncation', 'Generar imagen', 'condition', 'latent', 'Editar imagen', 'SGHM', 'MODNet', 'P3M-ViTAE', 'P3M', 'AIM', 'GFM', 'InstMatt', 'MGMatting', 'Extraer fondo', 'Reemplazar fondo', 'Guardar imagen', 'increase resolution', 'decrease resolution']

    return sg.Window('Synthetic Face Generator', layout, font=('Arial', 16)), button_list

#----------------------------------------------------------------------------

@click.command()
@click.option("--full", is_flag=True, show_default=True, default=False, help="Include truncation, edit mode and matting model options")
def simple_gui(full):
    """Simple GUI for generating synthetic faces."""
    
    print('Loading generator network from "%s"...' % NETWORK_PKL)
    if not os.path.isfile(NETWORK_PKL):
        if NETWORK_URL is None:
            print(f'{NETWORK_PKL} not found and download url not provided')
            sys.exit(1)
        print(f'Downloading generator network checkpoint...')
        if not os.path.exists(os.path.dirname(NETWORK_PKL)):
            os.makedirs(os.path.dirname(NETWORK_PKL))
        gdown.download(NETWORK_URL, NETWORK_PKL, quiet=False)
    with dnnlib.util.open_url(NETWORK_PKL) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE) # type: ignore

    if full:
        downloadable_matting_models = ['SGHM', 'MODNet', 'P3M-ViTAE', 'P3M', 'AIM', 'GFM', 'InstMatt', 'MGMatting']
        for matting_model in downloadable_matting_models:
            BackgroundRemoval(matting_model, device=DEVICE)
            torch.cuda.empty_cache()
            gc.collect()

    # Generate dummy image to set up Pytorch plugins
    gen_image_from_z(G, torch.zeros((1, G.z_dim), device=DEVICE), torch.zeros((1, G.c_dim), device=DEVICE))

    current_state = EasyDict(z=None, ws=None, truncation_psi=0.5, img=Image.new("RGB", (256, 256), (255, 255, 255)), editing_mode='latent', matting_model=None, background_removal=BackgroundRemoval('SGHM', device=DEVICE), guidance_calculator=GuidanceCalculator(device=DEVICE))
    window, button_list = init_window_extended(current_state) if full else init_window(current_state)

    # Run the Event Loop
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        for button in button_list:
            window[button].update(disabled=True)
        if event == 'increase truncation':
            update = update_truncation(window, 0.1)
        elif event == 'decrease truncation':
            update = update_truncation(window, -0.1)
        elif event == 'increase resolution':
            update = update_resolution(window, 128)
        elif event == 'decrease resolution':
            update = update_resolution(window, -128)
        elif event == 'Generar imagen':
            update = gen_img(G, window, current_state, values, full)
        elif event == 'Editar imagen':
            update = edit_img(G, current_state, values, full)
        elif event == 'Extraer fondo':
            update = extract_bg(G, current_state, values, full)
        elif event == 'Reemplazar fondo':
            update = replace_bg(G, current_state, values, full)
        elif event == 'Guardar imagen':
            update = save_img(current_state, values)
        if update:
            update_img(window, current_state)
        for button in button_list:
            window[button].update(disabled=False)

    window.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    simple_gui()