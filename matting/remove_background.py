import os
import numpy as np
import torch
import cv2
import toml
import gdown
import zipfile
import onedrivedownloader

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from skimage.transform import resize
from torchvision import transforms
from pymatting import estimate_foreground_ml
from easydict import EasyDict

from .P3M_Net.core.network import build_model as P3M_build_model
from .SemanticGuidedHumanMatting import utils as SGHM_utils
from .SemanticGuidedHumanMatting.model.model import HumanMatting as SGHM_HumanMatting
from .P3M.core.network.P3mNet import P3mNet
from .P3M.core.config import PRETRAINED_R34_MP
from .AIM.core.network.AimNet import AimNet
from .GFM.core.gfm import GFM
from .MODNet.src.models.modnet import MODNet
from .InstMatt import utils as InstMatt_utils
from .InstMatt import networks as InstMatt_networks
from .InstMatt.utils import CONFIG as InstMATT_CONFIG
from .MGMatting import utils as MGMatting_utils
from .MGMatting import networks as MGMatting_networks
from .MGMatting.utils import CONFIG as MGMatting_CONFIG


class BackgroundRemoval():
    def __init__(self, model_name, device='cuda'):
        self.device = device
        self.model = None
        self.preprocess = None
        self.matte_prediction = None
        self.__load_parameters(model_name)

    def __call__(self, img, guidance=None):
        sample = self.preprocess(img, guidance)
        if sample is None:
            return np.append(img, np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8), axis=2)
        matte = self.matte_prediction(sample)
        foreground = estimate_foreground_ml(img[:,:,::-1] / 255, matte / 255) * 255
        foreground = foreground[:,:,::-1]
        img_nb = np.concatenate((foreground, matte[:,:,np.newaxis]), axis=-1)
        return img_nb.astype(np.uint8)

    def __load_parameters(self, model_name):
        if not os.path.exists('matting/pretrained'):
            os.makedirs('matting/pretrained')
        if model_name == 'SGHM':
            self.__load_sghm()
        elif model_name == 'MODNet':
            self.__load_modnet()
        elif model_name == 'P3M-ViTAE' or model_name == 'P3M':
            self.__load_p3m(model_name)
        elif model_name == 'AIM':
            self.__load_aim()
        elif model_name == 'GFM':
            self.__load_gfm()
        elif model_name == 'InstMatt':
            self.__load_instmatt()
        elif model_name == 'MGMatting':
            self.__load_mgmatting()
        else:
            raise KeyError(f'{model_name} is not a valid matting model')

#----------------------------------------------------------------------------

    def __resize_dims(self, h, w, size=512, m=32):
        if max(h, w) < size or min(h, w) > size:
            if w >= h:
                resize_h = size
                resize_w = int(w / h * size)
            elif w < h:
                resize_w = size
                resize_h = int(h / w * size)
        else:
            resize_h = h
            resize_w = w
        resize_h = resize_h - resize_h % m
        resize_w = resize_w - resize_w % m
        return resize_h, resize_w

    def __calculate_padding(self, h, w):
        if h % 32 == 0 and w % 32 == 0:
            pad = 32
        else:
            target_h = 32 * ((h - 1) // 32 + 1)
            target_w = 32 * ((w - 1) // 32 + 1)
            pad = (target_w - w, target_h - h)
        return pad

    def __transform_image(self, img, pad, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(pad),
            transforms.Normalize(mean=mean,std=std)
        ])
        img = transform_img(img)
        img = img.to(self.device)
        return img

#----------------------------------------------------------------------------

    # Reference: https://github.com/cxgincsu/SemanticGuidedHumanMatting, SemanticGuidedHumanMatting/inference.py

    def __load_sghm(self):
        ckpt_path = 'matting/pretrained/SGHM-ResNet50.pth'
        url = 'https://drive.google.com/uc?id=1Ar5ASgfCUBmgZLwLHz6lThCQ-2EVnvqr'
        if not os.path.isfile(ckpt_path):
            print(f'Downloading SGHM checkpoint...')
            gdown.download(url, ckpt_path, quiet=False)
        state_dict = torch.load(ckpt_path)
        self.model = SGHM_HumanMatting(backbone='resnet50')
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.preprocess = self.__preprocess_sghm
        self.matte_prediction = self.__matte_prediction_sghm

    def __preprocess_sghm(self, img, guidance=None):
        h, w = img.shape[:2]
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        resize_h, resize_w = self.__resize_dims(h, w, size=1280, m=64)
        img = torch.nn.functional.interpolate(img, size=(resize_h, resize_w), mode='bilinear')
        
        sample = {'image': img, 'alpha_shape': (h, w)}
        return sample

    def __matte_prediction_sghm(self, sample):
        with torch.no_grad():
            pred = self.model(sample['image'])

        # progressive refine alpha
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        pred_alpha = alpha_pred_os8.clone().detach()
        weight_os4 = SGHM_utils.get_unknown_tensor_from_pred(pred_alpha, rand_width=30, train_mode=False)
        pred_alpha[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = SGHM_utils.get_unknown_tensor_from_pred(pred_alpha, rand_width=15, train_mode=False)
        pred_alpha[weight_os1>0] = alpha_pred_os1[weight_os1>0]

        pred_alpha = pred_alpha.repeat(1, 3, 1, 1)
        pred_alpha = torch.nn.functional.interpolate(pred_alpha, size=sample['alpha_shape'], mode='bilinear')
        alpha_np = pred_alpha[0].data.cpu().numpy().transpose(1, 2, 0)
        alpha_np = alpha_np[:, :, 0] * 255

        return alpha_np

#----------------------------------------------------------------------------

    # Reference: https://github.com/ZHKKKe/MODNet, MODNet/demo/image_matting/colab/inference.py

    def __load_modnet(self):
        ckpt_path = 'matting/pretrained/modnet_photographic_portrait_matting.ckpt'
        url = 'https://drive.google.com/uc?id=1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz'
        if not os.path.isfile(ckpt_path):
            print(f'Downloading MODNet checkpoint...')
            gdown.download(url, ckpt_path, quiet=False)
        state_dict = torch.load(ckpt_path)
        self.model = MODNet(backbone_pretrained=False)
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(state_dict, strict=False)     
        self.model = self.model.to(self.device)
        self.model.eval()
        self.preprocess = self.__preprocess_modnet
        self.matte_prediction = self.__matte_prediction_modnet

    def __preprocess_modnet(self, img, guidance=None):
        h, w = img.shape[:2]
        img = self.__transform_image(img, pad=0, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        img = img.unsqueeze(0)
        resize_h, resize_w = self.__resize_dims(h, w)
        img = torch.nn.functional.interpolate(img, size=(resize_h, resize_w), mode='area')
        
        sample = {'image': img, 'alpha_shape': (h, w)}
        return sample

    def __matte_prediction_modnet(self, sample):
        with torch.no_grad():
            _, _, alpha = self.model(sample['image'], True)
        alpha = torch.nn.functional.interpolate(alpha, size=sample['alpha_shape'], mode='area')
        alpha = alpha[0][0].data.cpu().numpy() * 255
        return alpha

#----------------------------------------------------------------------------

    # References: https://github.com/JizhiziLi/P3M, P3M/core/test.py
    #             https://github.com/ViTAE-Transformer/P3M-Net, P3M-Net/core/infer.py

    def __load_p3m(self, model_name):
        if model_name == 'P3M':
            ckpt_path = 'matting/pretrained/p3mnet_pretrained_on_p3m10k.pth'
            url = 'https://drive.google.com/uc?id=1smX2YQGIpzKbfwDYHAwete00a_YMwoG1'
            if not os.path.isfile(PRETRAINED_R34_MP):
                print(f'Downloading backbone resnet34 checkpoint...')
                gdown.download('https://drive.google.com/uc?id=18Pt-klsbkiyonMdGi6dytExQEjzBnHwY', PRETRAINED_R34_MP, quiet=False)
            self.model = P3mNet()
        elif model_name == 'P3M-ViTAE':
            ckpt_path = 'matting/pretrained/P3M-Net_ViTAE-S_trained_on_P3M-10k.pth'
            url = 'https://drive.google.com/uc?id=1QbSjPA_Mxs7rITp_a9OJiPeFRDwxemqK'
            self.model = P3M_build_model('vitae', pretrained=False)
        if not os.path.isfile(ckpt_path):
            print(f'Downloading {model_name}\'s checkpoint...')
            gdown.download(url, ckpt_path, quiet=False)
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.preprocess = self.__preprocess_p3m
        self.matte_prediction = self.__matte_prediction_p3m

    def __preprocess_p3m(self, img, guidance=None):
        h, w = img.shape[:2]
        resize_h, resize_w = self.__resize_dims(h, w)
        img = resize(img, (resize_h, resize_w))
        img = self.__transform_image(img, 0)
        img = img.to(torch.float32)
                
        sample = {'image': img.unsqueeze(0), 'alpha_shape': (h, w)}
        return sample

    def __matte_prediction_p3m(self, sample):
        with torch.no_grad():
            alpha = self.model(sample['image'])[2]
        alpha = alpha.data.cpu().numpy()[0,0,:,:]
        alpha = resize(alpha, sample['alpha_shape']) * 255
        return alpha

#----------------------------------------------------------------------------

    # Reference: https://github.com/JizhiziLi/AIM, P3M/core/test.py

    def __load_aim(self):
        ckpt_path = 'matting/pretrained/aimnet_pretrained_matting.pth'
        url = 'https://drive.google.com/uc?id=16dd1FGMcsMTqR6EfD2T9mtRmPwxnY0zs'
        if not os.path.isfile(PRETRAINED_R34_MP):
            print(f'Downloading backbone resnet34 checkpoint...')
            gdown.download('https://drive.google.com/uc?id=18Pt-klsbkiyonMdGi6dytExQEjzBnHwY', PRETRAINED_R34_MP, quiet=False) 
        if not os.path.isfile(ckpt_path):
            print(f'Downloading AIM checkpoint...')
            gdown.download(url, ckpt_path, quiet=False)
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']
        self.model = AimNet()
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.preprocess = self.__preprocess_aim
        self.matte_prediction = self.__matte_prediction_aim

    def __preprocess_aim(self, img, guidance=None):
        h, w = img.shape[:2]
        resize_h, resize_w = self.__resize_dims(h, w, size=256)
        img = resize(img, (resize_h, resize_w))
        img = self.__transform_image(img, 0)
        img = img.to(torch.float32)
                
        sample = {'image': img.unsqueeze(0), 'alpha_shape': (h, w)}
        return sample

    def __matte_prediction_aim(self, sample):
        with torch.no_grad():
            alpha = self.model(sample['image'])[2]
        alpha = alpha.data.cpu().numpy()[0,0,:,:]
        alpha = resize(alpha, sample['alpha_shape']) * 255
        return alpha

#----------------------------------------------------------------------------

    # References: https://github.com/JizhiziLi/GFM, P3M/core/test.py

    def __load_gfm(self):
        ckpt_path = 'matting/pretrained/model_r34_2b_gfm_tt.pth'
        url = 'https://drive.google.com/uc?id=1Y8dgOprcPWdUgHUPSdue0lkFAUVvW10Q'
        if not os.path.isfile(ckpt_path):
            print(f'Downloading GFM checkpoint...')
            gdown.download(url, ckpt_path, quiet=False)
        state_dict = torch.load(ckpt_path)
        self.model = GFM(EasyDict(backbone='r34_2b', rosta='TT'))
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.preprocess = self.__preprocess_gfm
        self.matte_prediction = self.__matte_prediction_gfm

    def __preprocess_gfm(self, img, guidance=None):
        h, w = img.shape[:2]
        resize_h, resize_w = self.__resize_dims(h, w)
        img = resize(img, (resize_h, resize_w)) * 255.0
        img = torch.from_numpy(img.astype(np.float32)).permute(2,0,1)
        img = img.to(self.device)
                
        sample = {'image': img.unsqueeze(0), 'alpha_shape': (h, w)}
        return sample

    def __matte_prediction_gfm(self, sample):
        with torch.no_grad():
            alpha = self.model(sample['image'])[2]
        alpha = alpha.data.cpu().numpy()[0,0,:,:]
        alpha = resize(alpha, sample['alpha_shape']) * 255
        return alpha

#----------------------------------------------------------------------------

    # Reference: https://github.com/nowsyn/InstMatt, InstMatt/infer.py

    def __load_instmatt(self):
        config = 'matting/InstMatt/config/InstMatt-stage2.toml'
        with open(config) as f:
            InstMatt_utils.load_config(toml.load(f))
        if InstMatt_utils.CONFIG.is_default:
            raise ValueError("No .toml config loaded.")
        ckpt_path = 'matting/pretrained/InstMatt_best_model.pth'
        url = 'https://drive.google.com/uc?id=1i_zQEqSG2i86G2jSz2IxgbanfyWqu-BF'
        if not os.path.isfile(ckpt_path):
            print(f'Downloading InstMatt checkpoint...')
            gdown.download(url, 'matting/pretrained/InstMatt.zip', quiet=False)
            with zipfile.ZipFile('matting/pretrained/InstMatt.zip', 'r') as zip_ref:
                zip_ref.extractall('matting/pretrained')
            os.rename('matting/pretrained/InstMatt/best_model.pth', 'matting/pretrained/InstMatt_best_model.pth')
            os.remove('matting/pretrained/InstMatt.zip')
            os.rmdir('matting/pretrained/InstMatt')
        ckpt = torch.load(ckpt_path)
        state_dict = InstMatt_utils.remove_prefix_state_dict(ckpt['state_dict'])
        self.model = InstMatt_networks.get_generator(
            InstMATT_CONFIG,
            encoder=InstMATT_CONFIG.model.arch.encoder,
            decoder=InstMATT_CONFIG.model.arch.decoder,
            refiner=InstMATT_CONFIG.model.arch.refiner,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.preprocess = self.__preprocess_instmatt
        self.matte_prediction = self.__matte_prediction_instmatt

    def __preprocess_instmatt(self, img, masks):
        if masks is None:
            return None
        
        h, w = img.shape[:2]
        pad = self.__calculate_padding(h, w)
        p_img = self.__transform_image(img, pad)
        
        masks = torch.tensor(masks, dtype=torch.float32)
        mask_batch = []
        for i in range(len(masks)):
            mask_t = masks[i]
            mask_r = torch.sum(masks, dim=0) - mask_t
            mask_b = 1 - torch.sum(masks, dim=0)
            mask_f = torch.stack([mask_t, mask_r, mask_b], dim=0)
            mask_batch.append(mask_f)
        mask_batch = torch.stack(mask_batch, dim=0)
        mask_batch = transforms.Pad(pad)(mask_batch)
        mask_batch = mask_batch.to(self.device)

        sample = {'image': p_img.unsqueeze(0).repeat(len(masks),1,1,1), 'mask': mask_batch, 'alpha_shape': (h, w), 'oh': h, 'ow': w}
        return sample

    def __single_inference_instmatt(self, sample):
        with torch.no_grad():
            image, mask = sample['image'], sample['mask']

            pred = [self.model(image[i:i+1], mask[i:i+1], is_training=False) for i in range(mask.size(0))]
            pred = InstMatt_utils.reduce_dict(pred)
            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

            ### refinement
            alpha_pred = alpha_pred_os8.clone().detach()
            weight_os4 = InstMatt_utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=InstMATT_CONFIG.model.self_refine_width1, train_mode=False)
            weight_os4 = weight_os4.max(dim=1, keepdim=True)[0]
            alpha_pred = alpha_pred * (weight_os4<=0).float() + alpha_pred_os4 * (weight_os4>0).float()
            weight_os1 = InstMatt_utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=InstMATT_CONFIG.model.self_refine_width2, train_mode=False)
            weight_os1 = weight_os1.max(dim=1, keepdim=True)[0]
            alpha_pred = alpha_pred * (weight_os1<=0).float() + alpha_pred_os1 * (weight_os1>0).float()

            if self.model.refiner is not None:
                alpha_pred_list = self.model.forward_refiner(image, alpha_pred.clone().detach(), pred['feature'].clone().detach(), is_training=False, nostop=False)
                alpha_pred = alpha_pred * (weight_os1<=0).float() + alpha_pred_list[-1] * (weight_os1>0).float()

        h, w = sample['alpha_shape']
        alpha_pred = alpha_pred[:, 0, ...].data.cpu().numpy()
        alpha_pred = np.transpose(alpha_pred, (1,2,0))
        alpha_pred = alpha_pred * 255
        alpha_pred = alpha_pred.astype(np.uint8)
        alpha_pred = alpha_pred[32:h+32, 32:w+32]
        if alpha_pred.shape[0] != sample['oh'] or alpha_pred.shape[1] != sample['ow']:
            alpha_pred = cv2.resize(alpha_pred, (sample['ow'], sample['oh']))
        if len(alpha_pred.shape) == 2:
            alpha_pred = alpha_pred[:,:,None]

        return alpha_pred

    def __matte_prediction_instmatt(self, sample):
        alpha_preds = self.__single_inference_instmatt(sample)

        # Determine the alpha matte of the person in the center of the image
        max_center_prob, center_alpha = float('-inf'), None
        for i in range(alpha_preds.shape[2]):
            alpha_pred = alpha_preds[:,:,i]
            h, w = alpha_pred.shape
            center = h // 2, w // 2
            if alpha_pred[center] > max_center_prob:
                max_center_prob = alpha_pred[center]
                center_alpha = alpha_pred
                
        return center_alpha

#----------------------------------------------------------------------------

    # Reference: https://github.com/yucornetto/MGMatting, MGMatting/code-base/infer.py

    def __load_mgmatting(self):
        config = 'matting/MGMatting/config/MGMatting-RWP-100k.toml'
        with open(config) as f:
            MGMatting_utils.load_config(toml.load(f))
        if MGMatting_utils.CONFIG.is_default:
            raise ValueError("No .toml config loaded.")
        ckpt_path = 'matting/pretrained/latest_model.pth'
        url = 'https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/qyu13_jh_edu/Edl8x0nQjy1JhGP6rcV0N-cB654HpmZZa5bwW9rYUvmsJg?e=J3lSba'
        if not os.path.isfile(ckpt_path):
            print(f'Downloading MGMatting checkpoint...')
            onedrivedownloader.download(url, ckpt_path)
        ckpt = torch.load(ckpt_path)
        state_dict = MGMatting_utils.remove_prefix_state_dict(ckpt['state_dict'])
        self.model = MGMatting_networks.get_generator(
            encoder=MGMatting_CONFIG.model.arch.encoder,
            decoder=MGMatting_CONFIG.model.arch.decoder
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.preprocess = self.__preprocess_mgmatting
        self.matte_prediction = self.__matte_prediction_mgmatting

    def __preprocess_mgmatting(self, img, mask):
        if mask is None:
            return None
        
        h, w = img.shape[:2]
        pad = self.__calculate_padding(h, w)
        p_img = self.__transform_image(img, pad)

        mask = np.expand_dims(mask, axis=0)
        mask = torch.tensor(mask, dtype=torch.float32)
        mask = transforms.Pad(pad)(mask)
        mask = mask.to(self.device)

        sample = {'image': p_img.unsqueeze(0), 'mask': mask.unsqueeze(0), 'alpha_shape': (h, w)}
        return sample

    def __single_inference_mgmatting(self, sample, post_process=False):
        with torch.no_grad():
            image, mask,alpha_shape = sample['image'], sample['mask'], sample['alpha_shape']
            pred = self.model(image, mask)
            alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

            ### refinement
            alpha_pred = alpha_pred_os8.clone().detach()
            weight_os4 = MGMatting_utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=MGMatting_CONFIG.model.self_refine_width1, train_mode=False)
            alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
            weight_os1 = MGMatting_utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=MGMatting_CONFIG.model.self_refine_width2, train_mode=False)
            alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]

            h, w = alpha_shape
            alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy()
            if post_process:
                alpha_pred = MGMatting_utils.postprocess(alpha_pred)
            alpha_pred = alpha_pred * 255
            alpha_pred = alpha_pred.astype(np.uint8)
            alpha_pred = alpha_pred[32:h+32, 32:w+32]

        return alpha_pred

    def __matte_prediction_mgmatting(self, sample):
        alpha_pred = self.__single_inference_mgmatting(sample)
        return alpha_pred

#----------------------------------------------------------------------------


class GuidanceCalculator():
    def __init__(self, device='cuda'):
        self.device = device
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.maskrcnn = maskrcnn_resnet50_fpn_v2(weights=weights)
        self.maskrcnn = self.maskrcnn.to(self.device)
        self.maskrcnn.eval()
        self.maskrcnn_preprocess = weights.transforms()
        self.model_guidance = {}
        self.model_guidance['InstMatt'] = self.__masks_calculation
        self.model_guidance['MGMatting'] = self.__center_mask_calculation

    def __call__(self, model_name, img):
        return self.model_guidance.get(model_name, lambda x: None)(img)

    def __instance_segmentation(self, img):
        batch = self.maskrcnn_preprocess(torch.from_numpy(img).permute(2,0,1)).unsqueeze(0)
        batch = batch.to(self.device)
        results = self.maskrcnn(batch)[0]
        return results['labels'], results['scores'], results['masks'] 

    def __masks_calculation(self, img):
        labels, scores, masks = self.__instance_segmentation(img)
        
        person_masks = []
        for label, score, mask in zip(labels, scores, masks):
            if label != 1 or score < 0.7:
                continue
            mask = mask[0].data.cpu().numpy()
            h, w = mask.shape
            ratio = (mask>0).sum() / float(h*w)
            if ratio<0.02:
                continue
            person_masks.append(mask>=0.5)
        person_masks = np.array(person_masks, dtype=np.uint8)

        if len(person_masks) == 0:
            return None

        return person_masks

    def __get_center_mask(self, labels, masks):
        max_center_prob, center_mask = float('-inf'), None
        for label, mask in zip(labels, masks):
            if label != 1:
                continue
            mask = mask[0]
            h, w = mask.shape
            center = h // 2, w // 2
            if mask[center] > max_center_prob:
                max_center_prob = mask[center]
                center_mask = mask
        return center_mask

    def __center_mask_calculation(self, img):
        labels, _, masks = self.__instance_segmentation(img)

        center_mask = self.__get_center_mask(labels, masks)
        if center_mask is None:
            return None

        center_mask = center_mask.data.cpu().numpy()
        center_mask = (center_mask >= 0.5).astype(np.float32)

        return center_mask
