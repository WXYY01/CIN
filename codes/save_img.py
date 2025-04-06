import os
import json
import hashlib
import numpy as np
from pathlib import Path
import torch
from torch import nn
import numpy as np
from thop import profile
from PIL import Image
import shutil
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml



from models.network_scunet import SCUNet

from utils.checkpoint import Checkpoint as check
from models.CIN import CIN

import torch
import models.Network as network

from models.CIN import CINBasic


def load_config(config_path):
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class ImageProcessingDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.image_paths = []
        self.rel_dirs = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        for root, _, files in os.walk(root_dir):
            rel_dir = os.path.relpath(root, root_dir)
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, f))
                    self.rel_dirs.append(rel_dir)

    def __len__(self):
        return len(self.image_paths)
    
    def generate_binary_seed(self, seed_str: str) -> int:
        seed_str = seed_str.lower().replace("\\", "/").split('/')[-1]
        seed_str = seed_str.split('.')[0]
        hash_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big")

    def generate_binary_data(self, seed: int, length: int = 30) -> list:
        rng = np.random.RandomState(seed)
        return rng.randint(0, 2, size=length).tolist()

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            seed = self.generate_binary_seed(img_path)
            binary_data = self.generate_binary_data(seed, 30)
            return self.transform(img), idx, torch.tensor(binary_data, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None, idx

def batch_process(model, dataloader, output_root, device):

    denormalize = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    ])

    with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
        for batch in dataloader:
            inputs, indices, message = batch
            
            valid_inputs = inputs.to(device)
            valid_indices = indices
            
            if valid_inputs.shape[0] == 0:
                continue

            normalized, message = valid_inputs, message.to(device)
            
            outputs = model(normalized, message)
            
            denorm_outputs = denormalize(outputs.cpu().clamp(-1, 1))
            
            for tensor, idx in zip(denorm_outputs, valid_indices):
                orig_path = dataset.image_paths[idx]
                rel_dir = dataset.rel_dirs[idx]
                filename = os.path.basename(orig_path)
                
                output_dir = os.path.join(output_root, rel_dir)
                os.makedirs(output_dir, exist_ok=True)
                
                img = transforms.ToPILImage()(tensor)
                img.save(os.path.join(output_dir, filename))
            
            pbar.update(1)

if __name__ == "__main__":
    input_root = "D:\Github\EXPERIMENT\gtos128_all\train"
    output_root = "D:\Github\EXPERIMENT\CIN-main\CIN-gtos\train"
    batch_size = 32
    num_workers = 4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # noise_config = []
    # hidden_config = HiDDenConfiguration(H=128, W=128,
    #                                     message_length=30,
    #                                     encoder_blocks=4, encoder_channels=64,
    #                                     decoder_blocks=7, decoder_channels=64,
    #                                     use_discriminator=True,
    #                                     use_vgg=False,
    #                                     discriminator_blocks=3, discriminator_channels=64,
    #                                     decoder_loss=1,
    #                                     encoder_loss=0.7,
    #                                     adversarial_loss=1e-3,
    #                                     enable_fp16=False
    #                                     )
    # noiser = Noiser(noise_config, device)
    # model = Hidden(hidden_config, device, noiser, None)

    # checkpoint = torch.load('/data/experiment/model/HiDDeN/runs/30_gtos_I_psnr 2025.03.22--12-31-28/checkpoints/30_gtos_I_psnr--epoch-180--psnr-35.13345308805767.pyt')
    
    
    checkpoint_path = 'D:/Github/EXPERIMENT/CIN-main/runs/epoch-3_step-2907-cinNet_psnr-45.18245837562963.pth'
    checkpoint = torch.load(checkpoint_path) 
    model_name = 'cinNet'
    


    config_path = 'D:\Github\EXPERIMENT\CIN-main\codes\options\opt.yml'  
    config = load_config(config_path)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CINBasic(config, device)
    
    model.load_state_dict(checkpoint[model_name])
    encoder = model.encoder_decoder.encoder
    encoder.eval()
    
    dataset = ImageProcessingDataset(input_root)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    batch_process(encoder, dataloader, output_root, device)