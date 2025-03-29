import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

dataset = 'lip'
model_restore = r".\checkpoints\lip_final.pth"
gpu = '0'
input_dir = r"C:\Users\ASUS\Desktop\Nandan\Dataset\train\image"
output_dir = r"C:\Users\ASUS\Desktop\Nandan\Dataset\train\image_parse"
save_logits = False

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
    }
}

def get_palette(num_cls):
    palette = [0] * (num_cls * 3)
    for j in range(num_cls):
        lab = j
        i = 0
        while lab:
            palette[j * 3] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def main():
    global dataset
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    num_classes = dataset_settings[dataset]['num_classes']
    input_size = dataset_settings[dataset]['input_size']

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
    state_dict = torch.load(model_restore)['state_dict']
    model.load_state_dict({k[7:]: v for k, v in state_dict.items()})
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

    dataset = SimpleFolderDataset(root=input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    os.makedirs(output_dir, exist_ok=True)
    palette = get_palette(num_classes)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            image, meta = batch
            img_name = meta['name'][0]
            c, s, w, h = meta['center'].numpy()[0], meta['scale'].numpy()[0], meta['width'].numpy()[0], meta['height'].numpy()[0]
            
            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0)).squeeze().permute(1, 2, 0)
            
            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result_path = os.path.join(output_dir, img_name[:-4] + '.png')
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)
            output_img.save(parsing_result_path)
            
            if save_logits:
                np.save(os.path.join(output_dir, img_name[:-4] + '.npy'), logits_result)

if __name__ == '__main__':
    main()