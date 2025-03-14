import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from MODNet.src.models.modnet import MODNet
warnings.filterwarnings("ignore")


class BGRemove:
    ref_size = 512
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((512, 512)),  # ✅ Ensure resizing here to avoid MPS issues
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # 🔹 Use Metal (MPS) if available, otherwise fallback to CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def __init__(self, ckpt_path):
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = nn.DataParallel(self.modnet)
        self.modnet.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.modnet.eval()
        self.modnet.to(self.device)  # 🔹 Move model to correct device

    def pre_process(self, im):
        im = cv2.resize(im, (512, 512))  # ✅ Ensure correct resizing
        im = self.im_transform(im)
        im = im[None, :, :, :]
        im = im.to(self.device)  # 🔹 Move image tensor to the correct device
        return im

    def post_process(self, mask_data, frame):
        # ✅ Ensure resizing works correctly on Mac without triggering MPS pool error
        matte = F.interpolate(mask_data, size=(frame.shape[0], frame.shape[1]), mode='bilinear', align_corners=False)
        matte = matte.repeat(1, 3, 1, 1)
        matte = matte[0].data.cpu().numpy().transpose(1, 2, 0)
        matte = np.uint8(matte * frame + (1 - matte) * 255)
        return matte

    def process_frame(self, frame):
        with torch.no_grad():  # Disable gradients for performance boost
            im = self.pre_process(frame)
            _, _, matte = self.modnet(im, inference=False)
            return self.post_process(matte, frame)