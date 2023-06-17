import os
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset

class LoadImages(Dataset):
    def __init__(self, main_dir, transform):
        #set the loading directory
        self.main_dir = main_dir
        self.transform = transform

        #list all images in folder and count them
        all_imgs = os.listdir(main_dir)
        self.total_images = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_images[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image