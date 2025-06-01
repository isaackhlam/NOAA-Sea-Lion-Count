import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import skimage
from pathlib import Path

class NewsDataset(Dataset):
    def __init__(self, args):
        self.r = args.scale_r
        self.patch_size = args.patch_size
        self.should_downscale_image = args.downscale
        self.data = []
        self.train_data_input_path = args.train_data_input_path
        self.train_data_label_path = args.train_data_label_path
        misalign_data = Path(args.train_data_input_path) / Path(args.train_misalign_data)

        with open(misalign_data, 'r') as f:
            self.misalign_ids = [l.strip() for l in f]

        N = 100 # replace to len(train_data) later
        for i in range(N):
            if str(i) in self.misalign_ids:
                return
            X, Y = self._load_data(i)
            if X:
                self.data.append([X, Y])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _load_data(self, idx):
        im1 = cv2.imread(Path(self.train_data_label_path) / Path(f'{idx}.jpg'))
        im2 = cv2.imread(Path(self.train_data_input_path) / Path(f'{idx}.jpg'))

        im3 = cv2.absdiff(im1, im2)
        mask = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        mask[mask < 50] = 0
        mask[mask > 0] = 255
        im4 = cv2.bitwise_or(im3, im3, mask=mask)

        im6 = np.max(im4, axis=2)
        blobs = skimage.feature.blob_log(im6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

        h, w, _ = im2.shape
        if self.should_downscale_image:
            res = np.zeros((int((w * self.scale_r) // self.patch_size) + 1, int((h * self.scale_r) // self.patch_size) + 1, 5), dtype='int16')
        else:
            res = np.zeros((int(w // self.patch_size) + 1, int(h // self.patch_size) + 1, 5), dtype='int16')

        im = cv2.GaussianBlur(im1, (5, 5), 0)
        for blob in blobs:
            y, x, s = blob
            b, g, r = im[int(y)][int(x)][:]
            if self.should_downscale_image:
                x1 = int((x * self.scale_r) // self.patch_size)
                y1 = int((y * self.scale_r) // self.patch_size)
            else:
                x1 = int(x // self.patch_size)
                y1 = int(y // self.patch_size)

            if r > 225 and b < 25 and g < 25: # RED
                res[x1, y1, 0]+=1
            elif r > 225 and b > 225 and g < 25: # MAGENTA
                res[x1, y1, 1]+=1
            elif r < 75 and b < 50 and 150 < g < 200: # GREEN
                res[x1, y1, 4]+=1
            elif r < 75 and  150 < b < 200 and g < 75: # BLUE
                res[x1, y1, 3]+=1
            elif 60 < r < 120 and b < 50 and g < 75:  # BROWN
                res[x1, y1, 2]+=1

        ma = cv2.cvtColor((1 * (np.sum(im1, axis=2) > 20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
        if self.should_downscale_image:
            im = cv2.resize(im2 * ma, (int(w * self.scale_r), int(h * self.scale_r)))
        else:
            im = cv2.resize(im2 * ma, (w, h))
        h, w, c = im.shape

        X = []
        Y = []

        for i in range(int(w // self.patch_size)):
            for j in range(int(h // self.patch_size)):
                Y.append(res[i, j, :])
                X.append(im[j * self.patch_size:j * self.patch_size + self.patch_size, i * self.patch_size: i * self.patch_size + self.patch_size, :])

        return np.array(X), np.array(Y)


class SegmentationDataset(Dataset):
    def __init__(self, args, img_size=(512, 512), dot_radius=5):
        self.raw_img_dir = Path(args.train_data_input_path)
        self.dot_img_dir = Path(args.train_data_label_path)
        self.img_paths = sorted(list(self.raw_img_dir.glob("*.jpg")))
        self.img_size = img_size
        self.dot_radius = dot_radius

        # Define colors and class names
        self.cls_colors = [
            [1., 0., 0.],        # red: adult males
            [1., 0., 1.],        # magenta: subadult males
            [0.647, 0.1647, 0.1647],  # brown: adult females
            [0., 0., 1.],        # blue: juveniles
            [0., 1., 0.],        # green: pups
        ]
        self.cls_names = ['male', 'sub_male', 'female', 'juv', 'pup']
        self.cls_colors_bgr = [
            (int(r * 255), int(g * 255), int(b * 255))
            for r, g, b in self.cls_colors
        ]
        self.color_tol = 5  # tolerance for color matching

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        raw_path = self.img_paths[idx]
        dot_path = self.dot_img_dir / raw_path.name

        # Load images
        image = cv2.imread(str(raw_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dot_img = cv2.imread(str(dot_path))

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Parse dots by color
        for cls_id, bgr_color in enumerate(self.cls_colors_bgr):
            lower = np.array(bgr_color) - self.color_tol
            upper = np.array(bgr_color) + self.color_tol
            lower = np.clip(lower, 0, 255)
            upper = np.clip(upper, 0, 255)

            dot_mask = cv2.inRange(dot_img, lower, upper)
            ys, xs = np.where(dot_mask > 0)
            for x, y in zip(xs, ys):
                cv2.circle(mask, (int(x), int(y)), self.dot_radius, int(cls_id + 1), -1)

        # Resize image and mask
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
        mask = torch.from_numpy(mask).long()

        return image, mask

def build_dataloader(args, dataset):

    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_data,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )


