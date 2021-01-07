import os
import tqdm
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
import PIL


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, min_len = 32):
        dataset = ImageFolder(folder)
        self.total_frames = 0
        self.lengths = []
        self.images = []

        for idx, (im, categ) in enumerate(tqdm.tqdm(dataset, desc='Counting total number of frames')):
            img_path, _ = dataset.imgs[idx]
            shorter, longer = min(im.width, im.height), max(im.width, im.height)
            length = longer // shorter
            if length >= min_len:
                self.images.append((img_path, categ))
                self.lengths.append(length)

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of frames {}".format(np.sum(self.lengths)))

    def __getitem__(self, item):
        path, label = self.images[item]
        im = PIL.Image.open(path)
        return im, label

    def __len__(self):
        return len(self.images)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset

        self.transforms = transform

    def __getitem__(self, item):
        if item != 0:
            video_id = np.searchsorted(self.dataset.cumsum, item) - 1
            frame_num = item - self.dataset.cumsum[video_id] - 1
        else:
            video_id = 0
            frame_num = 0

        video, target = self.dataset[video_id]
        video = np.array(video)

        horizontal = video.shape[1] > video.shape[0]

        if horizontal:
            i_from, i_to = video.shape[0] * frame_num, video.shape[0] * (frame_num + 1)
            frame = video[:,i_from: i_to, ::]
        else:
            i_from, i_to = video.shape[1] * frame_num, video.shape[1] * (frame_num + 1)
            frame = video[i_from: i_to,:,::]

        if frame.shape[0] == 0:
            print("video {}. From {} to {}. num {}".format(video.shape, i_from, i_to, item))

        return {"images": self.transforms(frame), "categories": target}

    def __len__(self):
        return self.dataset.cumsum[-1]


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, every_nth = 1, transform=None):
        self.dataset = dataset
        self.video_length = video_length
        self.every_nth = every_nth
        self.transforms = transform

    def __getitem__(self, item):
        video, target = self.dataset[item]
        video = np.array(video)

        horizontal = video.shape[1] > video.shape[0]
        shorter, longer = min(video.shape[0], video.shape[1]), max(video.shape[0], video.shape[1])
        video_len = longer // shorter

        if video_len >= self.video_length * self.every_nth:
            needed = self.every_nth * (self.video_length-1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif video_len >= self.video_length:
            subsequence_idx = np.arange(0, self.video_length)
        else:
            raise Exception("Length is too short id - {}, len - {}".format(self.dataset[item], video_len))

        frames = np.split(video, video_len, axis = 1 if horizontal else 0)
        selected = np.array([frames[s_id] for s_id in subsequence_idx])

        return {"images": self.transforms(selected), "categories": target}

    def __len__(self):
        return len(self.dataset)
