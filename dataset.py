import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn


def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    dali_aug = False,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    elif os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

    # Image Folder
    else:
        transform = transforms.Compose([
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        train_set = ImageFolder(root_dir, transform)

    # DALI
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank, dali_aug=dali_aug)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )
    return train_loader

def get_dataloader_subset(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    dali_aug = False,
    seed = 2048,
    num_workers = 2,
    ratio = 0.1
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    elif os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)
        from torch.utils.data import Subset
        labels = torch.load("ms1mv3_labels.pt")
        _, cts = labels.unique(return_counts = True)
        idxs = torch.arange(cts[:10000].sum())
        train_set = Subset(train_set, idxs)                       

    # Image Folder
    else:
        transform = transforms.Compose([
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        train_set = ImageFolder(root_dir, transform)

    # DALI
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank, dali_aug=dali_aug)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )
    return train_loader

def attr_get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    attr_repr_dir,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    if os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)
        
    attr_repr = torch.load(attr_repr_dir, map_location="cpu")    
    train_set = AttrDataset(train_set, attr_repr)
    
    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )
    return train_loader


def poison_get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    poison_module_dir,
    poison_ratio=1,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    if os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)
        
    # Load Poison Module
    # Will be revised soon
    from backbones.custom_generator import ResNetGenerator
    poison_module = ResNetGenerator()
    poison_module.load_state_dict(torch.load(poison_module_dir, map_location = "cpu"))
    poison_module = poison_module.eval().cuda()
    train_set = PoisonedDataset(
        train_set, poison_module, poison_ratio
    )
    
    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )
    return train_loader


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


# Wrapper for the data with attribution 
class AttrDataset(Dataset):
    def __init__(self, dataset, attr_repr):
        super().__init__()
        assert len(dataset) == attr_repr.size(0)
        self.dataset = dataset
        self.attr_repr = attr_repr
    
    def __getitem__(self, idx):
        x,y = self.dataset[idx]
        attr = self.attr_repr[idx]
        return x,y,attr
    
    def __len__(self):
        return len(self.dataset)
    
# Wrapper for the data with indices    
class IndexAugmentedDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        x,y = self.dataset.__getitem__(index)
        return x,y,index

    def __len__(self):
        return len(self.dataset)    
    
    
# Wrapper for Poisoned Dataset
class PoisonedDataset(Dataset):
    def __init__(self, dataset, poison_module, poison_ratio,
                 poison_idx = None, save_noise = False):
        super().__init__()
        self.dataset = dataset
        self.poison_module = poison_module.eval()
        self.poison_ratio = max(0, min(poison_ratio, 1))
        self.device = next(poison_module.parameters()).device
        
        # Initialization
        self.poison_idx = self.build_poison_idxs(poison_idx)
        self.idx_map = dict()
        counter = 0
        for idx in self.poison_idx:
            self.idx_map[idx] = counter
            counter += 1
        
        # Generate Poisons for selected indices
        self.poison_noises = self.build_poison_noises(save_noise = save_noise)    
        
            
    def __getitem__(self, idx):
        if idx in self.idx_map:
            x,y = self.dataset[idx]
            noise = self.poison_noises[self.idx_map[idx]]
            return x + noise, y
            
        else: 
            return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)

    def build_poison_idxs(self, poison_idx):
        if poison_idx != None:
            return poison_idx
        
        num_samples = len(self.dataset)
        num_poison_idx = int(num_samples * self.poison_ratio)
        return torch.randperm(num_samples)[:num_poison_idx].tolist()

    @torch.no_grad()
    def build_poison_noises(self, batch_size=128, save_noise=False):
        poison_dict = dict()        
        _poison_idx = torch.LongTensor(self.poison_idx)
        
        sub_dataset = Subset(self.dataset, self.poison_idx)
        sub_dataloader = DataLoader(sub_dataset, batch_size, shuffle = False, num_workers = 4)
        noise_buf = torch.zeros(len(self.poison_idx), 3, 112, 112)
        
        start = 0
        with torch.no_grad():
            for x, y in sub_dataloader:
                noise = self.poison_module(x.to(self.device))
                noise_buf[start:start+noise.size(0)] = noise.cpu()            
                start += noise.size(0)
            
        del x, y, noise
        del self.poison_module
        gc.collect()
        torch.cuda.empty_cache()            
        
        
        if save_noise:
            torch.save(noise_buf, "poison_data.pt")
            torch.save(_poison_idx, "poison_idx.pt")
        
        return noise_buf    
    
    
class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5),
    dali_aug=False
    ):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img
    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img
    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        img = fn.hsv(img, saturation=saturate)
        return img
    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img
    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        if dali_aug:
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()
