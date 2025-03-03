import os, random, math
import torch

class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_dir: str,
    ):
        super().__init__()        
        
        self.data_dir = data_dir

        self.data = []

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.pt') and file[0] != ".":
                    self.data.append(os.path.join(root, file))
        print(f"load cached videos: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dpath = self.data[idx]
        try:
            # print("==", dpath)
            data = torch.load(dpath, weights_only=True)
        except:
            print(">>> error file:", dpath)
            data = torch.load(self.data[idx + 1])

        index = random.randint(0, len(data["captions"]) - 1)
        latent, embeds = data["latents"][index], data["embedds"][index]
        assert not torch.isnan(latent.flatten()[0]), f"latent nan, {dpath}"
        assert not torch.isnan(embeds.flatten()[0]), f"embeds nan, {dpath}"
        
        return latent, embeds, data["masks"][index], data["captions"][index], data["meta_info"][index]

class BucketSampler(torch.utils.data.Sampler):
    def __init__(self):
        pass

class MultiDatasetWraper(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.data = datasets
        self.lenlist = [len(dd) for dd in datasets]
        print("self.lenlist", self.lenlist)        
        
    def __len__(self):
        return sum(self.lenlist)

    def __getitem__(self, idx):
        acc = 0
        for dindex, length in enumerate(self.lenlist):
            if idx <= length - 1 + acc and idx >= acc:
                real_index = idx - acc
                return self.data[dindex][real_index]    
            acc += length
            
        print("error!!, this should not be printed", idx)
        return self.data[0][0]

    @staticmethod
    def calc_index_for_multi_datasets(list_of_datasets):
        dataset_indices = []
        idxAccumulate = 0
        for dd in list_of_datasets:
            idxrange = [idxAccumulate + ii for ii in range(len(dd))]
            dataset_indices.append(idxrange)
            idxAccumulate += len(dd)
        return dataset_indices

class MixedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_indices, batch_size):
        self.dataset_indices = dataset_indices
        self.batch_size = batch_size
        self.counts = sum([ len(dd) for dd in self.dataset_indices])

    def __iter__(self):
        batch = []
        num_batch = self.counts // self.batch_size
        for _ in range(num_batch):
            idxs = random.choice(self.dataset_indices)
            batch = random.sample(idxs, self.batch_size)
            yield batch
            batch = []
        
    def __len__(self):
        return self.counts // self.batch_size