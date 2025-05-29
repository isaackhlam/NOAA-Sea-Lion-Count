from torch.utils.data import DataLoader, Dataset

class NewsDataset(Dataset):
    def __init__(self, args):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def build_dataloader(args, dataset):

    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_data,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )


