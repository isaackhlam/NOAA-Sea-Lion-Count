from utils.utils import set_seed
from utils.parser import build_parser
from dataset.dataset import SegmentationDataset, build_dataloader
from tqdm import tqdm

def main(args):
    dataset = SegmentationDataset(args)
    dataloader = build_dataloader(args, dataset)

    for x, y in tqdm(dataloader):
        print(x.shape)
        print(y.shape)
        break


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)

