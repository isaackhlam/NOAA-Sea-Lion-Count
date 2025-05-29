from utils.utils import set_seed
from utils.parser import build_parser

def main(args):
    print(args)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)

