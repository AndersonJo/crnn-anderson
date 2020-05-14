import os
from argparse import Namespace, ArgumentParser

BASE_DIR = os.path.dirname(__file__)


def init() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--data_path', default='train_data')
    parser.add_argument('--output', default='label.txt', type=str, help='the path of label output')
    opt = parser.parse_args()

    opt.data_path = os.path.join(BASE_DIR, opt.data_path)
    opt.output = os.path.join(BASE_DIR, opt.output)
    return opt


def generate_y_labels(opt: Namespace):
    chars = set()
    max_len = 0
    for filename in os.listdir(opt.data_path):
        filename = filename[:-4]
        [chars.add(c) for c in filename]
        max_len = max(max_len, len(filename))

    chars = sorted(list(chars))
    with open(opt.output, 'wt') as f:
        for c in chars:
            f.write(c + '\n')

    print('Y Label Generation has been done')
    print('number of characters:', len(chars))
    print('max len:', max_len)


def main():
    opt = init()
    generate_y_labels(opt)


if __name__ == '__main__':
    main()
