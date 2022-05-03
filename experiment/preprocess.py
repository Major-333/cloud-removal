import argparse
import logging
from preprocess.process import Preprocess

def get_args():
    parser = argparse.ArgumentParser(description='preprocess the dsen12ms_cr dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--origin_dir', type=str, help='dir of the dsen12ms_cr dataset', dest='origin_dir')
    parser.add_argument('-o', '--preprocessed_dir', type=str, help='dir to save the processed files', dest='preprocessed_dir')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(filename='logger.log',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            format='%(asctime)s %(levelname)-8s %(message)s')
    args = get_args()
    preprocessed_dir = args.preprocessed_dir
    origin_dir = args.origin_dir
    preprocess = Preprocess(origin_dir, preprocessed_dir, parallel_num=4, overwrite=False)
    # about 4 mins on gpu04
    preprocess.multiple_run()
