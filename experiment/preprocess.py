import argparse
import logging
import sys
from preprocess.process import Preprocess

def get_args():
    parser = argparse.ArgumentParser(description='preprocess the dsen12ms_cr dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--origin_dir', required=True, type=str, help='dir of the dsen12ms_cr dataset', dest='origin_dir')
    parser.add_argument('-o', '--preprocessed_dir', required=True, type=str, help='dir to save the processed files', dest='preprocessed_dir')
    return parser.parse_args()

def config_log():
    file_handler = logging.FileHandler(filename='tmp.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )
if __name__ == '__main__':
    config_log()
    args = get_args()
    preprocessed_dir = args.preprocessed_dir
    origin_dir = args.origin_dir
    preprocess = Preprocess(origin_dir, preprocessed_dir, parallel_num=4, overwrite=False, convert_to_uint8=False, is_crop=True)
    # about 4 mins on gpu04
    preprocess.multiple_run()
