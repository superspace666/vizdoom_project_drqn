import os
import argparse
from logging import getLogger
from loggered import get_dump_path, get_logger
from src.params_management import params_get


logger = getLogger()

parser = argparse.ArgumentParser(description='main runner')
parser.add_argument("--main_dump_path", type=str, default="./dumped",
                    help="Main dump path")
parser.add_argument("--exp_name", type=str, default="default",
                    help="Experiment name")
args, remaining = parser.parse_known_args()
assert len(args.exp_name.strip()) > 0
dump_path = get_dump_path('C:\\Users\\IsaacChen\\Desktop', 'exp1')
logger = get_logger(filepath=os.path.join(dump_path, 'train.log'))
logger.info('========== Running DOOM ==========')
logger.info('Experiment will be saved in: %s' % dump_path)

# begin the trainning
params_get.parse_game_args(remaining + ['--dump_path', dump_path])

