import argparse
import os.path
from multiprocessing import Pool
from utils import *
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--cache_path', type=str, default=None)
parser.add_argument('--threads', type=int, default=10)


def l_mp_multi_loader(args):
    source_path = args.cache_path
    source_act = os.path.join(source_path, 'actions')
    source_obs = os.path.join(source_path, 'observations')
    source_rew = os.path.join(source_path, 'rewards')
    target_path = args.path
    target_act_obs = os.path.join(target_path, 'obs_act')
    target_rew = os.path.join(target_path, 'rewards')
    print('make dir at ', target_act_obs)
    print('make dir at ', target_rew)
    os.makedirs(target_act_obs, exist_ok=True)
    os.makedirs(target_rew, exist_ok=True)
    print(f'compressing cache from {source_path} with multi process loader at: {target_path}')
    thread_num = args.threads
    files2load = os.listdir(source_obs)
    print('adding processes to the pool')
    pool = Pool(processes=thread_num)
    file_idx = 0
    num_files = len(files2load)
    while file_idx < num_files:
        obs_path = os.path.join(source_obs, files2load[file_idx])
        act_path = os.path.join(source_act, files2load[file_idx])
        rew_path = os.path.join(source_rew, files2load[file_idx])
        if os.path.isfile(obs_path) and os.path.isfile(act_path):
            ep_idx, suffix = files2load[file_idx].split('.')
            if suffix == 'npy':
                pool.apply_async(compress_npy, (obs_path, act_path, rew_path, target_act_obs, ep_idx, target_rew, ))
        file_idx += 1
    shutil.move(os.path.join(source_path, 'path_lengths.npy'), target_path)
    shutil.move(os.path.join(source_path, 'traj_returns.npy'), target_path)
    print('closing pool')
    pool.close()
    pool.join()  # make sure all processes are completed
    print('compression done')



def compress(args):
    l_mp_multi_loader(args)


if __name__ == '__main__':
    args = parser.parse_args()
    args.path = os.path.abspath(args.path)
    dir_name = os.path.basename(args.cache_path)
    args.cache_path = os.path.abspath(args.cache_path)
    args.path = os.path.join(args.path, dir_name)
    compress(args)