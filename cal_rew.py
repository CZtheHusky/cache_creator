import argparse
import os.path
import json
from tqdm import tqdm
from multiprocessing import Pool
import random
from utils import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--cache_path', type=str, default=None)
parser.add_argument('--threads', type=int, default=32)
parser.add_argument('--cache_loc', nargs='+', type=int, default=[-1])


def mp_multi_loader(args, dataset_path):
    datasets_status = {'ill_datasets': [], 'broken_datasets': []}
    print('calculating episode reward of dataset ', dataset_path)
    with open(os.path.join(dataset_path, '0_0_readme.json'), 'r') as f:
        dataset_attributes = json.load(f)

    thread_num = args.threads
    files2load = os.listdir(dataset_path)
    print('adding processes to the pool')
    global_max_ep_reward = -1e9
    global_min_ep_reward = 1e9
    global_max_ep_length = -1e9
    global_min_ep_length = 1e9
    pool = Pool(processes=thread_num)
    pool_results = []
    for file in files2load:
        file_path = os.path.join(dataset_path, file)
        if os.path.isfile(file_path):
            _, suffix = file.split('.')
            if suffix == 'hdf5':
                pool_results.append(pool.apply_async(calculator, (file_path,)))
    pool.close()
    tmp_res = []
    pbar = tqdm(total=len(pool_results))
    while len(pool_results) > 0:
        done_cal = 0
        for r in pool_results:
            if r.ready():
                tmp_res.append(r.get())
                done_cal += 1
                pool_results.remove(r)
        if len(tmp_res) >= 1 or len(pool_results) == 0:
            # [[observations, actions, rewards, path_lengths]]
            for item in tmp_res:
                # [observations, actions, rewards, path_lengths]
                max_ep_reward, min_ep_reward, max_ep_length, min_ep_length, ill_file_path, status = item
                if ill_file_path is None:
                    global_max_ep_reward = max(global_max_ep_reward, max_ep_reward)
                    global_min_ep_reward = min(global_min_ep_reward, min_ep_reward)
                    global_max_ep_length = max(global_max_ep_length, max_ep_length)
                    global_min_ep_length = min(global_min_ep_length, min_ep_length)
                elif status == 1:
                    print('ill dataset, file_path: ', ill_file_path)
                    datasets_status['ill_datasets'].append(ill_file_path)
                elif status == 0:
                    print('broken dataset, file_path: ', ill_file_path)
                    datasets_status['broken_datasets'].append(ill_file_path)
            tmp_res.clear()
        pbar.update(done_cal)
        time.sleep(5)  # ensures that this thread doesn't consume too much cpu
    pbar.close()
    pool.join()  # make sure all processes are completed
    dataset_attributes['Min_ep_reward'] = float(global_min_ep_reward)
    dataset_attributes['Max_ep_reward'] = float(global_max_ep_reward)
    dataset_attributes['Min_ep_length'] = int(global_min_ep_length)
    dataset_attributes['Max_ep_length'] = int(global_max_ep_length)
    print(datasets_status)
    print(dataset_attributes)
    dataset_attributes = json.dumps(dataset_attributes)
    datasets_status = json.dumps(datasets_status)
    with open(os.path.join(dataset_path, '0_0_readme.json'), 'w') as f:
        f.write(dataset_attributes)
        f.write(datasets_status)





def cache(args, dataset_path):
    dataset_path = os.path.join(args.path, dataset_path)
    mp_multi_loader(args, dataset_path)


if __name__ == '__main__':
    args = parser.parse_args()
    args.path = os.path.abspath(args.path)
    if 'dataset' not in os.path.basename(args.path) and 'cached' not in os.path.basename(args.path):
        dir_name = os.path.basename(args.path)
        args.path = os.path.dirname(args.path)
        cache(args, dir_name)
    else:
        dirs2load = os.listdir(args.path)
        for idx, dir_name in enumerate(dirs2load):
            cache(args, dir_name)
