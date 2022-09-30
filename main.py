import argparse
import os.path
import traceback
from tqdm import tqdm
from multiprocessing import Pool
import random
from utils import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--sp_loader', action='store_true', default=False)
parser.add_argument('--cache_path', type=str, default=None)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--cache_loc', nargs='+', type=int, default=[-1])


def sp_multi_loader(args, path, obs_path, action_path, reward_path):
    dataset_path = path
    cache_path = args.cache_path
    print('creating cache with single process loader at : ', cache_path)
    total_path_lengths = []
    cur_offset = 0
    files2load = os.listdir(dataset_path)
    random.shuffle(files2load)
    for file in tqdm(files2load):
        file_path = os.path.join(dataset_path, file)
        if os.path.isfile(file_path):
            _, suffix = file.split('.')
            if suffix == 'hdf5':
                file_path = 'E:\\cache_creator\\1_33.hdf5'
                dataset, h5path, status = mp_qlearning_loader(dataset_path=file_path, terminate_on_end=True)
                observations = dataset["observations"]
                actions = dataset["actions"]
                terminals = dataset["terminals"]
                rewards = dataset["rewards"]

                res = segment2((observations, actions, rewards), terminals)
                observations, actions, rewards = tuple(zip(*res))

                # since reward is definitely an array
                path_lengths = [len(x) for x in rewards]
                total_path_lengths.extend(path_lengths)
                n_traj = cache_data_multi(cur_offset, observations, actions, rewards, obs_path, action_path,
                                          reward_path)
                cur_offset += n_traj
    np.save(cache_path + os.sep + "path_lengths.npy", np.array(total_path_lengths))


def mp_multi_loader(args, path, obs_path, action_path, reward_path):
    dataset_path = path
    cache_path = args.cache_path
    print('creating cache with multi process loader at : ', cache_path)
    thread_num = args.threads
    total_path_lengths = []
    cur_offset = 0
    files2load = os.listdir(dataset_path)
    random.shuffle(files2load)
    print('adding processes to the pool')
    pool = Pool(processes=thread_num)
    pool_results = []
    for file in files2load:
        file_path = os.path.join(dataset_path, file)
        if os.path.isfile(file_path):
            _, suffix = file.split('.')
            if suffix == 'hdf5':
                pool_results.append(pool.apply_async(extraction, (file_path,)))
    pool.close()
    tmp_res = []
    print('loading datasets')
    pbar = tqdm(total=len(pool_results))
    while len(pool_results) > 0:
        done_cal = 0
        for r in pool_results:
            if r.ready():
                tmp_res.append(r.get())
                done_cal += 1
                pool_results.remove(r)
                print('thread done')
        if len(tmp_res) >= 1 or len(pool_results) == 0:
            # [[observations, actions, rewards, path_lengths]]
            for item in tmp_res:
                # [observations, actions, rewards, path_lengths]
                observations, actions, rewards, path_lengths = item
                total_path_lengths.extend(path_lengths)
                pbar.set_description('stored: {}'.format(len(total_path_lengths)))
                traj_num = cache_data_multi(cur_offset, observations, actions, rewards, obs_path, action_path,
                                            reward_path)
                cur_offset += traj_num
            tmp_res.clear()
        pbar.update(done_cal)
        time.sleep(5)  # ensures that this thread doesn't consume too much cpu
    print('closing pool')
    pbar.close()
    pool.join()  # make sure all processes are completed
    print('load done')
    np.save(cache_path + os.sep + "path_lengths.npy", np.array(total_path_lengths))


def l_mp_multi_loader(args, path, obs_path, action_path, reward_path):
    dataset_path = path
    ill_datasets = []
    broken_datasets = []
    cache_path = args.cache_path
    print('creating cache with multi process loader at : ', cache_path)
    thread_num = args.threads
    total_path_lengths = []
    cur_offset = 0
    files2load = os.listdir(dataset_path)
    random.shuffle(files2load)
    print('adding processes to the pool')
    pool = Pool(processes=thread_num)
    pool_results = []
    file_idx = 0
    num_files = len(files2load)
    pbar = tqdm(total=num_files - 1, desc='stored: {}'.format(len(total_path_lengths)))
    while file_idx < num_files or (file_idx > 0 and len(pool_results) > 0):
        if file_idx < num_files and len(pool_results) < 4 * thread_num:
            file_path = os.path.join(dataset_path, files2load[file_idx])
            if os.path.isfile(file_path):
                _, suffix = files2load[file_idx].split('.')
                if suffix == 'hdf5':
                    pool_results.append(pool.apply_async(extraction, (file_path,)))
            file_idx += 1
        else:
            tmp_res = []
            tqdm_step = 0
            for r in pool_results:
                if r.ready():
                    try:
                        tmp_res.append(r.get())
                        tqdm_step += 1
                        pool_results.remove(r)
                    except:
                        traceback.print_exc()
                        raise
            if len(tmp_res) >= 1 or len(pool_results) == 0:
                # [[observations, actions, rewards, path_lengths]]
                for item in tmp_res:
                    # [observations, actions, rewards, path_lengths]
                    observations, actions, rewards, path_lengths, ill_file_path, status = item
                    if ill_file_path is None:
                        total_path_lengths.extend(path_lengths)
                        traj_num = cache_data_multi(cur_offset, observations, actions, rewards, obs_path,
                                                    action_path, reward_path)
                        cur_offset += traj_num
                    elif status == 1:
                        print('ill dataset, file_path: ', ill_file_path)
                        ill_datasets.append(ill_file_path)
                    elif status == 0:
                        print('broken dataset, file_path: ', ill_file_path)
                        broken_datasets.append(ill_file_path)
                    pbar.set_description('stored: {}'.format(len(total_path_lengths)))
                tmp_res.clear()
            pbar.update(tqdm_step)
            time.sleep(5)
    pool_results.clear()
    print('closing pool')
    pool.close()
    pool.join()  # make sure all processes are completed
    print('load done')
    np.save(cache_path + os.sep + "path_lengths.npy", np.array(total_path_lengths))
    print('ill datasets')
    print(ill_datasets)
    print('broken datasets')
    print(broken_datasets)



def cache(args, dataset_path):
    dataset_path = os.path.join(args.path, dataset_path)
    env_name = os.path.basename(dataset_path) + '-expert-v0'
    args.cache_path = os.path.join(args.cache_path, env_name)
    obs_path = os.path.join(args.cache_path, 'observations')
    action_path = os.path.join(args.cache_path, 'actions')
    reward_path = os.path.join(args.cache_path, 'rewards')

    if not args.sp_loader:
        # mp_multi_loader(args) # this function causes memory leak, since the loading speed is way faster than writing
        l_mp_multi_loader(args, dataset_path, obs_path, action_path, reward_path)
    else:
        sp_multi_loader(args, dataset_path, obs_path, action_path, reward_path)


if __name__ == '__main__':
    args = parser.parse_args()
    args.path = os.path.abspath(args.path)
    if 'dataset' not in os.path.basename(args.path):
        args.cache_path = '/raid/gato_data_cache'
        dir_name = os.path.basename(args.path)
        args.path = os.path.dirname(args.path)
        cache(args, dir_name)
    else:
        cache_loc = args.cache_loc
        dirs2load = os.listdir(args.path)
        for idx, dir_name in enumerate(dirs2load):
            cache_target = None if cache_loc[idx % len(cache_loc)] == -1 else cache_loc[idx % len(cache_loc)]
            if cache_target is None:
                args.cache_path = '/raid/gato_data_cache'
            else:
                args.cache_path = '/nfs/dgx0' + str(cache_target) + '/raid/gato_data_cache'
            cache(args, dir_name)
