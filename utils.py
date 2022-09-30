import tree
import h5py
from copy import deepcopy
import os
import numpy as np
import shutil

def segment2(traj_input, terminals, max_path_length=None, file_path=None):
    """Segment data as a list of trajectory arrays
    traj_input: a tree-structure of np.ndarray
        whose first axes are of the same length
    terminals: sigals to split traj_input into lists
    max_path_length: default None, whether to truncate data
    when trajectory is too long.
    """
    if max_path_length is not None:
        assert max_path_length > 0

    data_size = set(tree.flatten(tree.map_structure(lambda x: len(x), traj_input)))
    try:
        assert len(data_size) == 1
    except:
        print(file_path)
        raise
    data_size = list(data_size)[0]
    assert data_size == len(terminals)

    # tmp_traj = []
    trajectories = []
    start = 0
    for i, term in enumerate(terminals):
        # tmp_traj.append(
        #     tree.map_structure(lambda x: x[i], traj_input)
        # )
        if term.squeeze() or (
            max_path_length is not None and i - start + 1 >= max_path_length
        ):
            trajectories.append(
                tree.map_structure(lambda x: x[start : i + 1], traj_input)
            )
            start = i + 1
    if start < i + 1:
        trajectories.append(tree.map_structure(lambda x: x[start : i + 1], traj_input))
    return trajectories


def extraction(file_path):
    dataset, h5path, status = mp_qlearning_loader(dataset_path=file_path, terminate_on_end=True)
    if h5path is not None:
        return [None, None, None, None, file_path, status]
    observations = dataset["observations"]
    actions = dataset["actions"]
    terminals = dataset["terminals"]
    rewards = dataset["rewards"]

    res = segment2((observations, actions, rewards), terminals, file_path=file_path)
    observations, actions, rewards = tuple(zip(*res))
    # since reward is definitely an array
    path_lengths = [len(x) for x in rewards]
    return observations, actions, rewards, path_lengths, None, None

def compress_npy(obs_path, act_path, rew_path, target_act_obs, ep_idx, target_rew):
    obs_array = np.load(obs_path)
    act_array = np.load(act_path)
    np.savez_compressed(os.path.join(target_act_obs, ep_idx + '.npz'), observations=obs_array, actions=act_array)
    shutil.move(rew_path, target_rew)
    os.remove(obs_path)
    os.remove(act_array)



def calculator(file_path):
    max_ep_reward = -1e9
    min_ep_reward = 1e9
    max_ep_length = -1e9
    min_ep_length = 1e9
    dataset, h5path, status = mp_qlearning_loader(dataset_path=file_path, terminate_on_end=True)
    if h5path is not None:
        return [None, None, None, None, file_path, status]
    observations = dataset["observations"]
    actions = dataset["actions"]
    terminals = dataset["terminals"]
    rewards = dataset["rewards"]
    res = segment2((observations, actions, rewards), terminals, file_path=file_path)
    observations, actions, rewards = tuple(zip(*res))
    # since reward is definitely an array
    for x in rewards:
        max_ep_length = max(max_ep_length, len(x))
        min_ep_length = min(min_ep_length, len(x))
        ep_rew = sum(x)
        max_ep_reward = max(max_ep_reward, ep_rew)
        min_ep_reward = min(ep_rew, min_ep_reward)
    return max_ep_reward, min_ep_reward, max_ep_length, min_ep_length, None, None


def ___get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys


def mp_qlearning_loader(
    dataset_path=None, terminate_on_end=False, **kwargs
):
    """Reference from TT code, but we do not need next_obs"""
    dataset, h5path, status = __get_dataset(dataset_path)
    if h5path is None:
        obs_ = deepcopy(tree.map_structure(lambda x: x, dataset["observations"]))
        action_ = dataset["actions"].copy()
        reward_ = dataset["rewards"].copy()
        terminal_done_ = dataset["terminals"].copy()
        if "timeouts" in dataset:
            timeout_done_ = dataset["timeouts"].copy()
            done_ = terminal_done_ | timeout_done_
        else:
            done_ = terminal_done_
        return {
            "observations": obs_,
            "actions": action_,
            "rewards": reward_[:, None],
            "terminals": done_[:, None],
            "realterminals": terminal_done_[:, None],
        }, None, None
    else:
        return None, h5path, status

def __get_dataset(h5path):
    data_dict = {}
    try:
        with h5py.File(h5path, 'r') as dataset_file:
            for k in ___get_keys(dataset_file):
                if not data_dict.get(k, None):
                    data_dict[k] = []
                try:  # first try loading as an array
                    # data_dict[k].append(dataset_file[k][:])
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    # data_dict[k].append(dataset_file[k][()])
                    data_dict[k] = dataset_file[k][()]
    except:
        print('unable to open file: ', h5path)
        return None, h5path, 0
    N_samples = data_dict['rewards'].shape[0]
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:, 0]
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    # Run a few quick sanity checks
    observations = data_dict["observations"]
    actions = data_dict["actions"]
    terminals = data_dict["terminals"]
    rewards = data_dict["rewards"]
    len_check = {len(observations), len(actions), len(terminals), len(rewards)}
    if len(len_check) != 1:
        print('ill dataset: ', h5path)
        return None, h5path, 1
    return data_dict, None, None


def ___get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys


def cache_data_multi(offset, observations, actions, rewards, obs_path, act_path, reward_path):
    os.makedirs(os.path.join(obs_path), exist_ok=True)
    os.makedirs(os.path.join(act_path), exist_ok=True)
    os.makedirs(os.path.join(reward_path), exist_ok=True)
    n_traj = len(rewards)
    for i in range(n_traj):
        np.save(
            obs_path + os.sep + str(i + offset) + ".npy", np.array(observations[i])
        )
        np.save(
            act_path + os.sep + str(i + offset) + ".npy",
            np.array(actions[i], dtype=actions[i].dtype),
        )
        np.save(
            reward_path + os.sep + str(i + offset) + ".npy",
            np.array(rewards[i], dtype=rewards[i].dtype),
        )
    return n_traj