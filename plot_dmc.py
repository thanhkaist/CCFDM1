import argparse
import ast
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob2

def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, nargs='+')
parser.add_argument('--radius', type=int, default=0)
parser.add_argument('--range', type=int, default=-1, help='Number of transitions want to plot')
parser.add_argument('--legend', type=str, default='', nargs='+')
parser.add_argument('--title', type=str, default='')
parser.add_argument('--shaded_std', type=bool, default=True)
parser.add_argument('--shaded_err', type=bool, default=False)
parser.add_argument('--train_test', action='store_true')
parser.add_argument('--score', action='store_true')
args = parser.parse_args()


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


def get_data_in_subdir(parent_path, x_key, y_key):
    child_paths = [os.path.abspath(os.path.join(path, '..'))
                   for path in glob2.glob(os.path.join(parent_path, '**', 'eval.log'))]

    data_in_subdir = []
    for path in child_paths:
        json_file = os.path.join(path, 'eval.log')
        data = []
        for line in open(json_file, 'r'):
            data.append(json.loads(line))

        len_data = len(data)
        x, y = [], []
        for i in range(len_data):
            x.append(data[i][x_key])
            y.append(data[i][y_key])
        x = np.array(x)
        y = np.array(y)
        y = smooth(y, radius=args.radius)
        data_in_subdir.append((x, y))

    info_env = get_info_env(child_paths[0])
    plot_info = info_env['domain_name'] + '-' + info_env['task_name'] + '-' + info_env['data_augs']

    return data_in_subdir, plot_info


def get_info_env(path):
    info = dict(
        domain_name=None,
        task_name=None,
        data_augs=""
    )
    json_file = os.path.join(path, 'args.json')
    with open(json_file, 'r') as f:
        data = json.load(f)
    for k in info.keys():
        if k in data.keys():
            info[k] = data[k]
    return info

def plot_multiple_results(directories):
    x_key = 'step'
    y_key = 'mean_episode_reward'

    collect_data = []
    plot_titles = []
    for directory in directories:
        data_in_subdir, plot_info = get_data_in_subdir(directory, x_key, y_key)
        collect_data.append(data_in_subdir)
        plot_titles.append(plot_info)

    # Plot data.
    return_means, return_medians, return_stds = [], [], []
    exp_step_idxs = []
    for i in range(len(collect_data)):
        data = collect_data[i]
        xs, ys = zip(*data)
        n_experiments = len(xs)

        # xs, ys = pad(xs), pad(ys)
        if args.range != -1:
            _xs = []
            _ys = []
            for k in range(n_experiments):
                found_idxes = np.argwhere(xs[k] >= args.range)
                if len(found_idxes) == 0:
                    print("[WARNING] Last index is {}, consider choose smaller range in {}".format(
                        xs[k][-1], directories[i]))
                    _xs.append(xs[k][:])
                    _ys.append(ys[k][:])
                else:
                    range_idx = found_idxes[0, 0]
                    _xs.append(xs[k][:range_idx])
                    _ys.append(ys[k][:range_idx])
            xs = _xs
            ys = _ys
        xs, ys = pad(xs), pad(ys)
        xs, ys = np.array(xs), np.array(ys)
        assert xs.shape == ys.shape

        usex = xs[0]
        ymean = np.nanmean(ys, axis=0)
        ymedian = np.nanmedian(ys, axis=0)
        ystd = np.nanstd(ys, axis=0)
        return_means.append(ymean)
        return_medians.append(ymedian)
        return_stds.append(ystd)
        exp_step_idxs.append(usex)
        ystderr = ystd / np.sqrt(len(ys))
        plt.plot(usex, ymean, label='config')
        if args.shaded_err:
            plt.fill_between(usex, ymean - ystderr, ymean + ystderr, alpha=0.4)
        if args.shaded_std:
            plt.fill_between(usex, ymean - ystd, ymean + ystd, alpha=0.2)
        if args.title == '':
            plt.title(plot_titles[i], fontsize='x-large')
        else:
            plt.title(args.title, fontsize='x-large')
        plt.xlabel('Number of steps', fontsize='x-large')
        plt.ylabel('Episode Return', fontsize='x-large')

    plt.tight_layout()
    if args.legend != '':
        assert len(args.legend) == len(
            directories), "Provided legend is not match with number of directories"
        legend_name = args.legend
    else:
        legend_name = [directories[i].split('/')[-1] for i in range(len(directories))]

    # Print scores
    if args.score:
        for i in range(len(collect_data)):
            idx_100k = np.where(exp_step_idxs[i] == 100000)[0]
            idx_200k = np.where(exp_step_idxs[i] == 200000)[0]
            idx_300k = np.where(exp_step_idxs[i] == 300000)[0]
            idx_400k = np.where(exp_step_idxs[i] == 400000)[0]
            idx_500k = np.where(exp_step_idxs[i] == 480000)[0]

            # idx_100k = np.where(exp_step_idxs[i] == 80000)[0]
            # idx_200k = np.where(exp_step_idxs[i] == 200000)[0]
            # idx_300k = np.where(exp_step_idxs[i] == 280000)[0]
            # idx_400k = np.where(exp_step_idxs[i] == 400000)[0]
            # idx_500k = np.where(exp_step_idxs[i] == 480000)[0]

            # idx_100k = np.where(exp_step_idxs[i] == 80000)[0]
            # idx_200k = np.where(exp_step_idxs[i] == 240000)[0]
            # idx_300k = np.where(exp_step_idxs[i] == 320000)[0]
            # idx_400k = np.where(exp_step_idxs[i] == 400000)[0]
            # idx_500k = np.where(exp_step_idxs[i] == 480000)[0]

            if len(idx_100k) < 1:
                print('[WARN] Not found value @ 100k of ', legend_name[i])
                continue
            else:
                idx = idx_100k[0]
                score_mean = return_means[i][idx]
                score_std = return_stds[i][idx]
                print("Ex: %s, score@100k=%.0f+%.0f" % (legend_name[i], score_mean, score_std))

            if len(idx_200k) < 1:
                print('[WARN] Not found value @ 200k of ', legend_name[i])
                continue
            else:
                idx = idx_200k[0]
                score_mean = return_means[i][idx]
                score_std = return_stds[i][idx]
                print("Ex: %s, score@200k=%.0f+%.0f" % (legend_name[i], score_mean, score_std))

            if len(idx_300k) < 1:
                print('[WARN] Not found value @ 300k of ', legend_name[i])
                continue
            else:
                idx = idx_300k[0]
                score_mean = return_means[i][idx]
                score_std = return_stds[i][idx]
                print("Ex: %s, score@300k=%.0f+%.0f" % (legend_name[i], score_mean, score_std))

            if len(idx_400k) < 1:
                print('[WARN] Not found value @ 400k of ', legend_name[i])
                continue
            else:
                idx = idx_400k[0]
                score_mean = return_means[i][idx]
                score_std = return_stds[i][idx]
                print("Ex: %s, score@400k=%.0f+%.0f" % (legend_name[i], score_mean, score_std))

            if len(idx_500k) < 1:
                print('[WARN] Not found value @ 500k of ', legend_name[i])
                continue
            else:
                idx = idx_500k[0]
                score_mean = return_means[i][idx]
                score_std = return_stds[i][idx]
                print("Ex: %s, score@500k=%.0f+%.0f" % (legend_name[i], score_mean, score_std))

            # idx = args.range
            # score_mean = return_means[i][idx]
            # score_std = return_stds[i][idx]
            # print("Ex: %s, score@500k=%.0f+%.0f" % (legend_name[i], score_mean, score_std))


    #plt.legend(legend_name, loc='best', fontsize='x-large')
    plt.legend(legend_name, loc='lower right', fontsize='x-large')
    plt.show()

if __name__ == '__main__':
    directory = []
    for i in range(len(args.dir)):
        if args.dir[i][-1] == '/':
            directory.append(args.dir[i][:-1])
        else:
            directory.append(args.dir[i])
    plot_multiple_results(directory)
