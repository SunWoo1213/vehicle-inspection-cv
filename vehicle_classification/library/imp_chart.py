import matplotlib.pyplot as plt


def plot_multi_line_detail(datas, window=None):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    for e in datas['list']:
        if window is None:
            ax.plot(e['data'], label=e['label'],
                    color=e['color'], animated=True, linewidth=1)
        else:
            ax.plot(e['data'][window[0]:window[1]], label=e['label'],
                    color=e['color'], animated=True, linewidth=1)
    plt.legend(loc='lower left')
    ax.set_title(datas['title'])
    plt.show()


def plot_multi_line(datas, window=None):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    for e in datas:
        if window is None:
            ax.plot(e, animated=True, linewidth=1)
        else:
            ax.plot(e[window[0]:window[1]], animated=True, linewidth=1)
    # plt.legend(loc='lower left')
    # ax.set_title(datas['title'])
    plt.show()


def plot_each_line(datas, window=None):
  row = len(datas)
  fig, axes = plt.subplots(nrows=row, ncols=1, figsize=(16,6), dpi=80)
  for i in range(row):
    if window is None:
      axes[i].plot(datas[i], label=f'{i}')
    else:
      axes[i].plot(datas[i][window[0]:window[1]], label=f'{i}')
