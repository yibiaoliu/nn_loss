import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter


class MyNormalize(mcolors.Normalize):
    def __call__(self, value, clip=None):
        n = (self.vmax + self.vmin) * 0.5
        x, y = [self.vmin, n, self.vmax], [0, 0.35, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_model(v, dh, vmin, vmax, title='', aspect=1, cmap='jet', normed=False, figsize=(5, 5), **kwargs):
    norm = MyNormalize(vmin=1.3, vmax=4.3)
    nx = v.shape[1]
    nz = v.shape[0]
    dx = dh
    if normed:
        par = {'extent': [0, nx * dx / 1000, nz * dx / 1000, 0], 'norm': norm, 'cmap': cmap}
    else:
        par = {'extent': [0, nx * dx / 1000, nz * dx / 1000, 0], 'vmax': vmax, 'vmin': vmin, 'cmap': cmap}
    par.update(kwargs)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(v / 1000, **par)
    plt.title(title, fontsize=25)
    ax.set_aspect(aspect)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xlabel('Position (km)', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylabel('Depth (km)', fontsize=16)

    # 设置颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=16)

    plt.pause(0.001)


def plot_shot(shot, dt, dx, title='', pclip=1.0, scale=1, **kwargs):
    nx = shot.shape[1]
    nt = shot.shape[0]
    vmax = pclip * np.max(np.abs(shot)) / scale
    vmin = - vmax
    par = {'extent': [0, nx * dx / 1000, nt * dt, 0], 'cmap': 'seismic', 'vmin': vmin, 'vmax': vmax};
    par.update(kwargs)
    plt.imshow(shot, **par)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.xlabel('Position (km)', fontsize=16)
    plt.ylabel('Time (s)', fontsize=16)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.formatter = ScalarFormatter()
    cbar.formatter.set_scientific(True)  # 启用科学计数法
    cbar.formatter.set_powerlimits((-3, 3))


    plt.title(title)
    plt.axis('auto')
    plt.pause(0.001)


def plot_diff_shot(true_shot, syn_shot, dt, dx, title="", pclip=1.0, scale1=1, scale2=1, **kwargs, ):
    nx, nt = true_shot.shape[1], true_shot.shape[0]

    vmax1 = pclip * np.max(np.abs(true_shot)) / scale1
    vmin1 = -vmax1

    vmax2 = pclip * np.max(np.abs(syn_shot)) / scale2
    vmin2 = -vmax2

    par = {'extent': [0, nx * dx / 1000, nt * dt, 0], 'cmap': 'seismic', }
    par.update(kwargs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'wspace': 0, 'hspace': 0})
    ax1.imshow(true_shot, vmax=vmax1, vmin=vmin1, **par)
    ax1.invert_xaxis()
    ax1.axis('auto')
    im2 = ax2.imshow(syn_shot, vmax=vmax2, vmin=vmin2, **par)
    ax2.axis('auto')
    ax2.yaxis.tick_right()
    fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.05, pad=0.04, )
    fig.suptitle(title, fontsize=25)
    fig.supylabel("Time(s)", y=0.5, x=0.08)
    fig.supxlabel('Offset (km)')
    plt.subplots_adjust(wspace=0, hspace=0)


def check_for_value(array, x):
    return 1 if np.any(array == x) else 0




