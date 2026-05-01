"""Shared style for Package_C figures (Ku project guidelines)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 16,
    'axes.titlepad': 6,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 13,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans','Arial'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.9,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#DDDDDD',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.8,
    'legend.frameon': False,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})
KU = ['#C94F4A','#E8943A','#4AACB0','#5B8DB8']  # warm muted earth-tone primary 4
KU_EXT = KU + ['#8C6B40','#A77BCA']              # extension for 6 classes

def panel(ax, label):
    """Place panel label in (A), (B) format at y=1.18 outside axes."""
    ax.text(0.0, 1.18, f'({label})', transform=ax.transAxes,
            fontsize=28, fontweight='bold', ha='left', va='bottom', clip_on=False)

def setup(ax, xlabel='', ylabel='', title=''):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title, pad=6, fontsize=14)
