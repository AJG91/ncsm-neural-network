"""
This file contains the functions used for plotting.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import separate_dataset

def plot_counts(x, param, kde=True, save=True):
    fig_path = "./plots/" + param
    
    fig, axes = plt.subplots(x.shape[0]-1, 1, figsize=(12,8), 
                             sharex=True, sharey=True)

    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    plot_labels = ["02-04", "02-06", "02-08", "02-10", "02-12"]

    for i in range(1, x.shape[0]):
        sns.histplot(x=x[i], ax=axes[i-1], binwidth=0.005, kde=kde)
        axes[i-1].tick_params(axis="both", labelsize=20)
        axes[i-1].set_ylabel(axes[i-1].get_ylabel(), fontsize=20)
        axes[i-1].text(0.02, 0.93, r"$N_{\mathrm{max}} =$" + plot_labels[i+1], 
                       transform=axes[i-1].transAxes, ha="left", va="top", 
                       bbox=props, fontsize=20)

    plt.xlabel(r"$E_{bind} \; [MeV]$", fontsize=20)
    plt.xlim(-29.02, -28.76)
    plt.tight_layout()
    
    if save:
        fig.savefig(fig_path + "_stacked_all.png", bbox_inches="tight")
    
    return None

def plot_prediction(data, pred, param, val_sep, save=True):
    fig_path = "./plots/" + param
    
    if val_sep == "Nmax":
        leg_label = r"$N_{max} = $"
        y_label = r"$\hbar \Omega \; [MeV]$"
        y_col = "hw"
        val_list = np.arange(data[val_sep].iloc[0], 
                             data[val_sep].iloc[-1] + 1, 2)
    elif val_sep == "hw":
        leg_label = r"$\hbar \Omega = $"
        y_label = r"$N_{max} = $"
        y_col = "Nmax"
        val_list = np.arange(data["hw"].iloc[0] + 9, 
                             data["hw"].iloc[-1] + 1, 1)
    else:
        raise ValueError("Wrong input!")
    
    fig = plt.figure()
    ax = fig.add_subplot()
    dots = dict(dash_capstyle="round", ls=(0, (0.1, 2)), zorder=2, lw=3)
    
    data = separate_dataset(data, val_sep, val_list)

    for i in range(len(data)):
        ax.plot(data[i][y_col], data[i][param], 
                label=leg_label + str((data[i].reset_index())[val_sep][0]), **dots)

    ax.plot(pred[y_col], pred[param], 
            label=leg_label + str(pred[val_sep][0]), c="k")

    ax.set_xlim(pred[y_col].iloc[0] - 1, pred[y_col].iloc[-1] + 1)
    ax.set_ylim(-30, -18)
    ax.set_xlabel(r"$E_{bind} \; [MeV]$", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    plt.legend(loc="upper right", ncol=2)

    fig.tight_layout()
    
    if save:
        fig.savefig(fig_path + "_prediction.png", bbox_inches="tight")

    return None




