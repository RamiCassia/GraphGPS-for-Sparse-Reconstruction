import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class Plot():

    def plot_unfolded(field, field_s, title = '', channel = 0, time = 0, save = False, epoch = 0, path = ''):

        n = np.shape(field)[2]
        fig = plt.figure(figsize=(13.5, 4.5))
        gs = GridSpec(3, 9, figure=fig, wspace=0.05, hspace=0.05)
        omit_subplots = [(0, 0), (0, 2), (0, 3), (2,0), (2,2), (2,3)]

        def plot_single(offset, f):
            for i in range(3):
                for j in range(4):
                    if (i, j) not in omit_subplots:
                        ax = fig.add_subplot(gs[i, j + offset])

                        if (i,j) == (1,0):
                            ax.imshow(np.fliplr(np.flipud(f[time,channel,0,:,:])))
                        if (i,j) == (1,1):
                            ax.imshow(np.flipud(f[time,channel,:,:,0]))
                        if (i,j) == (1,2):
                            ax.imshow(np.flipud(f[time,channel,n-1,:,:]))
                        if (i,j) == (1,3):
                            ax.imshow(np.fliplr(np.flipud(f[time,channel,:,:,n-1])))
                        if (i,j) == (0,1):
                            ax.imshow(np.flipud(f[time,channel,:,n-1,:]))
                        if (i,j) == (2,1):
                            ax.imshow(f[time,channel,:,0,:])

                        ax.set_xticks([])
                        ax.set_yticks([])

        plot_single(offset = 0, f = field)
        plot_single(offset = 5, f = field_s)

        fig.suptitle(title, fontsize=16)

        if save:
            plt.savefig(path + str(epoch) + '.png',  bbox_inches='tight')

        plt.show()



 

