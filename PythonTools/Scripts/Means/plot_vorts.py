import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean, sys, PlotTools, os, shutil, datetime, glob
from netCDF4 import Dataset
from matplotlib.colors import ListedColormap

dpi = PlotTools.dpi

try: # Try using mpi
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
except:
    rank = 0
    num_procs = 1
print("Proc {0:d} of {1:d}".format(rank+1,num_procs))

# Get the available filter files
files = glob.glob('filter_*.nc')

# If the Figures directory doesn't exist, create it.
# Same with the Figures/tmp
out_direct = os.getcwd() + '/Videos'
tmp_direct = out_direct + '/tmp'

if (rank == 0):
    print("Saving outputs to " + out_direct)
    print("  will use temporary directory " + tmp_direct)

    if not(os.path.exists(out_direct)):
        os.makedirs(out_direct)

    if not(os.path.exists(tmp_direct)):
        os.makedirs(tmp_direct)

source = Dataset('input.nc', 'r')
try:
    units = source.variables['latitude'].units
except:
    units = ''

# Create cmap for mask data
ref_cmap = cmocean.cm.gray
mask_cmap = ref_cmap(np.arange(ref_cmap.N))
mask_cmap[:,-1] = np.linspace(1, 0, ref_cmap.N)
mask_cmap = ListedColormap(mask_cmap)

# Some parameters for plotting
cbar_props     = dict(pad = 0.02, shrink = 0.85)
gridspec_props = dict(wspace = 0.15, hspace = 0.15, left = 0.1, right = 0.95, bottom = 0.1, top = 0.9)

# Plot time-mean
for fp in files[rank::num_procs]:
    with Dataset(fp, 'r') as results:

        scale = results.filter_scale
        print("{0:.3g}km".format(scale/1e3))

        # Get the grid from the first filter
        if units == 'm':
            latitude  = results.variables['latitude'][:] / 1e3
            longitude = results.variables['longitude'][:] / 1e3
        else:
            latitude  = results.variables['latitude'][:]
            longitude = results.variables['longitude'][:]
        LON, LAT = np.meshgrid(longitude, latitude)

        depth = results.variables['depth'][:]
        time  = results.variables['time'][:] * (60*60) # convert hours to seconds
        mask  = results.variables['mask'][:]

        Ntime = len(time)

        # Do some time handling tp adjust the epochs
        # appropriately
        epoch       = datetime.datetime(1950,1,1)   # the epoch of the time dimension
        dt_epoch    = datetime.datetime.fromtimestamp(0)  # the epoch used by datetime
        epoch_delta = dt_epoch - epoch  # difference
        time        = time - epoch_delta.total_seconds()  # shift

        # lat/lon lines to draw
        meridians = np.round(np.linspace(longitude.min(), longitude.max(), 5))
        parallels = np.round(np.linspace(latitude.min(),  latitude.max(), 5))

        # Map projection
        proj = PlotTools.MapProjection(longitude, latitude)
        Xp, Yp = proj(LON, LAT, inverse=False)

        rat   = (Xp.max() - Xp.min()) / (Yp.max() - Yp.min())
        rat  *= (2./1) * (1.2)
        fig_h = 8.

        ## Vorticity dichotomies
        if (Ntime > 1):

            # Initialize figure
            fig, axes = plt.subplots(1, 2,
                sharex=True, sharey=True, squeeze=False, 
                gridspec_kw = gridspec_props,
                figsize=(fig_h*rat, fig_h))

            fig.suptitle('Time average')

            for jj in range(1):
    
                if jj == 0:
                    vort = 'vort_r'
                if jj == 1:
                    vort = 'vort_lon'
                if jj == 2:
                    vort = 'vort_lat'

                to_plot_below = np.mean(results.variables['fine_'   + vort][:, 0, :, :], axis=0)
                to_plot_above = np.mean(results.variables['coarse_' + vort][:, 0, :, :], axis=0)
    
                CV_a = np.percentile(np.abs(to_plot_above[~to_plot_above.mask]), 99.9)
                CV_b = np.percentile(np.abs(to_plot_below[~to_plot_below.mask]), 99.9)
    
                qm_a  = axes[jj,0].pcolormesh(Xp, Yp, to_plot_above, 
                        cmap='cmo.balance', vmin = -CV_a, vmax = CV_a)
                qm_b  = axes[jj,1].pcolormesh(Xp, Yp, to_plot_below, 
                        cmap='cmo.balance', vmin = -CV_b, vmax = CV_b)

                cbar_a = plt.colorbar(qm_a, ax = axes[jj,0], **cbar_props)
                cbar_b = plt.colorbar(qm_b, ax = axes[jj,1], **cbar_props)
                PlotTools.ScientificCbar(cbar_a, units='m/s')
                PlotTools.ScientificCbar(cbar_b, units='m/s')

                # Add land and lat/lon lines
                for ax in axes[jj,:]:
                    ax.pcolormesh(Xp, Yp, mask, vmin=-1, vmax=1, cmap=mask_cmap)
                    PlotTools.AddParallels_and_Meridians(ax, proj, 
                        parallels, meridians, latitude, longitude)

                for ax in axes.ravel():
                    ax.set_aspect('equal')
        
            axes[0,0].set_ylabel('$\omega_r$', fontsize=16)
            #axes[1,0].set_ylabel('$\omega_\lambda$', fontsize=16)
            #axes[2,0].set_ylabel('$\omega_\phi$', fontsize=16)

            axes[0,0].set_title('Coarse $(>l)$')
            axes[0,1].set_title('Fine $(>l)$')
    
            plt.savefig(out_direct + '/{0:.4g}km/AVE_vorticity_dichotomies.png'.format(scale/1e3), dpi=dpi)
            plt.close()
