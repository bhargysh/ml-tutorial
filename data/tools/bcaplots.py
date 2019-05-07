import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker
import math


def hist(data, n_bins=10, plot_title='', xlabel='', ylabel='', 
         figsize=(12,8), xscale='linear', normalised=False, 
         tick_func=lambda x: x):
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = axs.hist(data, bins=n_bins, normed=normalised)

    # We'll color code by height, but you could use any scalar
    fracs = N.astype(float) / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    
    axs.set_title(plot_title)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_xscale(xscale)
    axs.xaxis.set_major_formatter(to_tick_func(tick_func))


    return fig


from matplotlib import cm 
from matplotlib import mlab

def hex_plot(x, y, z=None, 
             gridsize=40, 
             plot_title='', 
             xlabel='', 
             ylabel='', 
             figsize=(12,8), 
             tick_func=lambda x: "{:.2f}".format(x)):

    fig, axs = plt.subplots(1,1, figsize=figsize)

    plt.hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=None)
    plt.axis([x.min(), x.max(), y.min(), y.max()])

    cb = plt.colorbar()
    cb.set_label('Mean Value')
    
    # set new tick values

    cb_vals = [float(x.get_text()) for x in cb.ax.get_yticklabels()]
    new_vals = map(tick_func, cb_vals)
    cb.ax.set_yticklabels(new_vals)

    return fig

###################


def facet_scatter(df, xcol, ycol, facet, alpha=0.5, trendline=True, trend_order=1, ylim=None, tick_func=lambda x: "%s" % x):
    types = df[facet].unique()
    N = len(types)
    # Get a colour scheme for the plot
    colours = ['gray','lightblue','lightgreen', 'black', 'blue', 'green']
    
    # Create a figure
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    
    # Set the y axis range
    if ylim:
        ax.set_ylim(ylim)
    
    # map over the types
    for n, ptype in enumerate(types):
        s = df[df[facet] == ptype]
        x=s[xcol]
        y=s[ycol]
        ax.scatter(x, y, c=colours[n], alpha=alpha, label=ptype)
        
        if trendline:
            # calc the trendline
            z = np.polyfit(x, y, trend_order)
            p = np.poly1d(z)
            x_axis = np.linspace(x.min(), x.max(), 500)
            # The text for the line
            write = lambda c,i: "{:.6f}x^{}".format(c,i) if i !=0 else "{:.6f}".format(c)
            terms = [write(c,i) for i, c in enumerate(z[::-1])][::-1]
            line_eq = " + ".join(terms)
            
            ax.plot(x_axis,p(x_axis), '-', color=colours[N+n], label="trend {: <12} {}".format(ptype, line_eq))

    
    ax.yaxis.set_major_formatter(to_tick_func(tick_func))
    plt.legend(bbox_to_anchor=(1, 0.8), ncol=1)
    
########################

millnames = ['',' Thousand',' Million',' Billion',' Trillion']

def millify(n):
    """
    Formats large numbers into human readable strings
    """
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def to_tick_func(f):
    return ticker.FuncFormatter(lambda x, pos: f(x))