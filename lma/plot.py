"""
## lma.plot
Provides functions for visualizing motif analysis.

This module contains the following functions:

- `dfs_for_plotting` - Takes DataFrame from resample_trees functions and returns DataFrame for plotting.
- `make_cell_color_dict` - Returns cell color dictionary based on provided cell fates.
- `plot_frequency` - Displays frequency plot of `cutoff` number of subtrees in original dataset and all resamples.
- `plot_deviation` - Displays deviation plot of `cutoff` number of subtrees in original dataset and a subset of resamples.
"""
# +
# packages for both analysis and plotting
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import re

# packages for only plotting
import colorcet
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.patches as mpatches
from matplotlib.offsetbox import DrawingArea, AnnotationBbox
import matplotlib.font_manager as font_manager
from matplotlib.collections import PathCollection
from statsmodels.stats.multitest import multipletests
pyplot.rcParams['svg.fonttype'] = 'none'
mpl.rcParams.update({'font.size': 8})
mpl.rcParams['figure.dpi'] = 300


# -

def dfs_for_plotting(dfs_c, num_resamples, subtree_dict, cutoff='auto', num_null=1000):
    """Converts DataFrame from resample_trees functions into DataFrames for plotting.
    
    Calculates z-scores by comparing the observed count in the original trees to the mean/std across all resamples.
    Calculates null z-scores by comparing the observed count of `num_null` random resamples to the mean/std across the rest of 
    the resamples.
    
    Args:
        dfs_c (DataFrame): Indexed by values from `subtree_dict`.
            Last column is analytically solved expected count of each subtree.
            Second to last column is observed count of occurences in the original dataset.
            Rest of columns are the observed count of occurences in the resampled sets.
            Output from resample_trees functions.
        num_resamples (int): Number of resamples.
        subtree_dict (dict): Keys are subtrees, values are integers.
        cutoff (string or NoneType or int, optional): Take `cutoff` number of subtrees with largest absolute z-scores 
            to include in plots.
            If not provided explicitly, will be automatically determined to take all subtrees with abs z-score > 1.
            If NoneType, take all subtrees.
        num_null (int, optional): Take `num_null` number of resamples to calculate z-scores as part of null distribution.

    Returns:
        (tuple): Contains the following DataFrames.
        
        - df_true_melt_subset (DataFrame): DataFrame indexed by `cutoff` number of most significant subtrees for plotting.
            Sorted by z-score from most over-represented to most under-represented. Contains the following columns: 
                - subtree_val (int): Value corresponding to `subtree_dict`.
                - observed (float): Count in original trees.
                - expected (float): Analytically solved expected count.
                - z-score (float): Computed using observed values and mean/std across resamples.
                - abs z-score (float): Absolute value of z-score.
                - label (string): Key corresponding to `subtree_dict`.
                - null min (float): Minimum count across across all resamples.
                - null mean (float): Average count across across all resamples.
                - null max (float): Maximum count across across all resamples.
                - p_val (float): p-value, one-sided test, not corrected for multiple hypotheses testing.
                - adj_p_val_fdr_bh (float): adjusted p-value, corrected using the Benjamini and Hochberg FDR correction
                - adj_p_val_fdr_tsbh (float): adjusted p-value, corrected using the Benjamini and Hochberg FDR correction with two stage linear step-up procedure
                - null z-score min (float): Minimum z-score across across `num_null` random resamples.
                - null z-score mean (float): Average z-score across across `num_null` random resamples.
                - null z-score max (float): Maximum z-score across across `num_null` random resamples.
        - df_melt_subset (DataFrame): Melted DataFrame with observed count for `cutoff` number of most significant subtrees 
            across all resamples. Contains the following columns:
                - subtree_val (int): Value corresponding to `subtree_dict`.
                - observed (int): Counts across all resamples.
                - label (string): Key corresponding to `subtree_dict`.
        - df_melt_100resamples_subset (DataFrame): Melted DataFrame with observed count for `cutoff` number of most significant
            subtrees across 100 random resamples. Contains the following columns:
                - subtree_val (int): Value corresponding to `subtree_dict`.
                - observed (int): Counts across 100 random resamples.
                - label (string): Key corresponding to `subtree_dict`.
        - df_null_zscores_i_c_melt_subset (DataFrame): Melted DataFrame with null z-score for `cutoff` number of most significant
            subtrees across `num_null` random resamples. Contains the following columns:
                - subtree_val (int): Value corresponding to `subtree_dict`.
                - observed (float): Z-scores across `num_null` random resamples.
                - label (string): Key corresponding to `subtree_dict`.
        - df_null_zscores_i_c_melt_100resamples_subset (DataFrame): Melted DataFrame with null z-score for `cutoff` number of 
            most significant subtrees across 100 random resamples. Contains the following columns:
                - subtree_val (int): Value corresponding to `subtree_dict`.
                - observed (float): Z-scores across 100 random resamples.
                - label (string): Key corresponding to `subtree_dict`.
    """

    # slice out the subtrees of the original trees
    df_true_slice = dfs_c.loc[:,'observed']

    # dataframe of original trees
    data = {'subtree_val': df_true_slice.index,
            'observed': df_true_slice.values}
    df_true_melt = pd.DataFrame(data)

    # slice out the subtrees of the original trees
    expected = dfs_c.loc[:,'expected'].values

    # dataframe of resampled trees
    resamples = num_resamples - 1
    df_melt = pd.melt(dfs_c.loc[:,'0':f'{resamples}'].transpose(), var_name='subtree_val', value_name='observed')
    df_melt_100resamples = pd.melt(dfs_c.loc[:,'0':'99'].transpose(), var_name='subtree_val', value_name='observed')

    # calculate zscores
    zscores = []
    for i in df_true_slice.index:
        actual = df_true_slice[i]
        mean = np.mean(df_melt.loc[df_melt['subtree_val']==i]['observed'].values)
        std = np.std(df_melt.loc[df_melt['subtree_val']==i]['observed'].values)
        if std == 0:
            zscore = 0
        else:
            zscore = (actual - mean) / std
        zscores.append(zscore)

    # assign to dataframe and subset based on subtrees with top 10 significance values
    df_true_melt['expected'] = expected
    df_true_melt['z-score'] = zscores
    df_true_melt['abs z-score'] = abs(df_true_melt['z-score'])
    df_true_melt.fillna(0, inplace=True)
    df_true_melt.sort_values('abs z-score', axis=0, ascending=False, inplace=True)
    
    # subset based on the number of subtrees
    if cutoff == 'auto':
        cutoff = (df_true_melt['abs z-score'].values>1).sum()
        df_true_melt_subset = df_true_melt.iloc[:cutoff].copy()
    elif cutoff == None:
        df_true_melt_subset = df_true_melt
    else:
        df_true_melt_subset = df_true_melt.iloc[:cutoff].copy()
    
    df_true_melt_subset.sort_values('z-score', axis=0, ascending=False, inplace=True)
    df_true_melt_subset['label'] = [list(subtree_dict.keys())[i] for i in df_true_melt_subset['subtree_val'].values]

    # subset the resamples
    df_melt_subset_list = []
    for i in df_true_melt_subset['subtree_val']:
        df_melt_subtree = df_melt.loc[df_melt['subtree_val']==i].copy()
        df_melt_subtree['label']=list(subtree_dict.keys())[i]
        df_melt_subset_list.append(df_melt_subtree)
    df_melt_subset = pd.concat(df_melt_subset_list)
    
    df_melt_100resamples_subset_list = []
    for i in df_true_melt_subset['subtree_val']:
        df_melt_100resamples_subtree = df_melt_100resamples.loc[df_melt_100resamples['subtree_val']==i].copy()
        df_melt_100resamples_subtree['label']=list(subtree_dict.keys())[i]
        df_melt_100resamples_subset_list.append(df_melt_100resamples_subtree)
    df_melt_100resamples_subset = pd.concat(df_melt_100resamples_subset_list)

    df_true_melt_subset['null min'] = [df_melt_subset.groupby(['subtree_val']).min(numeric_only=True).loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    df_true_melt_subset['null mean'] = [df_melt_subset.groupby(['subtree_val']).mean(numeric_only=True).loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    df_true_melt_subset['null max'] = [df_melt_subset.groupby(['subtree_val']).max(numeric_only=True).loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    
    # calculate p-value (one-sided test)
    p_val_list = []
    for i, j in zip(df_true_melt_subset['subtree_val'].values, df_true_melt_subset['z-score'].values):
        resamples = dfs_c.iloc[i].values[:-1]
        actual = df_true_melt_subset.loc[df_true_melt_subset['subtree_val']==i]['observed'].values[0]
        if j > 0:
            pos = sum(resamples>=actual)
        elif j < 0:
            pos = sum(resamples<=actual)
        elif j == 0:
            pos=len(resamples)

        p_val = pos/len(resamples)
        p_val_list.append(p_val)

    df_true_melt_subset['p_val'] = p_val_list
    df_true_melt_subset['adj_p_val_fdr_bh'] = multipletests(p_val_list, method='fdr_bh')[1]
    df_true_melt_subset['adj_p_val_fdr_tsbh'] = multipletests(p_val_list, method='fdr_tsbh')[1]
    
    # calculate deviation of each resample
    df_null_zscores_i_list = []
    for i in tqdm(range(num_null)):
        df_true_slice_i = dfs_c[f'{i}'].copy()
        data = {'subtree_val': df_true_slice_i.index,
                'observed': df_true_slice_i.values}
        df_true_melt_i = pd.DataFrame(data)

        df_subset_i = dfs_c[dfs_c.columns[~dfs_c.columns.isin([f'{i}','observed', 'expected'])]].copy()
        df_melt_i = pd.melt(df_subset_i.transpose(), var_name='subtree_val', value_name='observed')

        zscores_i = []
        for j in df_true_slice_i.index:
            actual = df_true_slice_i[j]
            mean = np.mean(df_melt_i.loc[df_melt_i['subtree_val']==j]['observed'].values)
            std = np.std(df_melt_i.loc[df_melt_i['subtree_val']==j]['observed'].values)
            if std == 0:
                zscore = 0
            else:
                zscore = (actual - mean) / std
            zscores_i.append(zscore)

        df_null_zscores_i = pd.DataFrame(zscores_i, columns=[i])
        df_null_zscores_i_list.append(df_null_zscores_i)
        
    df_null_zscores_i_c = pd.concat(df_null_zscores_i_list, axis=1)
    df_null_zscores_i_c.fillna(0, inplace=True)
    
    df_null_zscores_i_c_melt = df_null_zscores_i_c.transpose().melt(var_name='subtree_val', value_name='observed')
    df_null_zscores_i_c_melt_100resamples = df_null_zscores_i_c.loc[:,:99].transpose().melt(var_name='subtree_val', value_name='observed')
    
    # subset the resamples
    df_null_zscores_i_c_melt_subset_list = []
    for i in df_true_melt_subset['subtree_val']:
        df_null_zscores_i_c_melt_subtree = df_null_zscores_i_c_melt.loc[df_null_zscores_i_c_melt['subtree_val']==i].copy()
        df_null_zscores_i_c_melt_subtree['label']=list(subtree_dict.keys())[i]
        df_null_zscores_i_c_melt_subset_list.append(df_null_zscores_i_c_melt_subtree)
    df_null_zscores_i_c_melt_subset = pd.concat(df_null_zscores_i_c_melt_subset_list)
    
    # subset the resamples
    df_null_zscores_i_c_melt_100resamples_subset_list = []
    for i in df_true_melt_subset['subtree_val']:
        df_null_zscores_i_c_melt_100resamples_subtree = df_null_zscores_i_c_melt_100resamples.loc[df_null_zscores_i_c_melt_100resamples['subtree_val']==i].copy()
        df_null_zscores_i_c_melt_100resamples_subtree['label']=list(subtree_dict.keys())[i]
        df_null_zscores_i_c_melt_100resamples_subset_list.append(df_null_zscores_i_c_melt_100resamples_subtree)
    df_null_zscores_i_c_melt_100resamples_subset = pd.concat(df_null_zscores_i_c_melt_100resamples_subset_list)
    
    df_true_melt_subset['null z-score min'] = [df_null_zscores_i_c_melt_subset.groupby(['subtree_val']).min(numeric_only=True).loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    df_true_melt_subset['null z-score mean'] = [df_null_zscores_i_c_melt_subset.groupby(['subtree_val']).mean(numeric_only=True).loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    df_true_melt_subset['null z-score max'] = [df_null_zscores_i_c_melt_subset.groupby(['subtree_val']).max(numeric_only=True).loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    
    return (df_true_melt_subset, df_melt_subset, df_melt_100resamples_subset, df_null_zscores_i_c_melt_subset, df_null_zscores_i_c_melt_100resamples_subset)

def make_color_dict(labels, colors):
    """Makes color dictionary based on provided labels (can be cell types or dataset names).
    
    If cell_fates not provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        - labels (list): List of string labels.
        - colors (list): List of string color codes.
    
    Returns:
        color_dict (dict): Keys are labels, values are colors.
    
    """
    color_dict = dict(zip(labels, colors))
    return color_dict


def _make_circle(color, size, x, y, alpha):
    da = DrawingArea(0, 0, 0, 0)
    p = mpatches.Circle((0, 0), size, color=color, alpha=alpha)
    da.add_artist(p)

    c1 = AnnotationBbox(da, 
                        (x,y),
                        xybox=(0, 0),
                        frameon=False,
                        xycoords=("data", "axes fraction"),
                        box_alignment=(0.5, 0.5),
                        boxcoords="offset points",
                        bboxprops={"edgecolor" : "none"},
                        pad=0)
    return c1

def _annot(number):
    if number < 0.0005:
        return '***'
    elif number < 0.005:
        return '**'
    elif number < 0.05:
        return '*'

def plot_frequency(subtree, 
                   df_true_melt_subset, 
                   df_melt_subset, 
                   df_melt_100resamples_subset, 
                   cell_color_dict,
                   fdr_type='fdr_tsbh',
                   cutoff='auto', 
                   title='auto',
                   multiple_datasets=False,
                   legend_bool=True, 
                   legend_pos='outside',
                   save=False, 
                   image_format='png',
                   dpi=300,
                   image_save_path=None):
    
    """Plots frequency of `cutoff` number of subtrees in original dataset and all resamples.
    
    Args:
        subtree (string): Type of subtree.
        df_true_melt_subset (DataFrame): DataFrame with `cutoff` number of most significant subtrees for plotting.
            Sorted by z-score from most over-represented to most under-represented.
            Output from `dfs_for_plotting` function.
        df_melt_subset (DataFrame): Melted DataFrame with observed count for `cutoff` number of most significant subtrees 
            across all resamples.
            Output from `dfs_for_plotting` function.
        df_melt_100resamples_subset (DataFrame): Melted DataFrame with observed count for `cutoff` number of most significant
            subtrees across 100 random resamples.
            Output from `dfs_for_plotting` function.
        cell_color_dict (dict): Keys are cell fates, values are colors.
        fdr_type (string, optional): Use the Benjamini and Hochberg FDR correction if 'fdr_bh', use Benjamini and Hochberg FDR correction
            with two stage linear step-up procedure if 'fdr_tsbh'. Uses 'fdr_tsbh' by default.
        cutoff (string or NoneType or int, optional): Take `cutoff` number of subtrees with largest absolute z-scores 
            to include in plots.
            If not provided explicitly, will be automatically determined to take all subtrees with abs z-score > 1.
            If NoneType, take all subtrees.
        title (string, optional): Title to use for plot. If not provided explicitly, will be automatically determined to read `subtree` frequency.
        multiple_datasets (bool, optional): Modify x-axis label depending if single or multiple datasets were used.
        legend_bool (bool, optional): Include legend in plot.
        legend_pos (string, optional): Position of legend (outside or inside).
        save (bool, optional): If True, save figure as file.
        image format (string, optional): Format of image file to be saved (png or svg).
        dpi (int, optional): Resolution of saved image file.
        image_save_path (string, optional): Path to saved image file.
    """

    df_true_melt_subset_sg = df_true_melt_subset.loc[df_true_melt_subset[f'adj_p_val_{fdr_type}']<0.05].copy()
    
    margins=0.05
    bbox_to_anchor=(0, 0)  
    figsize=(0.23*len(df_true_melt_subset)+margins, 2.5)

    sns.set_style('whitegrid')
    fig, ax = pyplot.subplots(figsize=figsize)
    pyplot.setp(ax.collections)

    sns.violinplot(x='label', 
                   y='observed', 
                   data=df_melt_subset, 
                   cut=0,
                   inner=None,
                   color='#BCBEC0',
                   scale='width',
                   linewidth=0,
                   )
    sns.stripplot(x='label', 
                  y='observed', 
                  data=df_melt_100resamples_subset, 
                  jitter=0.2,
                  color='gray',
                  size=0.5,
                 )
    pyplot.scatter(x='label', y='observed', data=df_true_melt_subset, color='red', label='Observed count', s=2.5)
    pyplot.scatter(x='label', y='null mean', data=df_true_melt_subset, color='gray', label='Count across all resamples', s=2.5)
    pyplot.scatter(x='label', y='expected', data=df_true_melt_subset, color='black', label='Expected count', s=2.5)
    pyplot.scatter(x='label', y='null min', data=df_true_melt_subset, color='gray', s=0, label='')
    pyplot.scatter(x='label', y='null max', data=df_true_melt_subset, color='gray', s=0, label='')
    pyplot.scatter(x='label', y='observed', data=df_true_melt_subset, color='red', label='', s=2.5)
    pyplot.scatter(x='label', y='observed', data=df_true_melt_subset_sg, color='red', s=25, alpha=0.35, label='Adjusted p-value < 0.05')

    # add annotations for adjusted p-value
    for label in df_true_melt_subset_sg['label'].values:
        adj_p_val = df_true_melt_subset_sg.loc[df_true_melt_subset_sg['label']==label][f'adj_p_val_{fdr_type}'].values[0]
        val = df_true_melt_subset_sg.loc[df_true_melt_subset_sg['label']==label]['observed'].values[0]
        null = df_true_melt_subset_sg.loc[df_true_melt_subset_sg['label']==label]['null mean'].values[0]
        if val > null:
            y_coord = val+max(df_true_melt_subset['observed'])/10
            pyplot.annotate(_annot(adj_p_val), xy=(label, y_coord), ha='center', va='bottom')
        else:
            y_coord = val-max(df_true_melt_subset['observed'])/10
            pyplot.annotate(_annot(adj_p_val), xy=(label, y_coord), ha='center', va='top')

    pyplot.margins(x=0.05, y=0.15)
    pyplot.grid(True)
    ax.set_xticklabels([])

    if title == 'auto':
        pyplot.title(f'{subtree.capitalize()} frequency', y=1.02, **{'fontname':'Arial', 'size':8}, fontweight='bold')
    else:
        pyplot.title(f'{title}', y=1.02, **{'fontname':'Arial', 'size':8}, fontweight='bold')
    pyplot.ylabel('Counts', **{'fontname':'Arial', 'size':8})
    pyplot.yticks(**{'fontname':'Arial', 'size':8})

    if legend_bool == True:
        legend_props = font_manager.FontProperties(family='Arial', style='normal', size=6)
        if legend_pos == 'outside':
            pyplot.legend(loc='upper left', framealpha=1, prop=legend_props, bbox_to_anchor=(1.05,1.0))
        elif legend_pos == 'inside':
            pyplot.legend(loc='upper right', framealpha=1, prop=legend_props)

    for i, artist in enumerate(ax.findobj(PathCollection)):
        artist.set_zorder(1)

    if subtree == 'doublet':   
        for i in range(len(df_true_melt_subset['label'].values)):
            c1_str = df_true_melt_subset['label'].values[i][1]
            c2_str = df_true_melt_subset['label'].values[i][3]

            x = i
            y = -0.06
            ax.add_artist(_make_circle(cell_color_dict[c1_str], 4.5, x, y, 0.4))
            ax.annotate(c1_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})

            x = i
            y = -0.15
            ax.add_artist(_make_circle(cell_color_dict[c2_str], 4.5, x, y, 0.4))
            ax.annotate(c2_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})  
        if cutoff==None:
            pyplot.xlabel(f'All {subtree} combinations', labelpad=22.5, **{'fontname':'Arial', 'size':8})
        else:
            if multiple_datasets == False:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=22.5, **{'fontname':'Arial', 'size':8})
            else:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score across all datasets)', labelpad=22.5, **{'fontname':'Arial', 'size':8})
    
    if subtree == 'triplet':
        for i in range(len(df_true_melt_subset['label'].values)):
            c1_str = df_true_melt_subset['label'].values[i][1]
            c2_str = df_true_melt_subset['label'].values[i][4]
            c3_str = df_true_melt_subset['label'].values[i][6]

            x = i
            y = -0.06
            ax.add_artist(_make_circle(cell_color_dict[c1_str], 4.5, x, y, 0.4))
            ax.annotate(c1_str, 
                        xy=(x, y), 
                        verticalalignment='center', 
                        horizontalalignment='center',
                        annotation_clip=False, 
                        xycoords=('data', 'axes fraction'),
                        **{'fontname':'Arial', 'size':8})

            x = i
            y = -0.18
            ax.add_artist(_make_circle(cell_color_dict[c2_str], 4.5, x, y, 0.4))
            ax.annotate(c2_str, 
                        xy=(x, y), 
                        verticalalignment='center', 
                        horizontalalignment='center',
                        annotation_clip=False, 
                        xycoords=('data', 'axes fraction'),
                        **{'fontname':'Arial', 'size':8})   

            x = i
            y = -0.27
            ax.add_artist(_make_circle(cell_color_dict[c3_str], 4.5, x, y, 0.4))
            ax.annotate(c3_str, 
                        xy=(x, y), 
                        verticalalignment='center', 
                        horizontalalignment='center',
                        annotation_clip=False, 
                        xycoords=('data', 'axes fraction'),
                        **{'fontname':'Arial', 'size':8}) 
            
        if cutoff==None:
            pyplot.xlabel(f'All {subtree} combinations', labelpad=40, **{'fontname':'Arial', 'size':8})
        else:
            if multiple_datasets == False:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=40, **{'fontname':'Arial', 'size':8})
            else:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score across all datasets)', labelpad=40, **{'fontname':'Arial', 'size':8})
    

    if subtree == 'quartet':
        for i in range(len(df_true_melt_subset['label'].values)):
            c1_str = df_true_melt_subset['label'].values[i][2]
            c2_str = df_true_melt_subset['label'].values[i][4]
            c3_str = df_true_melt_subset['label'].values[i][8]
            c4_str = df_true_melt_subset['label'].values[i][10]

            x = i
            y = -0.06
            ax.add_artist(_make_circle(cell_color_dict[c1_str], 4.5, x, y, 0.4))
            ax.annotate(c1_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})

            x = i
            y = -0.15
            ax.add_artist(_make_circle(cell_color_dict[c2_str], 4.5, x, y, 0.4))
            ax.annotate(c2_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})   

            x = i
            y = -0.27
            ax.add_artist(_make_circle(cell_color_dict[c3_str], 4.5, x, y, 0.4))
            ax.annotate(c3_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})  

            x = i
            y = -0.36
            ax.add_artist(_make_circle(cell_color_dict[c4_str], 4.5, x, y, 0.4))
            ax.annotate(c4_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})  
            
        if cutoff==None:
            pyplot.xlabel(f'All {subtree} combinations', labelpad=52.5, **{'fontname':'Arial', 'size':8})
        else:
            if multiple_datasets == False:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=52.5, **{'fontname':'Arial', 'size':8})
            else:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score across all datasets)', labelpad=52.5, **{'fontname':'Arial', 'size':8})
            
    if save==True:
        pyplot.savefig(f"{image_save_path}.{image_format}", dpi=dpi, bbox_inches="tight")

def plot_deviation(subtree, 
                   df_true_melt_subset, 
                   df_null_zscores_i_c_melt_subset, 
                   df_null_zscores_i_c_melt_100resamples_subset, 
                   cell_color_dict,
                   fdr_type='fdr_tsbh',
                   cutoff='auto', 
                   title='auto',
                   multiple_datasets=False,
                   num_null=1000,
                   legend_bool=True,
                   legend_pos='outside',
                   save=False, 
                   image_format='png',
                   dpi=300,
                   image_save_path=None):
    
    """Plots deviation of `cutoff` number of subtrees in original dataset and `num_null` resamples.
    
    Args:
        subtree (string): Type of subtree.
        df_true_melt_subset (DataFrame): DataFrame with cutoff number of most significant subtrees for plotting.
            Sorted by z-score from most over-represented to most under-represented.
            Output from `dfs_for_plotting` function.
        df_null_zscores_i_c_melt_subset (DataFrame): Melted DataFrame with null z-score for `cutoff` number of most significant
            subtrees across `num_null` random resamples.
            Output from `dfs_for_plotting` function.
        df_null_zscores_i_c_melt_100resamples_subset (DataFrame): Melted DataFrame with null z-score for `cutoff` number of 
            most significant subtrees across 100 random resamples.
            Output from `dfs_for_plotting` function.
        cell_color_dict (dict): Keys are cell fates, values are colors.
        fdr_type (string, optional): Use the Benjamini and Hochberg FDR correction if 'fdr_bh', use Benjamini and Hochberg FDR correction
            with two stage linear step-up procedure if 'fdr_tsbh'. Uses 'fdr_tsbh' by default.
        cutoff (string or NoneType or int, optional): Take `cutoff` number of subtrees with largest absolute z-scores 
            to include in plots.
            If not provided explicitly, will be automatically determined to take all subtrees with abs z-score > 1.
            If NoneType, take all subtrees.
        title (string, optional): Title to use for plot. If not provided explicitly, will be automatically determined to read `subtree` frequency.
        multiple_datasets (bool, optional): Modify x-axis label depending if single or multiple datasets were used.
        num_null (int, optional): Number of resamples used to calculate z-scores as part of null distribution.    
        legend_bool (bool, optional): Include legend in plot.
        legend_pos (string, optional): Position of legend (outside or inside).
        save (bool, optional): If True, save figure as file.
        image format (string, optional): Format of image file to be saved (png or svg).
        dpi (int, optional): Resolution of saved image file.
        image_save_path (string, optional): Path to saved image file.
    """

    df_true_melt_subset_sg = df_true_melt_subset.loc[df_true_melt_subset[f'adj_p_val_{fdr_type}']<0.05].copy()
    
    margins=0.05
    bbox_to_anchor=(0, 0)  
    figsize=(0.23*len(df_true_melt_subset)+margins, 2.5)

    sns.set_style('whitegrid')
    fig, ax = pyplot.subplots(figsize=figsize)
    pyplot.setp(ax.collections)

    sns.violinplot(x='label', 
                   y='observed', 
                   data=df_null_zscores_i_c_melt_subset, 
                   cut=0,
                   inner=None,
                   color='#BCBEC0',
                   scale='width',
                   linewidth=0,
                   )
    sns.stripplot(x='label', 
                  y='observed', 
                  data=df_null_zscores_i_c_melt_100resamples_subset, 
                  jitter=0.2,
                  color='gray',
                  size=0.5,
                 )
    pyplot.scatter(x="label", y="z-score", data=df_true_melt_subset, color='red', label='Observed count', s=2.5)
    pyplot.scatter(x="label", y="null z-score mean", data=df_true_melt_subset, color='gray', label='Resampled datasets', s=2.5)
    pyplot.scatter(x="label", y="null z-score mean", data=df_true_melt_subset, color='black', label=f'Average across {num_null} resamples', s=2.5)
    pyplot.scatter(x="label", y="null z-score min", data=df_true_melt_subset, color='gray', s=0, label='')
    pyplot.scatter(x="label", y="null z-score max", data=df_true_melt_subset, color='gray', s=0, label='')
    pyplot.scatter(x="label", y="z-score", data=df_true_melt_subset, color='red', label='', s=2.5)
    pyplot.scatter(x="label", y="z-score", data=df_true_melt_subset_sg, color='red', s=25, alpha=0.35, label='Adjusted p-value < 0.05')

    # add annotations for adjusted p-value
    for label in df_true_melt_subset_sg['label'].values:
        adj_p_val = df_true_melt_subset_sg.loc[df_true_melt_subset_sg['label']==label][f'adj_p_val_{fdr_type}'].values[0]
        val = df_true_melt_subset_sg.loc[df_true_melt_subset_sg['label']==label]['z-score'].values[0]
        null = df_true_melt_subset_sg.loc[df_true_melt_subset_sg['label']==label]['null z-score mean'].values[0]
        if val > null:
            y_coord = val+max(df_true_melt_subset['z-score'])/10
            pyplot.annotate(_annot(adj_p_val), xy=(label, y_coord), ha='center', va='bottom')
        else:
            y_coord = val-max(df_true_melt_subset['z-score'])/10
            pyplot.annotate(_annot(adj_p_val), xy=(label, y_coord), ha='center', va='top')


    pyplot.margins(x=0.05, y=0.15)
    pyplot.grid(True)
    ax.set_xticklabels([])

    if title == 'auto':
        pyplot.title('Deviation from resamples', y=1.02, **{'fontname':'Arial', 'size':8}, fontweight='bold')
    else:
        pyplot.title(f'{title}', y=1.02, **{'fontname':'Arial', 'size':8}, fontweight='bold')
    pyplot.ylabel('z-score', **{'fontname':'Arial', 'size':8})
    pyplot.yticks(**{'fontname':'Arial', 'size':8})

    if legend_bool == True:
        legend_props = font_manager.FontProperties(family='Arial', style='normal', size=6)
        if legend_pos == 'outside':
            pyplot.legend(loc='upper left', framealpha=1, prop=legend_props, bbox_to_anchor=(1.05,1.0))
        elif legend_pos == 'inside':
            pyplot.legend(loc='upper right', framealpha=1, prop=legend_props)
    for i, artist in enumerate(ax.findobj(PathCollection)):
        artist.set_zorder(1)

    if subtree == 'doublet':   
        for i in range(len(df_true_melt_subset['label'].values)):
            c1_str = df_true_melt_subset['label'].values[i][1]
            c2_str = df_true_melt_subset['label'].values[i][3]

            x = i
            y = -0.06
            ax.add_artist(_make_circle(cell_color_dict[c1_str], 4.5, x, y, 0.4))
            ax.annotate(c1_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})

            x = i
            y = -0.15
            ax.add_artist(_make_circle(cell_color_dict[c2_str], 4.5, x, y, 0.4))
            ax.annotate(c2_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})  
        if cutoff==None:
            pyplot.xlabel(f'All {subtree} combinations', labelpad=22.5, **{'fontname':'Arial', 'size':8})
        else:
            if multiple_datasets == False:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=22.5, **{'fontname':'Arial', 'size':8})
            else:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score across all datasets)', labelpad=22.5, **{'fontname':'Arial', 'size':8})
    
    if subtree == 'triplet':
        for i in range(len(df_true_melt_subset['label'].values)):
            c1_str = df_true_melt_subset['label'].values[i][1]
            c2_str = df_true_melt_subset['label'].values[i][4]
            c3_str = df_true_melt_subset['label'].values[i][6]

            x = i
            y = -0.06
            ax.add_artist(_make_circle(cell_color_dict[c1_str], 4.5, x, y, 0.4))
            ax.annotate(c1_str, 
                        xy=(x, y), 
                        verticalalignment='center', 
                        horizontalalignment='center',
                        annotation_clip=False, 
                        xycoords=('data', 'axes fraction'),
                        **{'fontname':'Arial', 'size':8})

            x = i
            y = -0.18
            ax.add_artist(_make_circle(cell_color_dict[c2_str], 4.5, x, y, 0.4))
            ax.annotate(c2_str, 
                        xy=(x, y), 
                        verticalalignment='center', 
                        horizontalalignment='center',
                        annotation_clip=False, 
                        xycoords=('data', 'axes fraction'),
                        **{'fontname':'Arial', 'size':8})   

            x = i
            y = -0.27
            ax.add_artist(_make_circle(cell_color_dict[c3_str], 4.5, x, y, 0.4))
            ax.annotate(c3_str, 
                        xy=(x, y), 
                        verticalalignment='center', 
                        horizontalalignment='center',
                        annotation_clip=False, 
                        xycoords=('data', 'axes fraction'),
                        **{'fontname':'Arial', 'size':8}) 
            
        if cutoff==None:
            pyplot.xlabel(f'All {subtree} combinations', labelpad=40, **{'fontname':'Arial', 'size':8})
        else:
            if multiple_datasets == False:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=40, **{'fontname':'Arial', 'size':8})
            else:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score across all datasets)', labelpad=40, **{'fontname':'Arial', 'size':8})
    

    if subtree == 'quartet':
        for i in range(len(df_true_melt_subset['label'].values)):
            c1_str = df_true_melt_subset['label'].values[i][2]
            c2_str = df_true_melt_subset['label'].values[i][4]
            c3_str = df_true_melt_subset['label'].values[i][8]
            c4_str = df_true_melt_subset['label'].values[i][10]

            x = i
            y = -0.06
            ax.add_artist(_make_circle(cell_color_dict[c1_str], 4.5, x, y, 0.4))
            ax.annotate(c1_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})

            x = i
            y = -0.15
            ax.add_artist(_make_circle(cell_color_dict[c2_str], 4.5, x, y, 0.4))
            ax.annotate(c2_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})   

            x = i
            y = -0.27
            ax.add_artist(_make_circle(cell_color_dict[c3_str], 4.5, x, y, 0.4))
            ax.annotate(c3_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})  

            x = i
            y = -0.36
            ax.add_artist(_make_circle(cell_color_dict[c4_str], 4.5, x, y, 0.4))
            ax.annotate(c4_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})  
            
        if cutoff==None:
            pyplot.xlabel(f'All {subtree} combinations', labelpad=52.5, **{'fontname':'Arial', 'size':8})
        else:
            if multiple_datasets == False:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=52.5, **{'fontname':'Arial', 'size':8})
            else:
                pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score across all datasets)', labelpad=52.5, **{'fontname':'Arial', 'size':8})
            
    if save==True:
        pyplot.savefig(f"{image_save_path}.{image_format}", dpi=dpi, bbox_inches="tight")

def multi_dataset_dfs_for_plotting(dfs_dataset_c, 
                                   dataset_names, 
                                   num_resamples, 
                                   subtree_dict, 
                                   cutoff='auto', 
                                   num_null=1000):
    """Converts DataFrame from `multi_dataset_resample_trees` function into DataFrames for plotting.
    
    Calculates z-scores by comparing the observed count in the original trees to the mean/std across all resamples.
    Calculates null z-scores by comparing the observed count of `num_null` random resamples to the mean/std across the rest of 
    the resamples.
    
    Args:
        dfs_dataset_c (list): List where each entry is a DataFrame with the following characteristics.
            Indexed by values from `subtree_dict`.
            Last column is dataset label.
            Second to last column is analytically solved expected count of each subtree.
            Third to last column is observed count of occurences in the original dataset.
            Rest of columns are the observed count of occurences in the resampled sets.
            Output from `multi_dataset_resample_trees` function.
        dataset_names (list): List where each entry is a string representing the dataset label. 
        num_resamples (int): Number of resamples.
        subtree_dict (dict): Keys are subtrees, values are integers.
        cutoff (string or NoneType or int, optional): Takes `cutoff` number of subtrees with largest absolute z-scores 
            across all datasets to include in plots.
            If not provided explicitly, will be automatically determined to take all subtrees with abs z-score > 1
                in at least one of the datasets provided.
            If NoneType, take all subtrees.
        num_null (int, optional): Takes `num_null` number of resamples to calculate z-scores as part of null distribution.

    Returns:
        (tuple): Contains the following DataFrames.

        - df_true_melt_dataset_label_c_c (DataFrame): DataFrame indexed by `cutoff` number of most significant subtrees for plotting.
            Sorted by z-score from most over-represented to most under-represented (using the most extreme z-score
            for each subtree across all datasets provided). Contains the following columns:
                - subtree_val (int): Value corresponding to `subtree_dict`.
                - observed (float): Count in original trees.
                - expected (float): Analytically solved expected count.
                - z-score (float): Computed using observed values and mean/std across resamples.
                - abs z-score (float): Absolute value of z-score.
                - label (string): Key corresponding to `subtree_dict`.
                - null min (float): Minimum count across across all resamples.
                - null mean (float): Average count across across all resamples.
                - null max (float): Maximum count across across all resamples.
                - p_val (float): p-value, one-sided test, not corrected for multiple hypotheses testing.
                - adj_p_val_fdr_bh (float): adjusted p-value, corrected using the Benjamini and Hochberg FDR correction
                - adj_p_val_fdr_tsbh (float): adjusted p-value, corrected using the Benjamini and Hochberg FDR correction with two stage linear step-up procedure
                - dataset (string): Dataset label.
                - null z-score min (float): Minimum z-score across across `num_null` random resamples.
                - null z-score mean (float): Average z-score across across `num_null` random resamples.
                - null z-score max (float): Maximum z-score across across `num_null` random resamples.
        - df_melt_subset_c_c (DataFrame): Melted DataFrame with observed count for `cutoff` number of most significant subtrees 
            across all resamples. Contains the following columns:
                - subtree_val (int): Value corresponding to `subtree_dict`.
                - observed (int): Counts across all resamples.
                - label (string): Key corresponding to `subtree_dict`.
                - dataset (string): Dataset label.
        - df_melt_100resamples_subset_c_c (DataFrame): Melted DataFrame with observed count for `cutoff` number of most significant
            subtrees across 100 random resamples. Contains the following columns:
                - subtree_val (int): Value corresponding to `subtree_dict`.
                - observed (int): Counts across 100 random resamples.
                - label (string): Key corresponding to `subtree_dict`.
                - dataset (string): Dataset label.
        - df_null_zscores_i_c_melt_subset_c_c (DataFrame): Melted DataFrame with null z-score for `cutoff` number of most significant
            subtrees across `num_null` random resamples. Contains the following columns:
                - subtree_val (int): Value corresponding to `subtree_dict`.
                - observed (float): Z-scores across `num_null` random resamples.
                - label (string): Key corresponding to `subtree_dict`.
                - dataset (string): Dataset label.
        - df_null_zscores_i_c_melt_100resamples_subset_c_c (DataFrame): Melted DataFrame with null z-score for `cutoff` number of 
            most significant subtrees across 100 random resamples. Contains the following columns:
                - subtree_val (int): Value corresponding to `subtree_dict`.
                - observed (float): Z-scores across 100 random resamples.
                - label (string): Key corresponding to `subtree_dict`.
                - dataset (string): Dataset label.
    """
    df_melt_list = []
    df_melt_100resamples_list = []
    df_true_melt_list = []
    df_null_zscores_i_c_melt_list = []
    df_null_zscores_i_c_melt_100resamples_list = []
    
    for index, dataset_name in enumerate(dataset_names):
        
        dfs_c = dfs_dataset_c.loc[dfs_dataset_c['dataset']==dataset_name]
    
        # slice out the triplets of the original trees
        df_true_slice = dfs_c.loc[:,'observed']

        # dataframe of original trees
        data = {'subtree_val': df_true_slice.index,
                'observed': df_true_slice.values}
        df_true_melt = pd.DataFrame(data)

        # slice out the triplets of the original trees
        expected = dfs_c.loc[:,'expected'].values

        # dataframe of resampled trees
        resamples = num_resamples - 1
        df_melt = pd.melt(dfs_c.loc[:,'0':f'{resamples}'].transpose(), var_name='subtree_val', value_name='observed')
        df_melt_100resamples = pd.melt(dfs_c.loc[:,'0':'99'].transpose(), var_name='subtree_val', value_name='observed')

        df_melt_list.append(df_melt)
        df_melt_100resamples_list.append(df_melt_100resamples)
        
        # calculate zscores
        zscores = []
        for i in df_true_slice.index:
            actual = df_true_slice[i]
            mean = np.mean(df_melt.loc[df_melt['subtree_val']==i]['observed'].values)
            std = np.std(df_melt.loc[df_melt['subtree_val']==i]['observed'].values)
            if std == 0:
                zscore = 0
            else:
                zscore = (actual - mean) / std
            zscores.append(zscore)

        # assign to dataframe and subset based on subtrees with top 10 significance values
        df_true_melt['expected'] = expected
        df_true_melt['z-score'] = zscores
        df_true_melt['abs z-score'] = abs(df_true_melt['z-score'])
        df_true_melt.fillna(0, inplace=True)
        df_true_melt.sort_values('abs z-score', axis=0, ascending=False, inplace=True)
        df_true_melt['label'] = [list(subtree_dict.keys())[i] for i in df_true_melt['subtree_val'].values]
        df_true_melt['null min'] = [df_melt.groupby(['subtree_val']).min(numeric_only=True).loc[i].values[0] for i in df_true_melt['subtree_val']]
        df_true_melt['null mean'] = [df_melt.groupby(['subtree_val']).mean(numeric_only=True).loc[i].values[0] for i in df_true_melt['subtree_val']]
        df_true_melt['null max'] = [df_melt.groupby(['subtree_val']).max(numeric_only=True).loc[i].values[0] for i in df_true_melt['subtree_val']]
        
        # calculate p-value (one-sided test)
        p_val_list = []
        for i, j in zip(df_true_melt['subtree_val'].values, df_true_melt['z-score'].values):
            resamples = dfs_c.iloc[i].values[:-1]
            actual = df_true_melt.loc[df_true_melt['subtree_val']==i]['observed'].values[0]
            if j > 0:
                pos = sum(resamples>=actual)
            elif j < 0:
                pos = sum(resamples<=actual)
            elif j == 0:
                pos=len(resamples)
            p_val = pos/len(resamples)
            p_val_list.append(p_val)
        df_true_melt['p_val'] = p_val_list
        df_true_melt['adj_p_val_fdr_bh'] = multipletests(p_val_list, method='fdr_bh')[1]
        df_true_melt['adj_p_val_fdr_tsbh'] = multipletests(p_val_list, method='fdr_tsbh')[1]
        df_true_melt['dataset'] = dataset_names[index]
        df_true_melt_list.append(df_true_melt)
        
        # calculate deviation of each resample
        df_null_zscores_i_list = []
        for i in tqdm(range(num_null)):
            df_true_slice_i = dfs_c[f'{i}'].copy()
            data = {'subtree_val': df_true_slice_i.index,
                    'observed': df_true_slice_i.values}
            df_true_melt_i = pd.DataFrame(data)

            df_subset_i = dfs_c[dfs_c.columns[~dfs_c.columns.isin([f'{i}', 'observed', 'expected', 'dataset'])]].copy()
            df_melt_i = pd.melt(df_subset_i.transpose(), var_name='subtree_val', value_name='observed')

            zscores_i = []
            for j in df_true_slice_i.index:
                actual = df_true_slice_i[j]
                mean = np.mean(df_melt_i.loc[df_melt_i['subtree_val']==j]['observed'].values)
                std = np.std(df_melt_i.loc[df_melt_i['subtree_val']==j]['observed'].values)
                if std == 0:
                    zscore = 0
                else:
                    zscore = (actual - mean) / std
                zscores_i.append(zscore)

            df_null_zscores_i = pd.DataFrame(zscores_i, columns=[i])
            df_null_zscores_i_list.append(df_null_zscores_i)

        df_null_zscores_i_c = pd.concat(df_null_zscores_i_list, axis=1)
        df_null_zscores_i_c.fillna(0, inplace=True)

        df_null_zscores_i_c_melt = df_null_zscores_i_c.transpose().melt(var_name='subtree_val', value_name='observed')
        df_null_zscores_i_c_melt_100resamples = df_null_zscores_i_c.loc[:,:99].transpose().melt(var_name='subtree_val', value_name='observed')
        
        df_null_zscores_i_c_melt_list.append(df_null_zscores_i_c_melt)
        df_null_zscores_i_c_melt_100resamples_list.append(df_null_zscores_i_c_melt_100resamples)

        df_true_melt['null z-score min'] = [df_null_zscores_i_c_melt.groupby(['subtree_val']).min(numeric_only=True).loc[i].values[0] for i in df_true_melt['subtree_val']]
        df_true_melt['null z-score mean'] = [df_null_zscores_i_c_melt.groupby(['subtree_val']).mean(numeric_only=True).loc[i].values[0] for i in df_true_melt['subtree_val']]
        df_true_melt['null z-score max'] = [df_null_zscores_i_c_melt.groupby(['subtree_val']).max(numeric_only=True).loc[i].values[0] for i in df_true_melt['subtree_val']]

    df_true_melt_c = pd.concat(df_true_melt_list)
    
    # Loop through each subtree and take the highest absolute z-score across all datasets
    df_true_melt_c_label_list = []
    for i in subtree_dict.keys():
        df_true_melt_c_label = df_true_melt_c.loc[df_true_melt_c['label']==i].copy()
        if len(df_true_melt_c_label) == 0:
            continue
        df_true_melt_c_label.sort_values('abs z-score', axis=0, ascending=False, inplace=True)
        df_true_melt_c_label = df_true_melt_c_label.iloc[[0]].copy()
        df_true_melt_c_label_list.append(df_true_melt_c_label)

    df_true_melt_c_label_c = pd.concat(df_true_melt_c_label_list)
    df_true_melt_c_label_c.sort_values('abs z-score', axis=0, ascending=False, inplace=True)
    
    # Subset based on the cutoff number of subtrees
    if cutoff == 'auto':
        cutoff = (df_true_melt_c_label_c['abs z-score'].values>1).sum()
        df_true_melt_c_label_c_subset = df_true_melt_c_label_c.iloc[:cutoff].copy()
    elif cutoff == None:
        df_true_melt_c_label_c_subset = df_true_melt_c_label_c.copy()
    else:
        df_true_melt_c_label_c_subset = df_true_melt_c_label_c.iloc[:cutoff].copy()

    df_true_melt_c_label_c_subset.sort_values('z-score', axis=0, ascending=False, inplace=True)
    
    # Subset the z-score DataFrame based on the cutoff number of subtrees in the DataFrames for each dataset
    df_true_melt_dataset_label_c_list = []
    for dataset in dataset_names:
        df_true_melt_dataset = df_true_melt_c.loc[df_true_melt_c['dataset']==dataset]
        df_true_melt_dataset_label_list = []
        for i in df_true_melt_c_label_c_subset['subtree_val']:
            df_true_melt_dataset_label = df_true_melt_dataset.loc[df_true_melt_dataset['subtree_val']==i]
            df_true_melt_dataset_label_list.append(df_true_melt_dataset_label)
        df_true_melt_dataset_label_c = pd.concat(df_true_melt_dataset_label_list)
        df_true_melt_dataset_label_c_list.append(df_true_melt_dataset_label_c)
    df_true_melt_dataset_label_c_c = pd.concat(df_true_melt_dataset_label_c_list)
    
    # Subset the melted DataFrames based on the cutoff number of subtrees in the DataFrames for each dataset
    df_melt_subset_c_list = []
    df_melt_100resamples_subset_c_list = []
    df_null_zscores_i_c_melt_subset_c_list = []
    df_null_zscores_i_c_melt_100resamples_subset_c_list = []
    for index, (df_melt, 
                df_melt_100resamples, 
                df_null_zscores_i_c_melt, 
                df_null_zscores_i_c_melt_100resamples) in enumerate(zip(df_melt_list, 
                                                                        df_melt_100resamples_list,
                                                                        df_null_zscores_i_c_melt_list, 
                                                                        df_null_zscores_i_c_melt_100resamples_list)):
        df_melt_subset_list = []
        for i in df_true_melt_c_label_c_subset['subtree_val']:
            df_melt_subtree = df_melt.loc[df_melt['subtree_val']==i].copy()
            df_melt_subtree['label']=list(subtree_dict.keys())[i]
            df_melt_subset_list.append(df_melt_subtree)
        df_melt_subset_c = pd.concat(df_melt_subset_list)
        df_melt_subset_c['dataset'] = dataset_names[index]
        df_melt_subset_c_list.append(df_melt_subset_c)

        df_melt_100resamples_subset_list = []
        for i in df_true_melt_c_label_c_subset['subtree_val']:
            df_melt_100resamples_subtree = df_melt_100resamples.loc[df_melt_100resamples['subtree_val']==i].copy()
            df_melt_100resamples_subtree['label']=list(subtree_dict.keys())[i]
            df_melt_100resamples_subset_list.append(df_melt_100resamples_subtree)
        df_melt_100resamples_subset_c = pd.concat(df_melt_100resamples_subset_list)
        df_melt_100resamples_subset_c['dataset'] = dataset_names[index]
        df_melt_100resamples_subset_c_list.append(df_melt_100resamples_subset_c)
        
        df_null_zscores_i_c_melt_subset_list = []
        for i in df_true_melt_c_label_c_subset['subtree_val']:
            df_null_zscores_i_c_melt_subtree = df_null_zscores_i_c_melt.loc[df_null_zscores_i_c_melt['subtree_val']==i].copy()
            df_null_zscores_i_c_melt_subtree['label']=list(subtree_dict.keys())[i]
            df_null_zscores_i_c_melt_subset_list.append(df_null_zscores_i_c_melt_subtree)
        df_null_zscores_i_c_melt_subset_c = pd.concat(df_null_zscores_i_c_melt_subset_list)
        df_null_zscores_i_c_melt_subset_c['dataset'] = dataset_names[index]
        df_null_zscores_i_c_melt_subset_c_list.append(df_null_zscores_i_c_melt_subset_c)
        
        df_null_zscores_i_c_melt_100resamples_subset_list = []
        for i in df_true_melt_c_label_c_subset['subtree_val']:
            df_null_zscores_i_c_melt_100resamples_subtree = df_null_zscores_i_c_melt_100resamples.loc[df_null_zscores_i_c_melt_100resamples['subtree_val']==i].copy()
            df_null_zscores_i_c_melt_100resamples_subtree['label']=list(subtree_dict.keys())[i]
            df_null_zscores_i_c_melt_100resamples_subset_list.append(df_null_zscores_i_c_melt_100resamples_subtree)
        df_null_zscores_i_c_melt_100resamples_subset_c = pd.concat(df_null_zscores_i_c_melt_100resamples_subset_list)
        df_null_zscores_i_c_melt_100resamples_subset_c['dataset'] = dataset_names[index]
        df_null_zscores_i_c_melt_100resamples_subset_c_list.append(df_null_zscores_i_c_melt_100resamples_subset_c)

    df_melt_subset_c_c = pd.concat(df_melt_subset_c_list)
    df_melt_100resamples_subset_c_c = pd.concat(df_melt_100resamples_subset_c_list)
    df_null_zscores_i_c_melt_subset_c_c = pd.concat(df_null_zscores_i_c_melt_subset_c_list)
    df_null_zscores_i_c_melt_100resamples_subset_c_c = pd.concat(df_null_zscores_i_c_melt_100resamples_subset_c_list)
        
    return (df_true_melt_dataset_label_c_c,
            df_melt_subset_c_c, 
            df_melt_100resamples_subset_c_c,
            df_null_zscores_i_c_melt_subset_c_c,
            df_null_zscores_i_c_melt_100resamples_subset_c_c)

def multi_dataset_plot_deviation(subtree, 
                                 dataset_names,
                                 df_true_melt_dataset_label_c_c, 
                                 dataset_color_dict,
                                 cell_color_dict,
                                 cutoff='auto',
                                 title='auto',
                                 legend_bool=True,
                                 legend_pos='outside',
                                 save=False, 
                                 image_format='png',
                                 dpi=300,
                                 image_save_path=None):
    
    """Plots deviation of `cutoff` number of subtrees in multiple datasets.
    
    Args:
        subtree (string): Type of subtree.
        dataset_names (list): List where each entry is a string representing the dataset label. 
        df_true_melt_dataset_label_c_c (DataFrame): DataFrame with cutoff number of most significant subtrees for plotting.
            Sorted by z-score from most over-represented to most under-represented.
            Output from `multi_dataset_dfs_for_plotting` function.
        dataset_color_dict (dict): Keys are dataset names, values are colors.
        cell_color_dict (dict): Keys are cell fates, values are colors.
        cutoff (string or NoneType or int, optional): Take `cutoff` number of subtrees with largest absolute z-scores 
            to include in plots.
            If not provided explicitly, will be automatically determined to take all subtrees with abs z-score > 1.
            If NoneType, take all subtrees.
        title (string, optional): Title to use for plot. If not provided explicitly, will be automatically determined to read `subtree` frequency.
        legend_bool (bool, optional): Include legend in plot.
        legend_pos (string, optional): Position of legend (outside or inside).
        save (bool, optional): If True, save figure as file.
        image format (string, optional): Format of image file to be saved (png or svg).
        dpi (int, optional): Resolution of saved image file.
        image_save_path (string, optional): Path to saved image file.
    """
    
    margins=0.05
    bbox_to_anchor=(0, 0)  
    figsize=(0.23*len(df_true_melt_dataset_label_c_c)/len(dataset_names)+margins, 2.5)

    sns.set_style('whitegrid')
    fig, ax = pyplot.subplots(figsize=figsize)
    pyplot.setp(ax.collections)

    pyplot.axhline(y=0, color='gray', linestyle='-', label='No deviation', zorder=1)

    for i, dataset in enumerate(dataset_names):
        pyplot.scatter(x="label", y="z-score", data=df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset], color=dataset_color_dict[dataset], label=f'{dataset}', s=10, zorder=i*5)
        pyplot.plot(df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset]['label'], df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset]['z-score'], color=dataset_color_dict[dataset], linewidth=0.75, zorder=i*10)

    pyplot.margins(x=0.05, y=0.15)
    pyplot.grid(True)
    ax.set_xticklabels([])

    if title == 'auto':
        pyplot.title('Deviation from resamples', y=1.02, **{'fontname':'Arial', 'size':8}, fontweight='bold')
    else:
        pyplot.title(f'{title}', y=1.02, **{'fontname':'Arial', 'size':8}, fontweight='bold')
    pyplot.ylabel('z-score', **{'fontname':'Arial', 'size':8})
    pyplot.yticks(**{'fontname':'Arial', 'size':8})

    if legend_bool == True:
        legend_props = font_manager.FontProperties(family='Arial', style='normal', size=6)
        if legend_pos == 'outside':
            pyplot.legend(loc='upper left', framealpha=1, prop=legend_props, bbox_to_anchor=(1.05,1.0))
        elif legend_pos == 'inside':
            pyplot.legend(loc='upper right', framealpha=1, prop=legend_props)
    for i, artist in enumerate(ax.findobj(PathCollection)):
        artist.set_zorder(1)

    if subtree == 'doublet':   
        for i in range(len(df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values)):
            c1_str = df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values[i][1]
            c2_str = df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values[i][3]

            x = i
            y = -0.06
            ax.add_artist(_make_circle(cell_color_dict[c1_str], 4.5, x, y, 0.4))
            ax.annotate(c1_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})

            x = i
            y = -0.15
            ax.add_artist(_make_circle(cell_color_dict[c2_str], 4.5, x, y, 0.4))
            ax.annotate(c2_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})  
        if cutoff==None:
            pyplot.xlabel(f'All {subtree} combinations', labelpad=22.5, **{'fontname':'Arial', 'size':8})
        else:
            pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {int(len(df_true_melt_dataset_label_c_c)/len(dataset_names))} by abs z-score)', labelpad=22.5, **{'fontname':'Arial', 'size':8})

    if subtree == 'triplet':
        for i in range(len(df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values)):
            c1_str = df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values[i][1]
            c2_str = df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values[i][4]
            c3_str = df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values[i][6]

            x = i
            y = -0.06
            ax.add_artist(_make_circle(cell_color_dict[c1_str], 4.5, x, y, 0.4))
            ax.annotate(c1_str, 
                        xy=(x, y), 
                        verticalalignment='center', 
                        horizontalalignment='center',
                        annotation_clip=False, 
                        xycoords=('data', 'axes fraction'),
                        **{'fontname':'Arial', 'size':8})

            x = i
            y = -0.18
            ax.add_artist(_make_circle(cell_color_dict[c2_str], 4.5, x, y, 0.4))
            ax.annotate(c2_str, 
                        xy=(x, y), 
                        verticalalignment='center', 
                        horizontalalignment='center',
                        annotation_clip=False, 
                        xycoords=('data', 'axes fraction'),
                        **{'fontname':'Arial', 'size':8})   

            x = i
            y = -0.27
            ax.add_artist(_make_circle(cell_color_dict[c3_str], 4.5, x, y, 0.4))
            ax.annotate(c3_str, 
                        xy=(x, y), 
                        verticalalignment='center', 
                        horizontalalignment='center',
                        annotation_clip=False, 
                        xycoords=('data', 'axes fraction'),
                        **{'fontname':'Arial', 'size':8}) 

        if cutoff==None:
            pyplot.xlabel(f'All {subtree} combinations', labelpad=40, **{'fontname':'Arial', 'size':8})
        else:
            pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {int(len(df_true_melt_dataset_label_c_c)/len(dataset_names))} by abs z-score)', labelpad=40, **{'fontname':'Arial', 'size':8})


    if subtree == 'quartet':
        for i in range(len(df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values)):
            c1_str = df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values[i][2]
            c2_str = df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values[i][4]
            c3_str = df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values[i][8]
            c4_str = df_true_melt_dataset_label_c_c.loc[df_true_melt_dataset_label_c_c['dataset']==dataset_names[0]]['label'].values[i][10]

            x = i
            y = -0.06
            ax.add_artist(_make_circle(cell_color_dict[c1_str], 4.5, x, y, 0.4))
            ax.annotate(c1_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})

            x = i
            y = -0.15
            ax.add_artist(_make_circle(cell_color_dict[c2_str], 4.5, x, y, 0.4))
            ax.annotate(c2_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})   

            x = i
            y = -0.27
            ax.add_artist(_make_circle(cell_color_dict[c3_str], 4.5, x, y, 0.4))
            ax.annotate(c3_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})  

            x = i
            y = -0.36
            ax.add_artist(_make_circle(cell_color_dict[c4_str], 4.5, x, y, 0.4))
            ax.annotate(c4_str, 
                        xy=(x, y), 
                        verticalalignment="center", 
                        horizontalalignment="center",
                        annotation_clip=False, 
                        xycoords=("data", "axes fraction"),
                        **{'fontname':'Arial', 'size':8})  

        if cutoff==None:
            pyplot.xlabel(f'All {subtree} combinations', labelpad=52.5, **{'fontname':'Arial', 'size':8})
        else:
            pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {int(len(df_true_melt_dataset_label_c_c)/len(dataset_names))} by abs z-score)', labelpad=52.5, **{'fontname':'Arial', 'size':8})

    if save==True:
        pyplot.savefig(f"{image_save_path}.{image_format}", dpi=dpi, bbox_inches="tight")
