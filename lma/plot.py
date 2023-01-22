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
pyplot.rcParams['svg.fonttype'] = 'none'
mpl.rcParams.update({'font.size': 8})


# -

def dfs_for_plotting(dfs_concat, num_resamples, subtree_dict, cutoff='auto', num_null=1000):
    '''
    Converts DataFrame out from resample_trees functions into DataFrames for plotting
    Calculates z-scores for subtrees by comparing the observed number to the mean/std across all resamples
    Calculates null z-scores by comparing the observed number of 100 random resamples 
    to the mean/std across the rest of the resamples
    
    ---------
    Parameters
    ---------
    dfs_concat: DataFrame 
        Indexed by subtree_dict
        Last column is analytically solved expected number of each quartet
        Second to last column is observed number of occurences in the original dataset
        Rest of columns are the observed number of occurences in the resampled sets
        Output from resample_trees functions
    num_resamples: int
        number of resamples
    subtree_dict : dictionary
        Keys are subtrees, values are integers
    cutoff: string or NoneType or int
        Take cutoff number of subtrees with largest absolute z-scores to include in plots
        If not provided explicitly, will be automatically determined to take all subtrees with abs z-score > 1
        If NoneType, take all subtrees
    num_null: int
        Take num_null number of resamples to calculate z-scores as part of null distribution
        
    Output
    --------
    df_true_melt_subset: DataFrame
        DataFrame with cutoff number of most significant subtrees for plotting
        Sorted by z-score from most over-represented to most under-represented
        Columns: subtree_val, observed, expected, z-score, abs z-score, label, min, mean, max, counts_adjusted_p_value
            subtree_val: int
                val corresponding to subtree_dict
            observed: int
                Count in original trees
            expected: int
                Analytically solved expected count
            z-score: int
                Computed using observed values and mean/std across resamples
            abs z-score: int
                absolute value of z-score
            label: string
                key corresponding to subtree_dict
            min: int
                minimum count across across all resamples
            mean: int
                average count across across all resamples
            max: int
                maximum count across across all resamples
            adj_p_val: float
                adjusted p-value, one-sided test, corrected using the Bonferonni correction
            z-score min: int
                minimum z-score across across 100 random resamples
            z-score mean: int
                average z-score across across 100 random resamples
            z-score max: int
                maximum z-score across across 100 random resamples
    df_melt_subset: DataFrame 
        Melted DataFrame with observed count for cutoff number of most significant subtrees (all resamples)
        Columns: subtree_val, observed, label
            subtree_val: int
                val corresponding to subtree_dict
            observed: int
                Count in original trees
            label: string
                key corresponding to subtree_dict
    df_melt_100resamples_subset: DataFrame 
        Melted DataFrame with observed count for cutoff number of most significant subtrees (100 random resamples)
        Columns: subtree_val, observed, label
            subtree_val: int
                val corresponding to subtree_dict
            observed: int
                Count in original trees
            label: string
                key corresponding to subtree_dict
    df_zscores_i_concat_melt_subset: DataFrame 
        Melted DataFrame with null z-score for cutoff number of most significant subtrees (1000 random resamples)
        Columns: subtree_val, observed, label
            subtree_val: int
                val corresponding to subtree_dict
            observed: int
                Count in original trees
            label: string
                key corresponding to subtree_dict
    df_zscores_i_concat_melt_100resamples_subset: DataFrame 
        Melted DataFrame with null z-score for cutoff number of most significant subtrees (100 random resamples)
        Columns: subtree_val, observed, label
            subtree_val: int
                val corresponding to subtree_dict
            observed: int
                Count in original trees
            label: string
                key corresponding to subtree_dict
    '''

    # slice out the triplets of the original trees
    df_true_slice = dfs_concat.loc[:,'observed']

    # dataframe of original trees
    data = {'subtree_val': df_true_slice.index,
            'observed': df_true_slice.values}
    df_true_melt = pd.DataFrame(data)

    # slice out the triplets of the original trees
    expected = dfs_concat.loc[:,'expected'].values

    # dataframe of resampled trees
    resamples = num_resamples - 1
    df_melt = pd.melt(dfs_concat.loc[:,'0':f'{resamples}'].transpose(), var_name='subtree_val', value_name='observed')
    df_melt_100resamples = pd.melt(dfs_concat.loc[:,'0':'99'].transpose(), var_name='subtree_val', value_name='observed')

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

    df_true_melt_subset['min'] = [df_melt_subset.groupby(['subtree_val']).min().loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    df_true_melt_subset['mean'] = [df_melt_subset.groupby(['subtree_val']).mean().loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    df_true_melt_subset['max'] = [df_melt_subset.groupby(['subtree_val']).max().loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    
    # calculate p-value (one-sided test)
    adj_p_val_list = []
    for i, j in zip(df_true_melt_subset['subtree_val'].values, df_true_melt_subset['z-score'].values):
        resamples = dfs_concat.iloc[i].values[:-1]
        actual = df_true_melt_subset.loc[df_true_melt_subset['subtree_val']==i]['observed'].values[0]
        if j > 0:
            pos = sum(resamples>=actual)
        elif j < 0:
            pos = sum(resamples<=actual)
        elif j == 0:
            pos=len(resamples)

        p_val = pos/len(resamples)*len(df_true_melt)
        adj_p_val_list.append(p_val)

    df_true_melt_subset['adj_p_val'] = adj_p_val_list
    
    # calculate deviation of each resample
    df_zscores_i_list = []
    for i in tqdm(range(num_null)):
        df_true_slice_i = dfs_concat[f'{i}'].copy()
        data = {'subtree_val': df_true_slice_i.index,
                'observed': df_true_slice_i.values}
        df_true_melt_i = pd.DataFrame(data)

        df_subset_i = dfs_concat[dfs_concat.columns[~dfs_concat.columns.isin([f'{i}',f'{num_resamples}'])]].copy()
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

        df_zscores_i = pd.DataFrame(zscores_i, columns=[i])
        df_zscores_i_list.append(df_zscores_i)
        
    df_zscores_i_concat = pd.concat(df_zscores_i_list, axis=1)
    df_zscores_i_concat.fillna(0, inplace=True)
    
    df_zscores_i_concat_melt = df_zscores_i_concat.transpose().melt(var_name='subtree_val', value_name='observed')
    df_zscores_i_concat_melt_100resamples = df_zscores_i_concat.loc[:,:99].transpose().melt(var_name='subtree_val', value_name='observed')
    
    # subset the resamples
    df_zscores_i_concat_melt_subset_list = []
    for i in df_true_melt_subset['subtree_val']:
        df_zscores_i_concat_melt_subtree = df_zscores_i_concat_melt.loc[df_zscores_i_concat_melt['subtree_val']==i].copy()
        df_zscores_i_concat_melt_subtree['label']=list(subtree_dict.keys())[i]
        df_zscores_i_concat_melt_subset_list.append(df_zscores_i_concat_melt_subtree)
    df_zscores_i_concat_melt_subset = pd.concat(df_zscores_i_concat_melt_subset_list)
    
    # subset the resamples
    df_zscores_i_concat_melt_100resamples_subset_list = []
    for i in df_true_melt_subset['subtree_val']:
        df_zscores_i_concat_melt_100resamples_subtree = df_zscores_i_concat_melt_100resamples.loc[df_zscores_i_concat_melt_100resamples['subtree_val']==i].copy()
        df_zscores_i_concat_melt_100resamples_subtree['label']=list(subtree_dict.keys())[i]
        df_zscores_i_concat_melt_100resamples_subset_list.append(df_zscores_i_concat_melt_100resamples_subtree)
    df_zscores_i_concat_melt_100resamples_subset = pd.concat(df_zscores_i_concat_melt_100resamples_subset_list)
    
    df_true_melt_subset['z-score min'] = [df_zscores_i_concat_melt_subset.groupby(['subtree_val']).min().loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    df_true_melt_subset['z-score mean'] = [df_zscores_i_concat_melt_subset.groupby(['subtree_val']).mean().loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    df_true_melt_subset['z-score max'] = [df_zscores_i_concat_melt_subset.groupby(['subtree_val']).max().loc[i].values[0] for i in df_true_melt_subset['subtree_val']]
    
    return (df_true_melt_subset, df_melt_subset, df_melt_100resamples_subset, df_zscores_i_concat_melt_subset, df_zscores_i_concat_melt_100resamples_subset)


def make_cell_color_dict(all_trees_sorted, cell_fates=None):
    """Make cell color based on cell fates.
    If cell_fates not provided, use automatically determined cell fates based on tree dataset.
    
    Parameters
    ----------
    all_trees_sorted : list
        List where each entry is a string representing a tree in NEWICK format. 
        Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order
    cell_fates : list
        List where each entry is a string representing a cell fate.
        Automatically determined based on tree dataset if not provided by user.
    
    Returns
    -------
    cell_color_dict : dictionary
        Keys are cell fates, values are colors
    
    """
    if cell_fates == None:
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted for i in sublist])))))
    colors = ['#5FC0D4', '#F0E135', '#F071AB', '#9C80B8', '#F89A3A', '#7EC352'][::-1]
    cell_color_dict = dict(zip(cell_fates, colors))
    return cell_color_dict


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


def plot_frequency(subtree, 
                   df_true_melt_subset, 
                   df_melt_subset, 
                   df_melt_100resamples_subset, 
                   cutoff='auto', 
                   legend_bool=True, 
                   legend_pos='outside',
                   save=False, 
                   image_format='png',
                   dpi=300,
                   image_save_path=None):
    
    '''
    Plot frequency of cutoff number of subtrees in original dataset and all resamples
    
    ---------
    Parameters
    ---------
    subtree : string
        type of subtree 
    df_true_melt_subset : DataFrame
        DataFrame with cutoff number of most significant subtrees for plotting
        Sorted by z-score from most over-represented to most under-represented
        Output from dfs_for_plotting function
    df_melt_subset : DataFrame 
        Melted DataFrame with observed count for cutoff number of most significant subtrees (all resamples)
        Output from dfs_for_plotting function
    df_melt_100resamples_subset : DataFrame 
        Melted DataFrame with observed count for cutoff number of most significant subtrees (100 random resamples)
        Output from dfs_for_plotting function
    cutoff : int
        Take cutoff number of subtrees with largest absolute z-scores to include in plots
        If not provided explicitly, will be automatically determined to take all subtrees with abs z-score > 1
    legend_bool : Boolean
        Include legend in plot
    legend_pos : string
        Position of legend (outside or inside)
    save : Boolean
        Save figure as file
    image format : string
        Format of image file to be saved (png or svg)
    dpi : int
        Resolution of saved image file
    image_save_path : string
        Path to saved image file
    '''

    df_true_melt_subset_sg = df_true_melt_subset.loc[df_true_melt_subset['adj_p_val']<0.05].copy()
    
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
    pyplot.scatter(x='label', y='mean', data=df_true_melt_subset, color='gray', label='Count of resamples', s=2.5)
    pyplot.scatter(x='label', y='expected', data=df_true_melt_subset, color='black', label='Expected count', s=2.5)
    pyplot.scatter(x='label', y='min', data=df_true_melt_subset, color='gray', s=0, label='')
    pyplot.scatter(x='label', y='max', data=df_true_melt_subset, color='gray', s=0, label='')
    pyplot.scatter(x='label', y='observed', data=df_true_melt_subset, color='red', label='', s=2.5)
    pyplot.scatter(x='label', y='observed', data=df_true_melt_subset_sg, color='red', s=25, alpha=0.35, label='Adjusted p-value < 0.05')

    pyplot.margins(margins)
    pyplot.grid(True)
    ax.set_xticklabels([])

    pyplot.title(f'{subtree.capitalize()} frequency', y=1.02, **{'fontname':'Arial', 'size':8}, fontweight='bold')
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
            pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=22.5, **{'fontname':'Arial', 'size':8})
    
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
            pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=40, **{'fontname':'Arial', 'size':8})
    

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
            pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=52.5, **{'fontname':'Arial', 'size':8})
            
    if save==True:
        pyplot.savefig(f"{image_save_path}.{image_format}", dpi=dpi, bbox_inches="tight")


def plot_deviation(subtree, 
                   df_true_melt_subset, 
                   df_zscores_i_concat_melt_subset, 
                   df_zscores_i_concat_melt_100resamples_subset, 
                   cutoff='auto', 
                   legend_bool=True,
                   legend_pos='outside',
                   save=False, 
                   image_format='png',
                   dpi=300,
                   image_save_path=None):
    
    '''
    Plot frequency of cutoff number of subtrees in original dataset and all resamples
    
    ---------
    Parameters
    ---------
    subtree : string
        type of subtree 
    df_true_melt_subset : DataFrame
        DataFrame with cutoff number of most significant subtrees for plotting
        Sorted by z-score from most over-represented to most under-represented
        Output from dfs_for_plotting function
    df_melt_subset : DataFrame 
        Melted DataFrame with observed count for cutoff number of most significant subtrees (all resamples)
        Output from dfs_for_plotting function
    df_melt_100resamples_subset : DataFrame 
        Melted DataFrame with observed count for cutoff number of most significant subtrees (100 random resamples)
        Output from dfs_for_plotting function
    cutoff : int
        Take cutoff number of subtrees with largest absolute z-scores to include in plots
        If not provided explicitly, will be automatically determined to take all subtrees with abs z-score > 1
    legend_bool : Boolean
        Include legend in plot
    legend_pos : string
        Position of legend (outside or inside)
    save : Boolean
        Save figure as file
    image format : string
        Format of image file to be saved (png or svg)
    dpi : int
        Resolution of saved image file
    image_save_path : string
        Path to saved image file
    '''

    df_true_melt_subset_sg = df_true_melt_subset.loc[df_true_melt_subset['adj_p_val']<0.05].copy()
    
    margins=0.05
    bbox_to_anchor=(0, 0)  
    figsize=(0.23*len(df_true_melt_subset)+margins, 2.5)

    sns.set_style('whitegrid')
    fig, ax = pyplot.subplots(figsize=figsize)
    pyplot.setp(ax.collections)

    sns.violinplot(x='label', 
                   y='observed', 
                   data=df_zscores_i_concat_melt_subset, 
                   cut=0,
                   inner=None,
                   color='#BCBEC0',
                   scale='width',
                   linewidth=0,
                   )
    sns.stripplot(x='label', 
                  y='observed', 
                  data=df_zscores_i_concat_melt_100resamples_subset, 
                  jitter=0.2,
                  color='gray',
                  size=0.5,
                 )
    pyplot.scatter(x="label", y="z-score", data=df_true_melt_subset, color='red', label='Observed count', s=2.5)
    pyplot.scatter(x="label", y="z-score mean", data=df_true_melt_subset, color='gray', label='Resampled datasets', s=2.5)
    pyplot.scatter(x="label", y="z-score mean", data=df_true_melt_subset, color='black', label='Average across resamples', s=2.5)
    pyplot.scatter(x="label", y="z-score min", data=df_true_melt_subset, color='gray', s=0, label='')
    pyplot.scatter(x="label", y="z-score max", data=df_true_melt_subset, color='gray', s=0, label='')
    pyplot.scatter(x="label", y="z-score", data=df_true_melt_subset, color='red', label='', s=2.5)
    pyplot.scatter(x="label", y="z-score", data=df_true_melt_subset_sg, color='red', s=25, alpha=0.35, label='Adjusted p-value < 0.05')

    pyplot.margins(margins)
    pyplot.grid(True)
    ax.set_xticklabels([])

    pyplot.title(f'Deviation from resamples', y=1.02, **{'fontname':'Arial', 'size':8}, fontweight='bold')
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
            pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=22.5, **{'fontname':'Arial', 'size':8})
    
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
            pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=40, **{'fontname':'Arial', 'size':8})
    

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
            pyplot.xlabel(f'{subtree.capitalize()} combinations \n(top {len(df_true_melt_subset)} by abs z-score)', labelpad=52.5, **{'fontname':'Arial', 'size':8})
            
    if save==True:
        pyplot.savefig(f"{image_save_path}.{image_format}", dpi=dpi, bbox_inches="tight")








