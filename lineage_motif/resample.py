"""
## lineage_motif.resample
Provides functions for resampling tree datasets.

This module contains the following functions:

- `read_dataset` - Returns sorted tree dataset.
- `resample_trees_doublets` - Returns subtree dictionary and DataFrame containing number of **doublets** across all resamples, the original trees, and the expected number (solved analytically).
- `resample_trees_triplets` - Returns subtree dictionary and DataFrame containing number of **triplets** across all resamples, the original trees, and the expected number (solved analytically).
- `resample_trees_quartets` - Returns subtree dictionary and DataFrame containing number of **quartets** across all resamples, the original trees, and the expected number (solved analytically).
"""
# +
# packages for both resampling and plotting
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import re

# packages for only resampling
from itertools import combinations_with_replacement
import random
from collections import Counter


# +
def _sorted_doublets(tree):
    """Sorts the doublets into alphabetical order (important for assigning doublet index).
    
    Args:
        tree (string): Tree in NEWICK format.
    
    Returns:
        tree (string): New tree in NEWICK format after sorted doublets alphabetically.
    """
    for i in re.findall("\(\w*,\w*\)", tree):
        i_escape = re.escape(i)
        i_split = re.split('[\(\),]', i)
        ingroup = sorted([i_split[1], i_split[2]])
        tree = re.sub(i_escape, f'({ingroup[0]},{ingroup[1]})', tree)
    return tree

def _align_triplets(tree):
    """Aligns triplets so that all of them are in the order of (outgroup, ingroup).
    
    Find all ((x,x),x) triplets, then replace them with the same triplet but in (x,(x,x)) form.
    
    Args:
        tree (string): Tree in NEWICK format.
    
    Returns:
        tree (string): New tree in NEWICK format after aligned triplets.
    """
    for i in re.findall("\(\(\w*,\w*\),\w*\)", tree):
        j = re.findall("\w*", i)
        i_escape = re.escape(i)
        tree = re.sub(i_escape, f'({j[7]},({j[2]},{j[4]}))', tree)
    return tree

def _sorted_quartets(tree):
    """Sorts the quartets so that it is in alphabetical order (important for assigning doublet index).
    
    Tree should have doublets sorted already.
    
    Args:
        tree (string): Tree in NEWICK format.
    
    Returns:
        tree (string): New tree in NEWICK format after sorted quartets alphabetically.
    """
    for i in re.findall("\(\(\w*,\w*\),\(\w*,\w*\)\)", tree):
        i_escape = re.escape(i)
        k = sorted([i[1:6], i[7:12]])
        subtree = f"({k[0]},{k[1]})"
        tree = re.sub(i_escape, subtree, tree)
    return tree


# -

def read_dataset(path):
    """Reads dataset txt file located at `path`.
    
    Args:
        path (string): Path to txt file of dataset. txt file should be formatted as NEWICK trees 
            separated with semi-colons and no spaces.
    
    Returns:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
    """
    with open(path) as f:
        lines = f.readlines()

    all_trees_unsorted = lines[0].split(';')
    all_trees_sorted = [_sorted_quartets(_sorted_doublets(_align_triplets(i))) for i in all_trees_unsorted]
    return all_trees_sorted


# +
def _make_cell_dict(cell_fates):
    """Makes a dictionary of all possible cell fates.
    
    Args:
        cell_fates (list): List with each entry as a cell fate.
    
    Returns:
        cell_dict (dict): Keys are cell types, values are integers.
    """
    
    cell_dict = {}
    for i, j in enumerate(cell_fates):
        cell_dict[j] = i
        
    return cell_dict

def _make_doublet_dict(cell_fates):
    """Makes a dictionary of all possible doublets.
    
    Args:
        cell_fates (list): List with each entry as a cell fate.
        
    Returns:
        doublet_dict (dict): Keys are doublets, values are integers.
    """

    total = '0123456789'
    doublet_combinations = []
    for j in list(combinations_with_replacement(total[:len(cell_fates)],2)):
        #print(j)
        k = sorted([cell_fates[int(j[0])], cell_fates[int(j[1])]])
        doublet = f"({k[0]},{k[1]})"
        doublet_combinations.append(doublet)

    doublet_dict = {}
    for i, j in enumerate(doublet_combinations):
        doublet_dict[j] = i
    return doublet_dict

def _make_triplet_dict(cell_fates):
    """Makes a dictionary of all possible triplets.
    
    Args:
        cell_fates (list): List with each entry as a cell fate.
    
    Returns:
        triplet_dict (dict): Keys are triplets, values are integers.
    """

    total = '0123456789'
    triplet_combinations = []
    for i in cell_fates:
        for j in list(combinations_with_replacement(total[:len(cell_fates)],2)):
            #print(j)
            k = sorted([cell_fates[int(j[0])], cell_fates[int(j[1])]])
            triplet = f"({i},({k[0]},{k[1]}))"
            triplet_combinations.append(triplet)

    triplet_dict = {}
    for i, j in enumerate(triplet_combinations):
        triplet_dict[j] = i
    return triplet_dict

def _make_quartet_dict(cell_fates):
    """Makes a dictionary of all possible quartets.
    
    Args:
        cell_fates (list): List with each entry as a cell fate.
    
    Returns:
        quartet_dict (dict): Keys are quartets, values are integers.
    """

    doublet_dict = _make_doublet_dict(cell_fates)
    z = [sorted([i, j]) for i in list(doublet_dict.keys()) for j in list(doublet_dict.keys())]
    x = [f'({i[0]},{i[1]})' for i in z]

    # get rid of duplicates
    y = []
    for i in x:
        if i not in y:
            y.append(i)
        
    quartet_dict = {}
    for i, j in enumerate(y):
        quartet_dict[j] = i
    return quartet_dict


# -

def _make_dicts(all_trees_sorted, cell_fates=None):
    """Makes subtree and cell dictionaries based on cell fates.
    
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
        cell_fates (NoneType or list, optional): List where each entry is a string representing a cell fate.
            If NoneType (i.e. not provided by user), automatically determined based on tree dataset.
    
    Returns:
        (tuple): Contains the following dictionaries.
        
        - quartet_dict (dict): Keys are quartets, values are integers.
        - triplet_dict (dict): Keys are triplets, values are integers.
        - doublet_dict (dict): Keys are doublets, values are integers.
        - cell_dict (dict): Keys are cell types, values are integers.
    
    """
    if cell_fates == None:
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted for i in sublist])))))
    
    if len(cell_fates)>10:
        print('warning!')
        
    quartet_dict = _make_quartet_dict(cell_fates)
    triplet_dict = _make_triplet_dict(cell_fates)
    doublet_dict = _make_doublet_dict(cell_fates)
    cell_dict = _make_cell_dict(cell_fates)
    return (quartet_dict, triplet_dict, doublet_dict, cell_dict)


# returns relavent subtrees
def _flatten_doublets(all_trees_sorted):
    """Makes a list of all doublets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
    
    Returns:
        doublets (list): List with each entry as a doublet (string).
    """
    doublets = []
    for i in all_trees_sorted:
        doublets.extend(re.findall("\(\w*,\w*\)", i))
    return doublets


def _flatten_all_cells(all_trees_sorted):
    """Makes a list of all cells in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
    
    Returns:
        all_cells (list): List with each entry as a cell (string).
    """
    
    all_cells = []
    for i in all_trees_sorted:
        for j in re.findall("[A-Za-z0-9]+", i):
            all_cells.extend(j)
    return all_cells


def _make_df_doublets(all_trees_sorted, doublet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all doublets in the set of trees provided.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
        doublet_dict (dict): Keys are doublets, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): If True, then index of resulting DataFrame uses `doublet_dict` keys.
            
    Returns:
        df_doublets (DataFrame): Rows are doublets, column is resample number.
    """
    doublets = _flatten_doublets(all_trees_sorted)
    doublets_resample_index = [doublet_dict[i] for i in doublets]
    df_doublets = pd.DataFrame.from_dict(Counter(doublets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_doublets = df_doublets.rename({v: k for k, v in doublet_dict.items()})
    return df_doublets


def _make_df_all_cells(all_trees_sorted, cell_dict, resample, labels_bool=False):
    """Makes a DataFrame of all cells in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
        cell_dict (dict): Keys are cell types, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): If True, then index of resulting DataFrame uses `doublet_dict` keys.
            
    Returns:
        df_doublets (DataFrame): Rows are cell types, column is resample number.
    """
    all_cells = _flatten_all_cells(all_trees_sorted)
    all_cells_resample_index = [cell_dict[i] for i in all_cells]
    df_all_cells = pd.DataFrame.from_dict(Counter(all_cells_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_all_cells = df_all_cells.rename({v: k for k, v in cell_dict.items()})
    return df_all_cells


# Replace all leaves drawing from repl_list
def _replace_all(tree, repl_list, replacement_bool):
    """Replaces all cells in tree with a cell drawing from `repl_list`.
    
    Args:
        tree (string): Tree in NEWICK format.
        repl_list (list): List of all cells.
        replacement_bool (bool): Draw with or without replacement from `repl_list`.
    
    Returns:
        new_tree_sorted (string): tree in NEWICK format.
            Tree is sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
    """
    if replacement_bool==False:
        def repl_all(var):
            return repl_list.pop()
    elif replacement_bool==True:
        def repl_all(var):
            return random.choice(repl_list)
    new_tree = re.sub("[A-Za-z0-9]+", repl_all, tree)
    new_tree_sorted = _sorted_quartets(_sorted_doublets(new_tree))
    return new_tree_sorted


def _process_dfs_doublet(df_doublet_true, dfs_doublet_new, num_resamples, doublet_dict, cell_dict, df_all_cells_true):
    """Arranges observed counts for each doublet in all resamples and original trees into a combined DataFrame.
    
    Last column is analytically solved expected number of each doublet.
        
    Args:
        df_doublet_true (DataFrame): DataFrame with number of each doublet in original trees, indexed by `doublet_dict`.
        dfs_doublet_new (list): List with each entry as DataFrame of number of each doublet in each set
            of resampled trees, indexed by doublet_dict.
        num_resamples (int): Number of resample datasets.
        doublet_dict (dict): Keys are doublets, values are integers.
        cell_dict (dict): Keys are cell types, values are integers.
        df_all_cells_true (DataFrame): DataFrame with number of each cell fate in original trees, indexed by `cell_dict`.
    
    Returns:
        dfs_c (DataFrame): Indexed by values from `doublet_dict`.
            Last column is analytically solved expected number of each doublet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    
    """
    
    dfs_list = [dfs_doublet_new[i] for i in range(num_resamples)] + [df_doublet_true]
    dfs_c = pd.concat(dfs_list, axis=1, sort=False)
    
    dfs_c.fillna(0, inplace=True)

    # for doublet df
    empty_indices = [i for i in range(0,len(doublet_dict)) if i not in dfs_c.index]
    for i in empty_indices:
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        dfs_c = pd.concat([dfs_c, df_to_append], axis=0)
    dfs_c.sort_index(inplace=True)
    
    # for all cells df
    empty_indices = [i for i in range(0,len(cell_dict)) if i not in df_all_cells_true.index]
    for i in empty_indices:
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_all_cells_true = pd.concat([df_all_cells_true, df_to_append], axis=0)
    
    df_all_cells_true_norm = df_all_cells_true/df_all_cells_true.sum()
    df_all_cells_true_norm = df_all_cells_true_norm.rename({v: k for k, v in cell_dict.items()})
    
    expected_list = []
    for key in doublet_dict.keys():
        split = key.split(',')
        cell_1 = split[0][-1]
        cell_2 = split[1][0]
        #print(cell_1, cell_2)
        p_cell_1 = df_all_cells_true_norm.loc[cell_1].values[0]
        p_cell_2 = df_all_cells_true_norm.loc[cell_2].values[0]
        #print(p_cell_1, p_cell_2)
        expected = dfs_c.sum()[0]*p_cell_1*p_cell_2
        if cell_1 != cell_2:
            expected *= 2
        #print(expected)
        expected_list.append(expected)
        
    dfs_c = dfs_c.copy()
    dfs_c['expected'] = expected_list
    dfs_c.fillna(0, inplace=True)
    
    return dfs_c


def resample_trees_doublets(all_trees_sorted, 
                            num_resamples=10000, 
                            replacement_bool=True, 
                            cell_fates='auto'
                            ):
    """Performs resampling of trees, drawing with or without replacement, returning subtree dictionary and DataFrame containing
    number of doublets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done by replacing each cell fate with a randomly chosen cell fate across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
    
    Returns:
        (tuple): Contains the following variables.
        - doublet_dict (dict): Keys are doublets, values are integers.
        - cell_fates (list): List where each entry is a string representing a cell fate.
        - dfs_c (DataFrame): Indexed by values from `doublet_dict`.
            Last column is analytically solved expected number of each doublet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.


    """
    # automatically determine cell fates if not explicitly provided
    if cell_fates == 'auto':
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted for i in sublist])))))
    
    # _make_subtree_dict functions can only handle 10 cell fates max
    if len(cell_fates)>10:
        print('warning!')
        
    doublet_dict = _make_doublet_dict(cell_fates)
    cell_dict = _make_cell_dict(cell_fates)
    
    # store result for each rearrangement in dfs list
    dfs_doublets_new = []
    df_doublets_true = _make_df_doublets(all_trees_sorted, doublet_dict, 'observed', False)
    df_all_cells_true = _make_df_all_cells(all_trees_sorted, cell_dict, 'observed', False)

    # rearrange leaves num_resamples times
    for resample in tqdm(range(0, num_resamples)):
        all_cells_true = _flatten_all_cells(all_trees_sorted)
        
        # shuffle if replacement=False
        if replacement_bool==False:
            random.shuffle(all_cells_true)
            
        new_trees = [_replace_all(i, all_cells_true, replacement_bool) for i in all_trees_sorted]
        df_doublets_new = _make_df_doublets(new_trees, doublet_dict, resample, False)
        dfs_doublets_new.append(df_doublets_new)
        
    dfs_c = _process_dfs_doublet(df_doublets_true, dfs_doublets_new, num_resamples, doublet_dict, cell_dict, df_all_cells_true)
    
    return (doublet_dict, cell_fates, dfs_c)


# returns relavent subtrees
def _flatten_triplets(all_trees_sorted):
    """Makes a list of all triplets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
    
    Returns:
        triplets (list): List with each entry as a triplet (string).
    """
    triplets = []
    for i in all_trees_sorted:
        triplets.extend(re.findall("\(\w*,\(\w*,\w*\)\)", i))
    return triplets


def _replace_doublets_blank(tree):
    """Erases all doublets in tree.
    
    Args:
        tree (string): tree in NEWICK format.
    
    Returns:
        new_tree (string): tree in NEWICK format without doublets.
    """
    def repl_doublets_blank(var):
        return ''
    new_tree = re.sub("\(\w*,\w*\)", repl_doublets_blank, tree)
    return new_tree


# returns relavent subtrees
def _flatten_singlets(all_trees_sorted):
    """Returns all singlets (non-doublets) in list of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
    
    Returns:
        singlets (list): List with each entry as a singlet (string).
    """
    x_doublets = [_replace_doublets_blank(i) for i in all_trees_sorted]
    singlets = []
    for i in x_doublets:
        for j in re.findall("[A-Za-z0-9]+", i):
            singlets.extend(j)
    return singlets


def _make_df_triplets(all_trees_sorted, triplet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all triplets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
        triplet_dict (dict): Keys are triplets, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): if True, then index of resulting DataFrame uses `triplet_dict` keys.
            
    Returns:
        df_triplets (DataFrame): Rows are triplets, column is resample number.
    """
    triplets = _flatten_triplets(all_trees_sorted)
    triplets_resample_index = [triplet_dict[i] for i in triplets]
    df_triplets = pd.DataFrame.from_dict(Counter(triplets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_triplets = df_triplets.rename({v: k for k, v in triplet_dict.items()})
    return df_triplets


def _make_df_singlets(all_trees_sorted, cell_dict, resample, labels_bool=False):
    """Makes a DataFrame of all singlets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
        cell_dict (dict): Keys are cell types, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): If True, then index of resulting DataFrame uses `cell_dict` keys.
    
    Returns:
        df_singlets (DataFrame): Rows are singlets, column is resample number.
    """
    singlets = _flatten_singlets(all_trees_sorted)
    singlets_resample_index = [cell_dict[i] for i in singlets]
    df_singlets = pd.DataFrame.from_dict(Counter(singlets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_singlets = df_singlets.rename({v: k for k, v in cell_dict.items()})
    return df_singlets


def _replace_doublets_symbol(tree):
    """Replaces all doublets in tree with "?".
    
    Args:
        tree (string): Tree in NEWICK format.
    
    Returns:
        new_tree (string): Tree in NEWICK format, doublets replaced with "?".
    """
    def repl_doublets_symbol(var):
        return '?'
    new_tree = re.sub("\(\w*,\w*\)", repl_doublets_symbol, tree)
    return new_tree


# Replace doublets drawing from doublets_true
def _replace_symbols(tree, doublets, replacement_bool):
    """Replaces all "?" in tree with a doublet drawing from `doublets`.
    
    Args:
        tree (string): Tree in NEWICK format.
        doublets (list): List with each entry as a doublet (string).
        replacement_bool (bool): Draw with or without replacement from `doublets`.
    
    Returns:
        tree (string): Tree in NEWICK format, "?" replaced with doublet.
    """
    if replacement_bool==False:
        def repl_symbols(var):
            return doublets.pop()
    elif replacement_bool==True:
        def repl_symbols(var):
            return random.choice(doublets)
    new_tree = re.sub("\?", repl_symbols, tree)
    return new_tree


def _process_dfs_triplet(df_triplets_true, dfs_triplets_new, num_resamples, triplet_dict, doublet_dict, cell_dict, df_doublets_true, df_singlets_true):
    """Arranges observed counts for each triplet in all resamples and original trees into a combined DataFrame.
    
    Last column is analytically solved expected number of each triplet.
        
    Args:
        df_triplet_true (DataFrame): DataFrame with number of each triplet in original trees, indexed by `triplet_dict`.
        dfs_triplet_new (list): List with each entry as DataFrame of number of each triplet in each set 
            of resampled trees, indexed by `triplet_dict`.
        num_resamples (int): Number of resample datasets.
        triplet_dict (dict): Keys are triplets, values are integers.
        doublet_dict (dict): Keys are doublets, values are integers.
        cell_dict (dict): Keys are cell types, values are integers.
        df_doublets_true (DataFrame): DataFrame with number of each doublet in original trees, indexed by `doublet_dict`.
        df_singlets_true (DataFrame): DataFrame with number of each cell fate in original trees, indexed by `cell_dict`.
    
    Returns:
        dfs_c (DataFrame): Indexed by values from `triplet_dict`.
            Last column is analytically solved expected number of each triplet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    
    """
    
    dfs_list = [dfs_triplets_new[i] for i in range(num_resamples)] + [df_triplets_true]
    dfs_c = pd.concat(dfs_list, axis=1, sort=False)
    
    dfs_c.fillna(0, inplace=True)

    # for triplet df
    empty_indices = [i for i in range(0,len(triplet_dict)) if i not in dfs_c.index]
    for i in empty_indices:
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        dfs_c = pd.concat([dfs_c, df_to_append], axis=0)
    dfs_c.sort_index(inplace=True)
    
    # for singlets df
    empty_indices = [i for i in range(0,len(cell_dict)) if i not in df_singlets_true.index]
    for i in empty_indices:
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_singlets_true = pd.concat([df_singlets_true, df_to_append], axis=0)

    df_singlets_true_norm = df_singlets_true/df_singlets_true.sum()
    df_singlets_true_norm = df_singlets_true_norm.rename({v: k for k, v in cell_dict.items()})
    
    # for doublets df
    empty_indices = [i for i in range(0,len(doublet_dict)) if i not in df_doublets_true.index]
    for i in empty_indices:
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_doublets_true = pd.concat([df_doublets_true, df_to_append], axis=0)

    df_doublets_true_norm = df_doublets_true/df_doublets_true.sum()
    df_doublets_true_norm = df_doublets_true_norm.rename({v: k for k, v in doublet_dict.items()})
    
    expected_list = []
    for key in triplet_dict.keys():
        cell_1 = key[1]
        cell_2 = key[3:8]
        #print(cell_1, cell_2)
        p_cell_1 = df_singlets_true_norm.loc[cell_1].values[0]
        p_cell_2 = df_doublets_true_norm.loc[cell_2].values[0]
        #print(p_cell_1, p_cell_2)
        expected = dfs_c.sum()[0]*p_cell_1*p_cell_2
        #print(expected)
        expected_list.append(expected)
        
    dfs_c = dfs_c.copy()
    dfs_c['expected'] = expected_list
    dfs_c.fillna(0, inplace=True)
    
    return dfs_c


def resample_trees_triplets(all_trees_sorted, 
                            num_resamples=10000, 
                            replacement_bool=True,
                            cell_fates='auto'
                           ):
    """Performs resampling of tree, drawing with or without replacement, returning subtree dictionary and DataFrame containing 
    number of triplets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done via (1) replacing each cell with a randomly chosen singlet across all trees and 
    (2) replacing each doublet with a randomly chosen doublet across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
    
    Returns:
        (tuple): Contains the following variables.
        - triplet_dict (dict): Keys are triplets, values are integers.
        - cell_fates (list): List where each entry is a string representing a cell fate.
        - dfs_c (DataFrame): Indexed by values from `triplet_dict`.
            Last column is analytically solved expected number of each triplet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    """
    # automatically determine cell fates if not explicitly provided
    if cell_fates == 'auto':
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted for i in sublist])))))
    
    # _make_subtree_dict functions can only handle 10 cell fates max
    if len(cell_fates)>10:
        print('warning!')
      
    triplet_dict = _make_triplet_dict(cell_fates)
    doublet_dict = _make_doublet_dict(cell_fates)
    cell_dict = _make_cell_dict(cell_fates)
    
    # store result for each rearrangement in dfs list
    dfs_triplets_new = []
    df_triplets_true = _make_df_triplets(all_trees_sorted, triplet_dict, 'observed', False)
    df_doublets_true = _make_df_doublets(all_trees_sorted, doublet_dict, 'observed', False)
    df_singlets_true = _make_df_singlets(all_trees_sorted, cell_dict, 'observed', False)

    # rearrange leaves num_resamples times
    for resample in tqdm(range(0, num_resamples)):
        doublets_true = _flatten_doublets(all_trees_sorted)
        singlets_true = _flatten_singlets(all_trees_sorted)
        
        # shuffle if replacement=False
        if replacement_bool==False:
            random.shuffle(doublets_true)
            random.shuffle(singlets_true)
        
        # first, replace the doublet with a symbol
        new_trees_1 = [_replace_doublets_symbol(i) for i in all_trees_sorted]
        # then, replace all other cells 
        new_trees_2 = [_replace_all(i, singlets_true, replacement_bool) for i in new_trees_1]
        # then, replace the symbols
        new_trees_3 = [_replace_symbols(i, doublets_true, replacement_bool) for i in new_trees_2]
        df_triplets_new = _make_df_triplets(new_trees_3, triplet_dict, resample, False)
        dfs_triplets_new.append(df_triplets_new)
        
    dfs_c = _process_dfs_triplet(df_triplets_true, dfs_triplets_new, num_resamples, triplet_dict, doublet_dict, cell_dict, df_doublets_true, df_singlets_true)
    
    return (triplet_dict, cell_fates, dfs_c)


# returns relavent subtrees
def _flatten_quartets(all_trees):
    """Makes a list of all quartets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
    
    Returns:
        quartets (list): List with each entry as a quartet (string).
    """
    quartets = []
    for i in all_trees:
        quartets.extend(re.findall("\(\(\w,\w\),\(\w,\w\)\)", i))
    return quartets


def _make_df_quartets(all_trees_sorted, quartet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all quartets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
        quartet_dict (dict): Keys are quartets, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): if True, then index of resulting DataFrame uses `quartet_dict` keys.
    
    Returns:
        df_quartets (DataFrame): Rows are quartets, column is resample number.
    """
    quartets = _flatten_quartets(all_trees_sorted)
    quartets_resample_index = [quartet_dict[i] for i in quartets]
    df_quartets = pd.DataFrame.from_dict(Counter(quartets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_quartets = df_quartets.rename({v: k for k, v in quartet_dict.items()})
    return df_quartets


# Replace doublets drawing from doublets_true
def _replace_doublets(tree, doublets_true, replacement_bool):
    """Replaces all doublets in tree with a new doublet drawing from `doublets_true`.
    
    Args:
        tree (string): Tree in NEWICK format.
        doublets_true (list): List with each entry as a doublet (string).
        replacement_bool (bool): Draw with or without replacement from `doublets_true`.
    
    Returns:
        new_tree_sorted_quartet (string): Tree in NEWICK format, doublets replaced with new doublets, 
            and all doublets/quartets in alphabetical order.
    """
    if replacement_bool==False:
        def repl_doublet(var):
            return doublets_true.pop()
    elif replacement_bool==True:
        def repl_doublet(var):
            return random.choice(doublets_true)
    new_tree = re.sub("\(\w*,\w*\)", repl_doublet, tree)
    new_tree_sorted_quartet = _sorted_quartets(new_tree)
    return new_tree_sorted_quartet


def _process_dfs_quartet(df_quartets_true, dfs_quartets_new, num_resamples, quartet_dict, doublet_dict, df_doublets_true):
    """Arranges observed counts for each quartet in all resamples and original trees into a combined DataFrame.
    
    Last column is analytically solved expected number of each quartet.
        
    Args:
        df_quartet_true (DataFrame): DataFrame with number of each quartet in original trees, indexed by `quartet_dict`.
        dfs_quartet_new (list): List with each entry as DataFrame of number of each quartet in each set 
            of resampled trees, indexed by `quartet_dict`.
        num_resamples (int): Number of resample datasets.
        quartet_dict (dict): Keys are quartets, values are integers.
        doublet_dict (dict): Keys are doublets, values are integers.
        df_doublets_true (DataFrame): DataFrame with number of each doublet in original trees, indexed by `doublet_dict`.
    
    Returns:
        dfs_c (DataFrame): Indexed by values from `quartet_dict`.
            Last column is analytically solved expected number of each quartet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    
    """
    
    dfs_list = [dfs_quartets_new[i] for i in range(num_resamples)] + [df_quartets_true]
    dfs_c = pd.concat(dfs_list, axis=1, sort=False)
    
    dfs_c.fillna(0, inplace=True)

    # for quartet df
    empty_indices = [i for i in range(0,len(quartet_dict)) if i not in dfs_c.index]
    for i in empty_indices:
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        dfs_c = pd.concat([dfs_c, df_to_append], axis=0)
    dfs_c.sort_index(inplace=True)
    
    # for doublets df
    empty_indices = [i for i in range(0,len(doublet_dict)) if i not in df_doublets_true.index]
    for i in empty_indices:
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_doublets_true = pd.concat([df_doublets_true, df_to_append], axis=0)

    df_doublets_true_norm = df_doublets_true/df_doublets_true.sum()
    df_doublets_true_norm = df_doublets_true_norm.rename({v: k for k, v in doublet_dict.items()})
    
    expected_list = []
    for key in quartet_dict.keys():
        cell_1 = key[1:6]
        cell_2 = key[7:12]
        #print(cell_1, cell_2)
        p_cell_1 = df_doublets_true_norm.loc[cell_1].values[0]
        p_cell_2 = df_doublets_true_norm.loc[cell_2].values[0]
        #print(p_cell_1, p_cell_2)
        expected = dfs_c.sum()[0]*p_cell_1*p_cell_2
        #print(expected)
        if cell_1 != cell_2:
            expected *= 2
        expected_list.append(expected)
        
    dfs_c = dfs_c.copy()
    dfs_c['expected'] = expected_list
    dfs_c.fillna(0, inplace=True)
    
    return dfs_c


def resample_trees_quartets(all_trees_sorted, 
                            num_resamples=10000, 
                            replacement_bool=True,
                            cell_fates='auto'
                           ):
    """Performs resampling of tree, drawing with or without replacement, returning subtree dictionary and DataFrame containing 
    the number of quartets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done via replacing each doublet with a randomly chosen doublet from across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
    
    Returns:
        (tuple): Contains the following variables.
        - quartet_dict (dict): Keys are quartets, values are integers.
        - cell_fates (list): List where each entry is a string representing a cell fate.
        - dfs_c (DataFrame): Indexed by values from `quartet_dict`.
            Last column is analytically solved expected number of each quartet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.


    """
    # automatically determine cell fates if not explicitly provided
    if cell_fates == 'auto':
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted for i in sublist])))))
    
    # _make_subtree_dict functions can only handle 10 cell fates max
    if len(cell_fates)>10:
        print('warning!')
      
    quartet_dict = _make_quartet_dict(cell_fates)
    doublet_dict = _make_doublet_dict(cell_fates)
    
    # store result for each rearrangement in dfs list
    dfs_quartets_new = []
    df_quartets_true = _make_df_quartets(all_trees_sorted, quartet_dict, 'observed', False)
    df_doublets_true = _make_df_doublets(all_trees_sorted, doublet_dict, 'observed', False)

    # rearrange leaves num_resamples times
    for resample in tqdm(range(0, num_resamples)):
        doublets_true = _flatten_doublets(all_trees_sorted)
        
        # shuffle if replacement=False
        if replacement_bool==False:
            random.shuffle(doublets_true)
        
        new_trees = [_replace_doublets(i, doublets_true, replacement_bool) for i in all_trees_sorted]
        df_quartets_new = _make_df_quartets(new_trees, quartet_dict, resample, False)
        dfs_quartets_new.append(df_quartets_new)
        
    dfs_c = _process_dfs_quartet(df_quartets_true, dfs_quartets_new, num_resamples, quartet_dict, doublet_dict, df_doublets_true)
    
    return (quartet_dict, cell_fates, dfs_c)

def multi_dataset_resample_trees(datasets,
                                 dataset_names,
                                 subtree,
                                 num_resamples=10000, 
                                 replacement_bool=True, 
                                 cell_fates='auto',
                                 ):
    """Performs resampling of trees, drawing with or without replacement, returning number of subtrees across
        all resamples, the original trees, and the expected number (solved analytically) 
        **for multiple datasets**. The cell fates used are the composite set across all datasets provided.
    
    Resampling is done as described in each of the `resample_trees_subtrees` functions.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree datasets.
    
    Args:
        datasets (list): List where each entry is a path to txt file of dataset. 
            txt file should be formatted as NEWICK trees separated with semi-colons and no spaces
        dataset_names (list): List where each entry is a string representing the dataset label. 
        subtree (string): type of subtree to be analyzed. Should be 'doublet', 'triplet', or 'quartet'.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
    
    Returns:
        (tuple): Contains the following variables.
        - subtree_dict (dict): Keys are subtrees, values are integers.
        - cell_fates (list): List where each entry is a string representing a cell fate.
        - dfs_dataset_c (list): List where each entry is a DataFrame with the following characteristics.
            Indexed by values from `subtree_dict`.
            Last column is dataset label.
            Second to last column is analytically solved expected number of each subtree.
            Third to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.


    """
    # automatically determine cell fates if not explicitly provided
    if cell_fates == 'auto':
        all_trees_sorted_list = []
        for dataset in datasets:
            all_trees_sorted = read_dataset(dataset)
            all_trees_sorted_list.append(all_trees_sorted)
        all_trees_sorted_list_flattened = [i for sublist in all_trees_sorted_list for i in sublist]
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted_list_flattened for i in sublist])))))

    # _make_subtree_dict functions can only handle 10 cell fates max
    if len(cell_fates)>10:
        print('warning!')
        
    # next, resample each dataset using composite cell fates list
    dfs_dataset_list = []
    for index, dataset in enumerate(tqdm(datasets)):
        all_trees_sorted = read_dataset(dataset)
        if subtree == 'doublet':
            (subtree_dict, cell_fates, dfs_dataset) = resample_trees_doublets(all_trees_sorted, 
                                                          num_resamples, 
                                                          replacement_bool,
                                                          cell_fates=cell_fates
                                                          )
            dfs_dataset['dataset'] = dataset_names[index]
            
        elif subtree == 'triplet':
            (subtree_dict, cell_fates, dfs_dataset) = resample_trees_triplets(all_trees_sorted, 
                                                          num_resamples, 
                                                          replacement_bool,
                                                          cell_fates=cell_fates
                                                          )
            dfs_dataset['dataset'] = dataset_names[index]
            
        elif subtree == 'quartet':
            (subtree_dict, cell_fates, dfs_dataset) = resample_trees_quartets(all_trees_sorted, 
                                                          num_resamples, 
                                                          replacement_bool,
                                                          cell_fates=cell_fates
                                                          )
            dfs_dataset['dataset'] = dataset_names[index]
            
        dfs_dataset_list.append(dfs_dataset)
    dfs_dataset_c = pd.concat(dfs_dataset_list)
    return (subtree_dict, cell_fates, dfs_dataset_c)
