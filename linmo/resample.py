"""
## linmo.resample
Provides functions for resampling tree datasets.

This module contains the following functions:

- `sort_align_tree` - Sorts and aligns trees.
- `read_dataset` - Returns sorted tree dataset.
- `resample_trees_doublets` - Returns subtree dictionary and DataFrame containing number of **doublets** across all resamples, the original trees, and the expected number (solved analytically).
- `resample_trees_triplets` - Returns subtree dictionary and DataFrame containing number of **triplets** across all resamples, the original trees, and the expected number (solved analytically).
- `resample_trees_quartets` - Returns subtree dictionary and DataFrame containing number of **quartets** across all resamples, the original trees, and the expected number (solved analytically).
- `resample_trees_asym_quartets` - Returns subtree dictionary and DataFrame containing number of **asymmetric quartets** across all resamples, the original trees, and the expected number (solved analytically).
- `resample_trees_asym_quintets` - Returns subtree dictionary and DataFrame containing number of **asymmetric quintets** across all resamples, the original trees, and the expected number (solved analytically).
- `resample_trees_asym_sextets` - Returns subtree dictionary and DataFrame containing number of **asymmetric sextets** across all resamples, the original trees, and the expected number (solved analytically).
- `resample_trees_asym_septets` - Returns subtree dictionary and DataFrame containing number of **asymmetric septets** across all resamples, the original trees, and the expected number (solved analytically).
- `resample_trees_sextets` - Returns subtree dictionary and DataFrame containing number of **sextets** across all resamples, the original trees, and the expected number (solved analytically).
- `resample_trees_octets` - Returns subtree dictionary and DataFrame containing number of **octets** across all resamples, the original trees, and the expected number (solved analytically).
- `multi_dataset_resample_trees` - Returns subtree dictionary and DataFrame containing number of a defined subtree type across all resamples, the original trees, and the expected number (solved analytically) across all datasets.
"""
# +
# packages for both resampling and plotting
import numpy as np
import pandas as pd
from tqdm import tqdm
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
    for i in re.findall("\(\w,\w\)", tree):
        i_escape = re.escape(i)
        i_split = re.split('[\(\),]', i)
        ingroup = sorted([i_split[1], i_split[2]])
        tree = re.sub(i_escape, f'({ingroup[0]},{ingroup[1]})', tree)
    return tree

def _align_triplets(tree):
    """Aligns triplets so that all of them are in the order of (outgroup, ingroup).
    
    Find all ((x,x),x) triplets, then replace them with the same triplet but in (x,(x,x)) form.
    
    Args:
        tree (string): Tree in NEWICK format. Tree should have doublets sorted already.
    
    Returns:
        tree (string): New tree in NEWICK format after aligned triplets.
    """
    for i in re.findall("\(\(\w,\w\),\w\)", tree):
        j = re.findall("\w*", i)
        i_escape = re.escape(i)
        tree = re.sub(i_escape, f'({j[7]},({j[2]},{j[4]}))', tree)
    return tree

def _sorted_quartets(tree):
    """Sorts the quartets so that it is in alphabetical order (important for assigning doublet index).
    
    Args:
        tree (string): Tree in NEWICK format. Tree should have doublets sorted already.
    
    Returns:
        tree (string): New tree in NEWICK format after sorted quartets alphabetically.
    """
    for i in re.findall("\(\(\w,\w\),\(\w,\w\)\)", tree):
        i_escape = re.escape(i)
        k = sorted([i[1:6], i[7:12]])
        subtree = f"({k[0]},{k[1]})"
        tree = re.sub(i_escape, subtree, tree)
    return tree

def _sorted_octets(tree):
    """Sorts the octets so that it is in alphabetical order (important for assigning doublet index).
    
    Args:
        tree (string): Tree in NEWICK format. Tree should have quartets and doublets sorted already.
    
    Returns:
        tree (string): New tree in NEWICK format after sorted octets alphabetically.
    """
    for i in re.findall("\(\(\(\w,\w\),\(\w,\w\)\),\(\(\w,\w\),\(\w,\w\)\)\)", tree):
        i_escape = re.escape(i)
        k = sorted([i[1:14], i[15:28]])
        subtree = f"({k[0]},{k[1]})"
        tree = re.sub(i_escape, subtree, tree)
    return tree

def _align_sextet(tree):
    """Aligns sextet so that all of them are in the order of (doublet, quartet).
    
    Find all (((x,x),(x,x)),(x,x)) sextets, then replace them with the same sextet but in ((x,x),((x,x),(x,x))) form.
    
    Args:
        tree (string): Tree in NEWICK format. Tree should have doublets and quartets sorted and triplets aligned already.
    
    Returns:
        tree (string): New tree in NEWICK format after aligned asymmetric quartet.
    """
    for i in re.findall("\(\(\(\w,\w\),\(\w,\w\)\),\(\w,\w\)\)", tree):
        j = re.findall("\w*", i)
        i_escape = re.escape(i)
        tree = re.sub(i_escape, f'(({j[16]},{j[18]}),(({j[3]},{j[5]}),({j[9]},{j[11]})))', tree)
    return tree

def _align_asym_quartet(tree):
    """Aligns asymmetric quartet so that all of them are in the order of (outgroup 1, outgroup 2, ingroup).
    
    Find all ((x,(x,x)),x) quartets, then replace them with the same asymmetric quartet but in (x,(x,(x,x))) form.
    
    Args:
        tree (string): Tree in NEWICK format. Tree should have doublets sorted and triplets aligned already.
    
    Returns:
        tree (string): New tree in NEWICK format after aligned asymmetric quartet.
    """
    for i in re.findall("\(\(\w,\(\w,\w\)\),\w\)", tree):
        j = re.findall("\w*", i)
        i_escape = re.escape(i)
        tree = re.sub(i_escape, f'({j[11]},({j[2]},({j[5]},{j[7]})))', tree)
    return tree

def _align_asym_quintet(tree):
    """Aligns asymmetric quintet so that all of them are in the order of (outgroup 1, outgroup 2, ingroup).
    
    Find all ((x,(x,(x,x))),x) quintets, then replace them with the same asymmetric quintet but in (x,(x,(x,(x,x)))) form.
    
    Args:
        tree (string): Tree in NEWICK format. Tree should have doublets sorted, and triplets and asymmetric quartets aligned already.
    
    Returns:
        tree (string): New tree in NEWICK format after aligned asymmetric quintet.
    """
    for i in re.findall("\(\(\w,\(\w,\(\w,\w\)\)\),\w\)", tree):
        j = re.findall("\w*", i)
        i_escape = re.escape(i)
        tree = re.sub(i_escape, f'({j[15]},({j[2]},({j[5]},({j[8]},{j[10]}))))', tree)
    return tree

def _align_asym_sextet(tree):
    """Aligns asymmetric sextet so that all of them are in the order of (outgroup 1, outgroup 2, ingroup).
    
    Find all ((x,(x,(x,(x,x)))),x) sextets, then replace them with the same asymmetric sextet but in (x,(x,(x,(x,(x,x))))) form.
    
    Args:
        tree (string): Tree in NEWICK format. Tree should have doublets sorted, and triplets, asymmetric quartets, 
        and asymmetric quintets aligned already.
    
    Returns:
        tree (string): New tree in NEWICK format after aligned asymmetric sextet.
    """
    for i in re.findall("\(\(\w,\(\w,\(\w,\(\w,\w\)\)\)\),\w\)", tree):
        j = re.findall("\w*", i)
        i_escape = re.escape(i)
        tree = re.sub(i_escape, f'({j[19]},({j[2]},({j[5]},({j[8]},({j[11]},{j[13]})))))', tree)
    return tree

def _align_asym_septet(tree):
    """Aligns asymmetric septet so that all of them are in the order of (outgroup 1, outgroup 2, ingroup).
    
    Find all ((x,(x,(x,(x,(x,x))))),x) septets, then replace them with the same asymmetric septet but in (x,(x,(x,(x,(x,(x,x)))))) form.
    
    Args:
        tree (string): Tree in NEWICK format. Tree should have doublets sorted, and triplets, asymmetric quartets, asymmetric quintets,
        and asymmetric sextets aligned already.
    
    Returns:
        tree (string): New tree in NEWICK format after aligned asymmetric septet.
    """
    for i in re.findall("\(\(\w,\(\w,\(\w,\(\w,\(\w,\w\)\)\)\)\),\w\)", tree):
        j = re.findall("\w*", i)
        i_escape = re.escape(i)
        tree = re.sub(i_escape, f'({j[23]},({j[2]},({j[5]},({j[8]},({j[11]},({j[14]},{j[16]}))))))', tree)
    return tree

# -

def sort_align_tree(tree):
    """Sort and align provided tree. 
    
    Args:
        tree (string): Tree in NEWICK format.
    
    Returns:
        tree (string): Tree in NEWICK format.
            Trees are sorted to have all asymmetric septets in (x,(x,(x,(x,(x,(x,x)))))) format, asymmetric sextets in (x,(x,(x,(x,(x,x))))) format, 
            asymmetric quintets in (x,(x,(x,(x,x)))), asymmetric quartets in (x,(x,(x,x))) format, triplets in (x,(x,x)) format, 
            and all octets/quartets/doublets in alphabetical order.
    """
    tree = _align_asym_septet(_align_asym_sextet(_align_asym_quintet(_align_asym_quartet(_align_sextet(_sorted_octets(_sorted_quartets(_sorted_doublets(_align_triplets(tree)))))))))
    return tree

def read_dataset(path):
    """Reads dataset txt file located at `path`.
    
    Args:
        path (string): Path to txt file of dataset. txt file should be formatted as NEWICK trees 
            separated with semi-colons and no spaces.
    
    Returns:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    """
    with open(path) as f:
        lines = f.readlines()

    all_trees_unsorted = lines[0].split(';')
    all_trees_sorted = [sort_align_tree(i) for i in all_trees_unsorted]
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

### for doublet analysis

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

# returns relavent subtrees
def _flatten_doublets(all_trees_sorted):
    """Makes a list of all doublets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        doublets (list): List with each entry as a doublet (string).
    """
    doublets = []
    for i in all_trees_sorted:
        doublets.extend(re.findall("\(\w,\w\)", i))
    return doublets


def _flatten_all_cells(all_trees_sorted):
    """Makes a list of all cells in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
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
            Trees are sorted using the `sort_align_tree` function.
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
            Trees are sorted using the `sort_align_tree` function.
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
            Tree is sorted to have all triplets in (x,(x,x)) format, and all octets/quartets/doublets in alphabetical order.
    """
    if replacement_bool==False:
        def repl_all(var):
            return repl_list.pop()
    elif replacement_bool==True:
        def repl_all(var):
            return random.choice(repl_list)
    new_tree = re.sub("[A-Za-z0-9]+", repl_all, tree)
    new_tree_sorted = sort_align_tree(new_tree)
    return new_tree_sorted


def _process_dfs_doublet(df_doublet_true, dfs_doublet_new, num_resamples, doublet_dict, cell_dict, df_all_cells_true, calc_expected=True):
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
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
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
    df_to_append_list = []
    for i in tqdm(empty_indices):
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        df_to_append_list.append(df_to_append)
    dfs_c = pd.concat([dfs_c]+df_to_append_list, axis=0)
    dfs_c.sort_index(inplace=True)

    if calc_expected==False:
        return dfs_c
    
    # for all cells df
    empty_indices = [i for i in range(0,len(cell_dict)) if i not in df_all_cells_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_all_cells_true = pd.concat([df_all_cells_true]+df_to_append_list, axis=0)
    
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
                            cell_fates='auto', 
                            calc_expected=True
                            ):
    """Performs resampling of trees, drawing with or without replacement, returning subtree dictionary and DataFrame containing
    number of doublets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done by replacing each cell fate with a randomly chosen cell fate across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
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
        
    dfs_c = _process_dfs_doublet(df_doublets_true, dfs_doublets_new, num_resamples, doublet_dict, cell_dict, df_all_cells_true, calc_expected=True)
    
    return (doublet_dict, cell_fates, dfs_c)

### for triplet analysis

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

# returns relavent subtrees
def _flatten_triplets(all_trees_sorted):
    """Makes a list of all triplets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        triplets (list): List with each entry as a triplet (string).
    """
    triplets = []
    for i in all_trees_sorted:
        triplets.extend(re.findall("\(\w,\(\w,\w\)\)", i))
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
    new_tree = re.sub("\(\w,\w\)", repl_doublets_blank, tree)
    return new_tree


# returns relavent subtrees
def _flatten_non_doublets(all_trees_sorted):
    """Returns all non-doublets in list of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        non_doublets (list): List with each entry as a non_doublet (string).
    """
    x_doublets = [_replace_doublets_blank(i) for i in all_trees_sorted]
    non_doublets = []
    for i in x_doublets:
        for j in re.findall("[A-Za-z0-9]+", i):
            non_doublets.extend(j)
    return non_doublets


def _make_df_triplets(all_trees_sorted, triplet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all triplets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
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


def _make_df_non_doublets(all_trees_sorted, cell_dict, resample, labels_bool=False):
    """Makes a DataFrame of all non_doublets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        cell_dict (dict): Keys are cell types, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): If True, then index of resulting DataFrame uses `cell_dict` keys.
    
    Returns:
        df_non_doublets (DataFrame): Rows are non_doublets, column is resample number.
    """
    non_doublets = _flatten_non_doublets(all_trees_sorted)
    non_doublets_resample_index = [cell_dict[i] for i in non_doublets]
    df_non_doublets = pd.DataFrame.from_dict(Counter(non_doublets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_non_doublets = df_non_doublets.rename({v: k for k, v in cell_dict.items()})
    return df_non_doublets


def _replace_doublets_symbol(tree):
    """Replaces all doublets in tree with "?".
    
    Args:
        tree (string): Tree in NEWICK format.
    
    Returns:
        new_tree (string): Tree in NEWICK format, doublets replaced with "?".
    """
    def repl_doublets_symbol(var):
        return '?'
    new_tree = re.sub("\(\w,\w\)", repl_doublets_symbol, tree)
    return new_tree


def _replace_symbols(tree, subtrees, replacement_bool):
    """Replaces all "?" in tree with a subtree drawing from `subtrees`.
    
    Args:
        tree (string): Tree in NEWICK format.
        subtrees (list): List with each entry as a subtree (string).
        replacement_bool (bool): Draw with or without replacement from `subtrees`.
    
    Returns:
        tree (string): Tree in NEWICK format, "?" replaced with triplet.
    """
    if replacement_bool==False:
        def repl_symbols(var):
            return subtrees.pop()
    elif replacement_bool==True:
        def repl_symbols(var):
            return random.choice(subtrees)
    new_tree = re.sub("\?", repl_symbols, tree)
    return new_tree


def _process_dfs_triplet(df_triplets_true, dfs_triplets_new, num_resamples, triplet_dict, doublet_dict, cell_dict, df_doublets_true, df_non_doublets_true, calc_expected=True):
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
        df_non_doublets_true (DataFrame): DataFrame with number of each cell fate in original trees, indexed by `cell_dict`.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
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
    df_to_append_list = []
    for i in tqdm(empty_indices):
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        df_to_append_list.append(df_to_append)
    dfs_c = pd.concat([dfs_c]+df_to_append_list, axis=0)
    dfs_c.sort_index(inplace=True)

    if calc_expected==False:
        return dfs_c
    
    # for non_doublets df
    empty_indices = [i for i in range(0,len(cell_dict)) if i not in df_non_doublets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_non_doublets_true = pd.concat([df_non_doublets_true]+df_to_append_list, axis=0)

    df_non_doublets_true_norm = df_non_doublets_true/df_non_doublets_true.sum()
    df_non_doublets_true_norm = df_non_doublets_true_norm.rename({v: k for k, v in cell_dict.items()})
    
    # for doublets df
    empty_indices = [i for i in range(0,len(doublet_dict)) if i not in df_doublets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_doublets_true = pd.concat([df_doublets_true]+df_to_append_list, axis=0)

    df_doublets_true_norm = df_doublets_true/df_doublets_true.sum()
    df_doublets_true_norm = df_doublets_true_norm.rename({v: k for k, v in doublet_dict.items()})
    
    expected_list = []
    for key in triplet_dict.keys():
        cell_1 = key[1]
        cell_2 = key[3:-1]
        #print(cell_1, cell_2)
        p_cell_1 = df_non_doublets_true_norm.loc[cell_1].values[0]
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
                            cell_fates='auto', 
                            calc_expected=True
                           ):
    """Performs resampling of tree, drawing with or without replacement, returning subtree dictionary and DataFrame containing 
    number of triplets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done via (1) replacing each cell with a randomly chosen non_doublet across all trees and 
    (2) replacing each doublet with a randomly chosen doublet across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
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
    df_non_doublets_true = _make_df_non_doublets(all_trees_sorted, cell_dict, 'observed', False)

    # rearrange leaves num_resamples times
    for resample in tqdm(range(0, num_resamples)):
        doublets_true = _flatten_doublets(all_trees_sorted)
        non_doublets_true = _flatten_non_doublets(all_trees_sorted)
        
        # shuffle if replacement=False
        if replacement_bool==False:
            random.shuffle(doublets_true)
            random.shuffle(non_doublets_true)
        
        # first, replace the doublet with a symbol
        new_trees_1 = [_replace_doublets_symbol(i) for i in all_trees_sorted]
        # then, replace all other cells 
        new_trees_2 = [_replace_all(i, non_doublets_true, replacement_bool) for i in new_trees_1]
        # then, replace the symbols
        new_trees_3 = [_replace_symbols(i, doublets_true, replacement_bool) for i in new_trees_2]
        df_triplets_new = _make_df_triplets(new_trees_3, triplet_dict, resample, False)
        dfs_triplets_new.append(df_triplets_new)
        
    dfs_c = _process_dfs_triplet(df_triplets_true, dfs_triplets_new, num_resamples, triplet_dict, doublet_dict, cell_dict, df_doublets_true, df_non_doublets_true, calc_expected)
    
    return (triplet_dict, cell_fates, dfs_c)

### for quartet analysis

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

# returns relavent subtrees
def _flatten_quartets(all_trees):
    """Makes a list of all quartets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
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
            Trees are sorted using the `sort_align_tree` function.
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
            and all octets/quartets/doublets in alphabetical order.
    """
    if replacement_bool==False:
        def repl_doublet(var):
            return doublets_true.pop()
    elif replacement_bool==True:
        def repl_doublet(var):
            return random.choice(doublets_true)
    new_tree = re.sub("\(\w,\w\)", repl_doublet, tree)
    new_tree_sorted_quartet = sort_align_tree(new_tree)
    return new_tree_sorted_quartet


def _process_dfs_quartet(df_quartets_true, dfs_quartets_new, num_resamples, quartet_dict, doublet_dict, df_doublets_true, calc_expected=True):
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
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
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
    df_to_append_list = []
    for i in tqdm(empty_indices):
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        df_to_append_list.append(df_to_append)
    dfs_c = pd.concat([dfs_c]+df_to_append_list, axis=0)
    dfs_c.sort_index(inplace=True)

    if calc_expected==False:
        return dfs_c
    
    # for doublets df
    empty_indices = [i for i in range(0,len(doublet_dict)) if i not in df_doublets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_doublets_true = pd.concat([df_doublets_true]+df_to_append_list, axis=0)

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
                            cell_fates='auto', 
                            calc_expected=True
                           ):
    """Performs resampling of tree, drawing with or without replacement, returning subtree dictionary and DataFrame containing 
    the number of quartets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done via replacing each doublet with a randomly chosen doublet from across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
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
        
    dfs_c = _process_dfs_quartet(df_quartets_true, dfs_quartets_new, num_resamples, quartet_dict, doublet_dict, df_doublets_true, calc_expected)
    
    return (quartet_dict, cell_fates, dfs_c)

### for asymmetric quartet analysis

def _make_asym_quartet_dict(cell_fates):
    """Makes a dictionary of all possible asymmetric quartets.
    
    Args:
        cell_fates (list): List with each entry as a cell fate.
    
    Returns:
        asym_quartet_dict (dict): Keys are asymmetric quartets, values are integers.
    """

    total = '0123456789'
    asym_quartet_combinations = []
    for h in cell_fates:
        for i in cell_fates:
            for j in list(combinations_with_replacement(total[:len(cell_fates)],2)):
                #print(j)
                k = sorted([cell_fates[int(j[0])], cell_fates[int(j[1])]])
                asym_quartet = f"({h},({i},({k[0]},{k[1]})))"
                asym_quartet_combinations.append(asym_quartet)

    asym_quartet_dict = {}
    for i, j in enumerate(asym_quartet_combinations):
        asym_quartet_dict[j] = i
    return asym_quartet_dict


def _replace_triplets_blank(tree):
    """Erases all triplets in tree.
    
    Args:
        tree (string): tree in NEWICK format.
    
    Returns:
        new_tree (string): tree in NEWICK format without triplets.
    """
    def repl_triplets_blank(var):
        return ''
    new_tree = re.sub("\(\w,\(\w,\w\)\)", repl_triplets_blank, tree)
    return new_tree

# returns relavent subtrees
def _flatten_non_triplets(all_trees_sorted):
    """Returns all cells outside triplets in list of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        non_triplets (list): List with each entry as a non_triplet (string).
    """
    x_triplets = [_replace_triplets_blank(i) for i in all_trees_sorted]
    non_triplets = []
    for i in x_triplets:
        for j in re.findall("[A-Za-z0-9]+", i):
            non_triplets.extend(j)
    return non_triplets

# returns relavent subtrees
def _flatten_asym_quartets(all_trees_sorted):
    """Makes a list of all asymmetric quartets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        asym_quartets (list): List with each entry as an asymmetric quartet (string).
    """
    asym_quartets = []
    for i in all_trees_sorted:
        asym_quartets.extend(re.findall("\(\w,\(\w,\(\w,\w\)\)\)", i))
    return asym_quartets

def _replace_triplets_symbol(tree):
    """Replaces all triplets in tree with "?".
    
    Args:
        tree (string): Tree in NEWICK format.
    
    Returns:
        new_tree (string): Tree in NEWICK format, triplets replaced with "?".
    """
    def repl_triplets_symbol(var):
        return '?'
    new_tree = re.sub("\(\w,\(\w,\w\)\)", repl_triplets_symbol, tree)
    return new_tree

def _process_dfs_asym_quartet(df_asym_quartets_true, dfs_asym_quartets_new, num_resamples, asym_quartet_dict, triplet_dict, cell_dict, df_triplets_true, df_non_triplets_true, calc_expected=True):
    """Arranges observed counts for each asymmetric quartet in all resamples and original trees into a combined DataFrame.
    
    Last column is analytically solved expected number of each triplet.
        
    Args:
        df_asym_quartet_true (DataFrame): DataFrame with number of each asymmetric quartet in original trees, indexed by `asym_quartet_dict`.
        dfs_asym_quartet_new (list): List with each entry as DataFrame of number of each asymmetric quartet in each set 
            of resampled trees, indexed by `asym_quartet_dict`.
        num_resamples (int): Number of resample datasets.
        asym_quartet_dict (dict): Keys are asymmetric quartets, values are integers.
        triplet_dict (dict): Keys are triplets, values are integers.
        cell_dict (dict): Keys are cell types, values are integers.
        df_triplets_true (DataFrame): DataFrame with number of each triplet in original trees, indexed by `triplet_dict`.
        df_non_triplets_true (DataFrame): DataFrame with number of each cell fate in original trees, indexed by `cell_dict`.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        dfs_c (DataFrame): Indexed by values from `asym_quartet_dict`.
            Last column is analytically solved expected number of each asymmetric quartet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    
    """
    
    dfs_list = [dfs_asym_quartets_new[i] for i in range(num_resamples)] + [df_asym_quartets_true]
    dfs_c = pd.concat(dfs_list, axis=1, sort=False)
    
    dfs_c.fillna(0, inplace=True)

    # for asymmetric quartet df
    empty_indices = [i for i in range(0,len(asym_quartet_dict)) if i not in dfs_c.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        df_to_append_list.append(df_to_append)
    dfs_c = pd.concat([dfs_c]+df_to_append_list, axis=0)
    dfs_c.sort_index(inplace=True)

    if calc_expected==False:
        return dfs_c
    
    # for non_triplets df
    empty_indices = [i for i in range(0,len(cell_dict)) if i not in df_non_triplets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_non_triplets_true = pd.concat([df_non_triplets_true]+df_to_append_list, axis=0)

    df_non_triplets_true_norm = df_non_triplets_true/df_non_triplets_true.sum()
    df_non_triplets_true_norm = df_non_triplets_true_norm.rename({v: k for k, v in cell_dict.items()})
    
    # for triplets df
    empty_indices = [i for i in range(0,len(triplet_dict)) if i not in df_triplets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_triplets_true = pd.concat([df_triplets_true]+df_to_append_list, axis=0)

    df_triplets_true_norm = df_triplets_true/df_triplets_true.sum()
    df_triplets_true_norm = df_triplets_true_norm.rename({v: k for k, v in triplet_dict.items()})
    
    expected_list = []
    for key in asym_quartet_dict.keys():
        cell_1 = key[1]
        cell_2 = key[3:-1]
        #print(cell_1, cell_2)
        p_cell_1 = df_non_triplets_true_norm.loc[cell_1].values[0]
        p_cell_2 = df_triplets_true_norm.loc[cell_2].values[0]
        #print(p_cell_1, p_cell_2)
        expected = dfs_c.sum()[0]*p_cell_1*p_cell_2
        #print(expected)
        expected_list.append(expected)
        
    dfs_c = dfs_c.copy()
    dfs_c['expected'] = expected_list
    dfs_c.fillna(0, inplace=True)
    
    return dfs_c

def _make_df_asym_quartets(all_trees_sorted, asym_quartet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all asym_quartets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        asym_quartet_dict (dict): Keys are asym_quartets, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): if True, then index of resulting DataFrame uses `asym_quartet_dict` keys.
            
    Returns:
        df_asym_quartets (DataFrame): Rows are asym_quartets, column is resample number.
    """
    asym_quartets = _flatten_asym_quartets(all_trees_sorted)
    asym_quartets_resample_index = [asym_quartet_dict[i] for i in asym_quartets]
    df_asym_quartets = pd.DataFrame.from_dict(Counter(asym_quartets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_asym_quartets = df_asym_quartets.rename({v: k for k, v in asym_quartet_dict.items()})
    return df_asym_quartets

def _make_df_non_triplets(all_trees_sorted, cell_dict, resample, labels_bool=False):
    """Makes a DataFrame of all non_triplets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        cell_dict (dict): Keys are cell types, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): If True, then index of resulting DataFrame uses `cell_dict` keys.
    
    Returns:
        df_non_triplets (DataFrame): Rows are non_triplets, column is resample number.
    """
    non_triplets = _flatten_non_triplets(all_trees_sorted)
    non_triplets_resample_index = [cell_dict[i] for i in non_triplets]
    df_non_triplets = pd.DataFrame.from_dict(Counter(non_triplets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_non_triplets = df_non_triplets.rename({v: k for k, v in cell_dict.items()})
    return df_non_triplets

def resample_trees_asym_quartets(all_trees_sorted, 
                            num_resamples=10000, 
                            replacement_bool=True,
                            cell_fates='auto', 
                            calc_expected=True
                           ):
    """Performs resampling of tree, drawing with or without replacement, returning subtree dictionary and DataFrame containing 
    number of asymmetric quartets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done via (1) replacing each triplet with a randomly chosen triplet across all trees and 
    (2) replacing every other cell with a randomly chosen non-triplet cell across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        (tuple): Contains the following variables.
        - asym_quartet_dict (dict): Keys are asym_quartets, values are integers.
        - cell_fates (list): List where each entry is a string representing a cell fate.
        - dfs_c (DataFrame): Indexed by values from `asym_quartet_dict`.
            Last column is analytically solved expected number of each asym_quartet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    """
    # automatically determine cell fates if not explicitly provided
    if cell_fates == 'auto':
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted for i in sublist])))))
    
    # _make_subtree_dict functions can only handle 10 cell fates max
    if len(cell_fates)>10:
        print('warning!')
      
    asym_quartet_dict = _make_asym_quartet_dict(cell_fates)
    triplet_dict = _make_triplet_dict(cell_fates)
    cell_dict = _make_cell_dict(cell_fates)
    
    # store result for each rearrangement in dfs list
    dfs_asym_quartets_new = []
    df_asym_quartets_true = _make_df_asym_quartets(all_trees_sorted, asym_quartet_dict, 'observed', False)
    df_triplets_true = _make_df_triplets(all_trees_sorted, triplet_dict, 'observed', False)
    df_non_triplets_true = _make_df_non_triplets(all_trees_sorted, cell_dict, 'observed', False)

    # rearrange leaves num_resamples times
    for resample in tqdm(range(0, num_resamples)):
        triplets_true = _flatten_triplets(all_trees_sorted)
        non_triplets_true = _flatten_non_triplets(all_trees_sorted)
        
        # shuffle if replacement=False
        if replacement_bool==False:
            random.shuffle(triplets_true)
            random.shuffle(non_triplets_true)
        
        # first, replace the doublet with a symbol
        new_trees_1 = [_replace_triplets_symbol(i) for i in all_trees_sorted]
        # then, replace all other cells 
        new_trees_2 = [_replace_all(i, non_triplets_true, replacement_bool) for i in new_trees_1]
        # then, replace the symbols
        new_trees_3 = [_replace_symbols(i, triplets_true, replacement_bool) for i in new_trees_2]
        df_asym_quartets_new = _make_df_asym_quartets(new_trees_3, asym_quartet_dict, resample, False)
        dfs_asym_quartets_new.append(df_asym_quartets_new)
        
    dfs_c = _process_dfs_asym_quartet(df_asym_quartets_true, dfs_asym_quartets_new, num_resamples, asym_quartet_dict, triplet_dict, cell_dict, df_triplets_true, df_non_triplets_true, calc_expected)
    
    return (asym_quartet_dict, cell_fates, dfs_c)

### for asymmetric quintet analysis

def _make_asym_quintet_dict(cell_fates):
    """Makes a dictionary of all possible asymmetric quintets.
    
    Args:
        cell_fates (list): List with each entry as a cell fate.
    
    Returns:
        asym_quintet_dict (dict): Keys are asymmetric quintets, values are integers.
    """

    total = '0123456789'
    asym_quintet_combinations = []
    for g in cell_fates:
        for h in cell_fates:
            for i in cell_fates:
                for j in list(combinations_with_replacement(total[:len(cell_fates)],2)):
                    #print(j)
                    k = sorted([cell_fates[int(j[0])], cell_fates[int(j[1])]])
                    asym_quintet = f"({g},({h},({i},({k[0]},{k[1]}))))"
                    asym_quintet_combinations.append(asym_quintet)

    asym_quintet_dict = {}
    for i, j in enumerate(asym_quintet_combinations):
        asym_quintet_dict[j] = i
    return asym_quintet_dict

def _replace_asym_quartets_blank(tree):
    """Erases all asymmetric quartets in tree.
    
    Args:
        tree (string): tree in NEWICK format.
    
    Returns:
        new_tree (string): tree in NEWICK format without asym_quartets.
    """
    def repl_asym_quartets_blank(var):
        return ''
    new_tree = re.sub("\(\,\(\,\(\,\\)\)\)", repl_asym_quartets_blank, tree)
    return new_tree

# returns relavent subtrees
def _flatten_non_asym_quartets(all_trees_sorted):
    """Returns all cells outside asymmetric quartets in list of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        non_asym_quartets (list): List with each entry as a non_asym_quartet (string).
    """
    x_asym_quartets = [_replace_asym_quartets_blank(i) for i in all_trees_sorted]
    non_asym_quartets = []
    for i in x_asym_quartets:
        for j in re.findall("[A-Za-z0-9]+", i):
            non_asym_quartets.extend(j)
    return non_asym_quartets

# returns relavent subtrees
def _flatten_asym_quintets(all_trees_sorted):
    """Makes a list of all asymmetric quartets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        asym_quintets (list): List with each entry as an asymmetric quartet (string).
    """
    asym_quintets = []
    for i in all_trees_sorted:
        asym_quintets.extend(re.findall("\(\w,\(\w,\(\w,\(\w,\w\)\)\)\)", i))
    return asym_quintets

def _replace_asym_quartets_symbol(tree):
    """Replaces all asymmetric quartets in tree with "?".
    
    Args:
        tree (string): Tree in NEWICK format.
    
    Returns:
        new_tree (string): Tree in NEWICK format, asymmetric quartets replaced with "?".
    """
    def repl_asym_quartets_symbol(var):
        return '?'
    new_tree = re.sub("\(\w,\(\w,\(\w,\w\)\)\)", repl_asym_quartets_symbol, tree)
    return new_tree

def _process_dfs_asym_quintet(df_asym_quintets_true, dfs_asym_quintets_new, num_resamples, asym_quintet_dict, asym_quartet_dict, cell_dict, df_asym_quartets_true, df_non_asym_quartets_true, calc_expected=True):
    """Arranges observed counts for each asymmetric quartet in all resamples and original trees into a combined DataFrame.
    
    Last column is analytically solved expected number of each asym_quartet.
        
    Args:
        df_asym_quintet_true (DataFrame): DataFrame with number of each asymmetric quartet in original trees, indexed by `asym_quintet_dict`.
        dfs_asym_quintet_new (list): List with each entry as DataFrame of number of each asymmetric quartet in each set 
            of resampled trees, indexed by `asym_quintet_dict`.
        num_resamples (int): Number of resample datasets.
        asym_quintet_dict (dict): Keys are asymmetric quartets, values are integers.
        asym_quartet_dict (dict): Keys are asym_quartets, values are integers.
        cell_dict (dict): Keys are cell types, values are integers.
        df_asym_quartets_true (DataFrame): DataFrame with number of each asymmetric quartet in original trees, indexed by `asym_quartet_dict`.
        df_non_asym_quartets_true (DataFrame): DataFrame with number of each cell fate in original trees, indexed by `cell_dict`.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        dfs_c (DataFrame): Indexed by values from `asym_quintet_dict`.
            Last column is analytically solved expected number of each asymmetric quartet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    
    """
    
    dfs_list = [dfs_asym_quintets_new[i] for i in range(num_resamples)] + [df_asym_quintets_true]
    dfs_c = pd.concat(dfs_list, axis=1, sort=False)
    
    dfs_c.fillna(0, inplace=True)

    # for asymmetric quartet df
    empty_indices = [i for i in range(0,len(asym_quintet_dict)) if i not in dfs_c.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        df_to_append_list.append(df_to_append)
    dfs_c = pd.concat([dfs_c]+df_to_append_list, axis=0)
    dfs_c.sort_index(inplace=True)

    if calc_expected==False:
        return dfs_c
    
    # for non_asym_quartets df
    empty_indices = [i for i in range(0,len(cell_dict)) if i not in df_non_asym_quartets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_non_asym_quartets_true = pd.concat([df_non_asym_quartets_true]+df_to_append_list, axis=0)

    df_non_asym_quartets_true_norm = df_non_asym_quartets_true/df_non_asym_quartets_true.sum()
    df_non_asym_quartets_true_norm = df_non_asym_quartets_true_norm.rename({v: k for k, v in cell_dict.items()})
    
    # for asymmetric quartets df
    empty_indices = [i for i in range(0,len(asym_quartet_dict)) if i not in df_asym_quartets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_asym_quartets_true = pd.concat([df_asym_quartets_true]+df_to_append_list, axis=0)

    df_asym_quartets_true_norm = df_asym_quartets_true/df_asym_quartets_true.sum()
    df_asym_quartets_true_norm = df_asym_quartets_true_norm.rename({v: k for k, v in asym_quartet_dict.items()})
    
    expected_list = []
    for key in asym_quintet_dict.keys():
        cell_1 = key[1]
        cell_2 = key[3:-1]
        #print(cell_1, cell_2)
        p_cell_1 = df_non_asym_quartets_true_norm.loc[cell_1].values[0]
        p_cell_2 = df_asym_quartets_true_norm.loc[cell_2].values[0]
        #print(p_cell_1, p_cell_2)
        expected = dfs_c.sum()[0]*p_cell_1*p_cell_2
        #print(expected)
        expected_list.append(expected)
        
    dfs_c = dfs_c.copy()
    dfs_c['expected'] = expected_list
    dfs_c.fillna(0, inplace=True)
    
    return dfs_c

def _make_df_asym_quintets(all_trees_sorted, asym_quintet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all asym_quintets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        asym_quintet_dict (dict): Keys are asym_quintets, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): if True, then index of resulting DataFrame uses `asym_quintet_dict` keys.
            
    Returns:
        df_asym_quintets (DataFrame): Rows are asym_quintets, column is resample number.
    """
    asym_quintets = _flatten_asym_quintets(all_trees_sorted)
    asym_quintets_resample_index = [asym_quintet_dict[i] for i in asym_quintets]
    df_asym_quintets = pd.DataFrame.from_dict(Counter(asym_quintets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_asym_quintets = df_asym_quintets.rename({v: k for k, v in asym_quintet_dict.items()})
    return df_asym_quintets

def _make_df_non_asym_quartets(all_trees_sorted, cell_dict, resample, labels_bool=False):
    """Makes a DataFrame of all non_asym_quartets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        cell_dict (dict): Keys are cell types, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): If True, then index of resulting DataFrame uses `cell_dict` keys.
    
    Returns:
        df_non_asym_quartets (DataFrame): Rows are non_asym_quartets, column is resample number.
    """
    non_asym_quartets = _flatten_non_asym_quartets(all_trees_sorted)
    non_asym_quartets_resample_index = [cell_dict[i] for i in non_asym_quartets]
    df_non_asym_quartets = pd.DataFrame.from_dict(Counter(non_asym_quartets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_non_asym_quartets = df_non_asym_quartets.rename({v: k for k, v in cell_dict.items()})
    return df_non_asym_quartets

def resample_trees_asym_quintets(all_trees_sorted, 
                            num_resamples=10000, 
                            replacement_bool=True,
                            cell_fates='auto', 
                            calc_expected=True
                           ):
    """Performs resampling of tree, drawing with or without replacement, returning subtree dictionary and DataFrame containing 
    number of asymmetric quartets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done via (1) replacing each asymmetric quartet with a randomly chosen asymmetric quartet across all trees and 
    (2) replacing every other cell with a randomly chosen non-asymmetric quartet cell across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        (tuple): Contains the following variables.
        - asym_quintet_dict (dict): Keys are asym_quintets, values are integers.
        - cell_fates (list): List where each entry is a string representing a cell fate.
        - dfs_c (DataFrame): Indexed by values from `asym_quintet_dict`.
            Last column is analytically solved expected number of each asym_quintet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    """
    # automatically determine cell fates if not explicitly provided
    if cell_fates == 'auto':
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted for i in sublist])))))
    
    # _make_subtree_dict functions can only handle 10 cell fates max
    if len(cell_fates)>10:
        print('warning, _make_subtree_dict functions can only handle 10 cell fates max!')
      
    asym_quintet_dict = _make_asym_quintet_dict(cell_fates)
    asym_quartet_dict = _make_asym_quartet_dict(cell_fates)
    cell_dict = _make_cell_dict(cell_fates)
    
    # store result for each rearrangement in dfs list
    dfs_asym_quintets_new = []
    df_asym_quintets_true = _make_df_asym_quintets(all_trees_sorted, asym_quintet_dict, 'observed', False)
    df_asym_quartets_true = _make_df_asym_quartets(all_trees_sorted, asym_quartet_dict, 'observed', False)
    df_non_asym_quartets_true = _make_df_non_asym_quartets(all_trees_sorted, cell_dict, 'observed', False)

    # rearrange leaves num_resamples times
    for resample in tqdm(range(0, num_resamples)):
        asym_quartets_true = _flatten_asym_quartets(all_trees_sorted)
        non_asym_quartets_true = _flatten_non_asym_quartets(all_trees_sorted)
        
        # shuffle if replacement=False
        if replacement_bool==False:
            random.shuffle(asym_quartets_true)
            random.shuffle(non_asym_quartets_true)
        
        # first, replace the doublet with a symbol
        new_trees_1 = [_replace_asym_quartets_symbol(i) for i in all_trees_sorted]
        # then, replace all other cells 
        new_trees_2 = [_replace_all(i, non_asym_quartets_true, replacement_bool) for i in new_trees_1]
        # then, replace the symbols
        new_trees_3 = [_replace_symbols(i, asym_quartets_true, replacement_bool) for i in new_trees_2]
        df_asym_quintets_new = _make_df_asym_quintets(new_trees_3, asym_quintet_dict, resample, False)
        dfs_asym_quintets_new.append(df_asym_quintets_new)
        
    dfs_c = _process_dfs_asym_quintet(df_asym_quintets_true, dfs_asym_quintets_new, num_resamples, asym_quintet_dict, asym_quartet_dict, cell_dict, df_asym_quartets_true, df_non_asym_quartets_true, calc_expected)
    
    return (asym_quintet_dict, cell_fates, dfs_c)

### for asymmetric sextet analysis

def _make_asym_sextet_dict(cell_fates):
    """Makes a dictionary of all possible asymmetric sextets.
    
    Args:
        cell_fates (list): List with each entry as a cell fate.
    
    Returns:
        asym_sextet_dict (dict): Keys are asymmetric sextets, values are integers.
    """

    total = '0123456789'
    asym_sextet_combinations = []
    for f in cell_fates:
        for g in cell_fates:
            for h in cell_fates:
                for i in cell_fates:
                    for j in list(combinations_with_replacement(total[:len(cell_fates)],2)):
                        #print(j)
                        k = sorted([cell_fates[int(j[0])], cell_fates[int(j[1])]])
                        asym_sextet = f"({f},({g},({h},({i},({k[0]},{k[1]})))))"
                        asym_sextet_combinations.append(asym_sextet)

    asym_sextet_dict = {}
    for i, j in enumerate(asym_sextet_combinations):
        asym_sextet_dict[j] = i
    return asym_sextet_dict

def _replace_asym_quintets_blank(tree):
    """Erases all asymmetric quintets in tree.
    
    Args:
        tree (string): tree in NEWICK format.
    
    Returns:
        new_tree (string): tree in NEWICK format without asym_quintets.
    """
    def repl_asym_quintets_blank(var):
        return ''
    new_tree = re.sub("\(\w,\(\w,\(\w,\(\w,\w\)\)\)\)", repl_asym_quintets_blank, tree)
    return new_tree

# returns relavent subtrees
def _flatten_non_asym_quintets(all_trees_sorted):
    """Returns all cells outside asymmetric quintets in list of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        non_asym_quintets (list): List with each entry as a non_asym_quintet (string).
    """
    x_asym_quintets = [_replace_asym_quintets_blank(i) for i in all_trees_sorted]
    non_asym_quintets = []
    for i in x_asym_quintets:
        for j in re.findall("[A-Za-z0-9]+", i):
            non_asym_quintets.extend(j)
    return non_asym_quintets

# returns relavent subtrees
def _flatten_asym_sextets(all_trees_sorted):
    """Makes a list of all asymmetric quintets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        asym_sextets (list): List with each entry as an asymmetric quintet (string).
    """
    asym_sextets = []
    for i in all_trees_sorted:
        asym_sextets.extend(re.findall("\(\w,\(\w,\(\w,\(\w,\(\w,\w\)\)\)\)\)", i))
    return asym_sextets

def _replace_asym_quintets_symbol(tree):
    """Replaces all asymmetric quintets in tree with "?".
    
    Args:
        tree (string): Tree in NEWICK format.
    
    Returns:
        new_tree (string): Tree in NEWICK format, asymmetric quintets replaced with "?".
    """
    def repl_asym_quintets_symbol(var):
        return '?'
    new_tree = re.sub("\(\w,\(\w,\(\w,\(\w,\w\)\)\)\)", repl_asym_quintets_symbol, tree)
    return new_tree

def _process_dfs_asym_sextet(df_asym_sextets_true, dfs_asym_sextets_new, num_resamples, asym_sextet_dict, asym_quintet_dict, cell_dict, df_asym_quintets_true, df_non_asym_quintets_true, calc_expected=True):
    """Arranges observed counts for each asymmetric quintet in all resamples and original trees into a combined DataFrame.
    
    Last column is analytically solved expected number of each asym_quintet.
        
    Args:
        df_asym_sextet_true (DataFrame): DataFrame with number of each asymmetric quintet in original trees, indexed by `asym_sextet_dict`.
        dfs_asym_sextet_new (list): List with each entry as DataFrame of number of each asymmetric quintet in each set 
            of resampled trees, indexed by `asym_sextet_dict`.
        num_resamples (int): Number of resample datasets.
        asym_sextet_dict (dict): Keys are asymmetric quintets, values are integers.
        asym_quintet_dict (dict): Keys are asym_quintets, values are integers.
        cell_dict (dict): Keys are cell types, values are integers.
        df_asym_quintets_true (DataFrame): DataFrame with number of each asymmetric quintet in original trees, indexed by `asym_quintet_dict`.
        df_non_asym_quintets_true (DataFrame): DataFrame with number of each cell fate in original trees, indexed by `cell_dict`.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        dfs_c (DataFrame): Indexed by values from `asym_sextet_dict`.
            Last column is analytically solved expected number of each asymmetric quintet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    
    """
    
    dfs_list = [dfs_asym_sextets_new[i] for i in range(num_resamples)] + [df_asym_sextets_true]
    dfs_c = pd.concat(dfs_list, axis=1, sort=False)
    
    dfs_c.fillna(0, inplace=True)

    # for asymmetric sextet df
    empty_indices = [i for i in range(0,len(asym_sextet_dict)) if i not in dfs_c.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        df_to_append_list.append(df_to_append)
    dfs_c = pd.concat([dfs_c]+df_to_append_list, axis=0)
    dfs_c.sort_index(inplace=True)

    if calc_expected==False:
        return dfs_c
    
    # for non_asym_quintets df
    empty_indices = [i for i in range(0,len(cell_dict)) if i not in df_non_asym_quintets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_non_asym_quintets_true = pd.concat([df_non_asym_quintets_true]+df_to_append_list, axis=0)

    df_non_asym_quintets_true_norm = df_non_asym_quintets_true/df_non_asym_quintets_true.sum()
    df_non_asym_quintets_true_norm = df_non_asym_quintets_true_norm.rename({v: k for k, v in cell_dict.items()})
    
    # for asymmetric quintets df
    empty_indices = [i for i in range(0,len(asym_quintet_dict)) if i not in df_asym_quintets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_asym_quintets_true = pd.concat([df_asym_quintets_true]+df_to_append_list, axis=0)

    df_asym_quintets_true_norm = df_asym_quintets_true/df_asym_quintets_true.sum()
    df_asym_quintets_true_norm = df_asym_quintets_true_norm.rename({v: k for k, v in asym_quintet_dict.items()})
    
    expected_list = []
    for key in tqdm(asym_sextet_dict.keys()):
        cell_1 = key[1]
        cell_2 = key[3:-1]
        #print(cell_1, cell_2)
        p_cell_1 = df_non_asym_quintets_true_norm.loc[cell_1].values[0]
        p_cell_2 = df_asym_quintets_true_norm.loc[cell_2].values[0]
        #print(p_cell_1, p_cell_2)
        expected = dfs_c.sum()[0]*p_cell_1*p_cell_2
        #print(expected)
        expected_list.append(expected)
        
    dfs_c = dfs_c.copy()
    dfs_c['expected'] = expected_list
    dfs_c.fillna(0, inplace=True)
    
    return dfs_c

def _make_df_asym_sextets(all_trees_sorted, asym_sextet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all asym_sextets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        asym_sextet_dict (dict): Keys are asym_sextets, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): if True, then index of resulting DataFrame uses `asym_sextet_dict` keys.
            
    Returns:
        df_asym_sextets (DataFrame): Rows are asym_sextets, column is resample number.
    """
    asym_sextets = _flatten_asym_sextets(all_trees_sorted)
    asym_sextets_resample_index = [asym_sextet_dict[i] for i in asym_sextets]
    df_asym_sextets = pd.DataFrame.from_dict(Counter(asym_sextets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_asym_sextets = df_asym_sextets.rename({v: k for k, v in asym_sextet_dict.items()})
    return df_asym_sextets

def _make_df_non_asym_quintets(all_trees_sorted, cell_dict, resample, labels_bool=False):
    """Makes a DataFrame of all non_asym_quintets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        cell_dict (dict): Keys are cell types, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): If True, then index of resulting DataFrame uses `cell_dict` keys.
    
    Returns:
        df_non_asym_quintets (DataFrame): Rows are non_asym_quintets, column is resample number.
    """
    non_asym_quintets = _flatten_non_asym_quintets(all_trees_sorted)
    non_asym_quintets_resample_index = [cell_dict[i] for i in non_asym_quintets]
    df_non_asym_quintets = pd.DataFrame.from_dict(Counter(non_asym_quintets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_non_asym_quintets = df_non_asym_quintets.rename({v: k for k, v in cell_dict.items()})
    return df_non_asym_quintets

def resample_trees_asym_sextets(all_trees_sorted, 
                            num_resamples=10000, 
                            replacement_bool=True,
                            cell_fates='auto', 
                            calc_expected=True
                           ):
    """Performs resampling of tree, drawing with or without replacement, returning subtree dictionary and DataFrame containing 
    number of asymmetric quintets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done via (1) replacing each asymmetric quintet with a randomly chosen asymmetric quintet across all trees and 
    (2) replacing every other cell with a randomly chosen non-asymmetric quintet cell across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        (tuple): Contains the following variables.
        - asym_sextet_dict (dict): Keys are asym_sextets, values are integers.
        - cell_fates (list): List where each entry is a string representing a cell fate.
        - dfs_c (DataFrame): Indexed by values from `asym_sextet_dict`.
            Last column is analytically solved expected number of each asym_sextet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    """
    # automatically determine cell fates if not explicitly provided
    if cell_fates == 'auto':
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted for i in sublist])))))
    
    # _make_subtree_dict functions can only handle 10 cell fates max
    if len(cell_fates)>10:
        print('warning, _make_subtree_dict functions can only handle 10 cell fates max!')
      
    asym_sextet_dict = _make_asym_sextet_dict(cell_fates)
    asym_quintet_dict = _make_asym_quintet_dict(cell_fates)
    cell_dict = _make_cell_dict(cell_fates)
    
    # store result for each rearrangement in dfs list
    dfs_asym_sextets_new = []
    df_asym_sextets_true = _make_df_asym_sextets(all_trees_sorted, asym_sextet_dict, 'observed', False)
    df_asym_quintets_true = _make_df_asym_quintets(all_trees_sorted, asym_quintet_dict, 'observed', False)
    df_non_asym_quintets_true = _make_df_non_asym_quintets(all_trees_sorted, cell_dict, 'observed', False)

    # rearrange leaves num_resamples times
    for resample in tqdm(range(0, num_resamples)):
        asym_quintets_true = _flatten_asym_quintets(all_trees_sorted)
        non_asym_quintets_true = _flatten_non_asym_quintets(all_trees_sorted)
        
        # shuffle if replacement=False
        if replacement_bool==False:
            random.shuffle(asym_quintets_true)
            random.shuffle(non_asym_quintets_true)
        
        # first, replace the doublet with a symbol
        new_trees_1 = [_replace_asym_quintets_symbol(i) for i in all_trees_sorted]
        # then, replace all other cells 
        new_trees_2 = [_replace_all(i, non_asym_quintets_true, replacement_bool) for i in new_trees_1]
        # then, replace the symbols
        new_trees_3 = [_replace_symbols(i, asym_quintets_true, replacement_bool) for i in new_trees_2]
        df_asym_sextets_new = _make_df_asym_sextets(new_trees_3, asym_sextet_dict, resample, False)
        dfs_asym_sextets_new.append(df_asym_sextets_new)
        
    dfs_c = _process_dfs_asym_sextet(df_asym_sextets_true, dfs_asym_sextets_new, num_resamples, asym_sextet_dict, asym_quintet_dict, cell_dict, df_asym_quintets_true, df_non_asym_quintets_true, calc_expected)
    
    return (asym_sextet_dict, cell_fates, dfs_c)

### for asymmetric septet analysis

def _make_asym_septet_dict(cell_fates):
    """Makes a dictionary of all possible asymmetric septets.
    
    Args:
        cell_fates (list): List with each entry as a cell fate.
    
    Returns:
        asym_septet_dict (dict): Keys are asymmetric septets, values are integers.
    """

    total = '0123456789'
    asym_septet_combinations = []
    for e in cell_fates:
        for f in cell_fates:
            for g in cell_fates:
                for h in cell_fates:
                    for i in cell_fates:
                        for j in list(combinations_with_replacement(total[:len(cell_fates)],2)):
                            #print(j)
                            k = sorted([cell_fates[int(j[0])], cell_fates[int(j[1])]])
                            asym_septet = f"({e},({f},({g},({h},({i},({k[0]},{k[1]}))))))"
                            asym_septet_combinations.append(asym_septet)

    asym_septet_dict = {}
    for i, j in enumerate(asym_septet_combinations):
        asym_septet_dict[j] = i
    return asym_septet_dict

def _replace_asym_sextets_blank(tree):
    """Erases all asymmetric sextets in tree.
    
    Args:
        tree (string): tree in NEWICK format.
    
    Returns:
        new_tree (string): tree in NEWICK format without asym_sextets.
    """
    def repl_asym_sextets_blank(var):
        return ''
    new_tree = re.sub("\(\w,\(\w,\(\w,\(\w,\(\w,\w\)\)\)\)\)", repl_asym_sextets_blank, tree)
    return new_tree

# returns relavent subtrees
def _flatten_non_asym_sextets(all_trees_sorted):
    """Returns all cells outside asymmetric sextets in list of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        non_asym_sextets (list): List with each entry as a non_asym_sextet (string).
    """
    x_asym_sextets = [_replace_asym_sextets_blank(i) for i in all_trees_sorted]
    non_asym_sextets = []
    for i in x_asym_sextets:
        for j in re.findall("[A-Za-z0-9]+", i):
            non_asym_sextets.extend(j)
    return non_asym_sextets

# returns relavent subtrees
def _flatten_asym_septets(all_trees_sorted):
    """Makes a list of all asymmetric sextets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        asym_septets (list): List with each entry as an asymmetric sextet (string).
    """
    asym_septets = []
    for i in all_trees_sorted:
        asym_septets.extend(re.findall("\(\w,\(\w,\(\w,\(\w,\(\w,\(\w,\w\)\)\)\)\)\)", i))
    return asym_septets

def _replace_asym_sextets_symbol(tree):
    """Replaces all asymmetric sextets in tree with "?".
    
    Args:
        tree (string): Tree in NEWICK format.
    
    Returns:
        new_tree (string): Tree in NEWICK format, asymmetric sextets replaced with "?".
    """
    def repl_asym_sextets_symbol(var):
        return '?'
    new_tree = re.sub("\(\w,\(\w,\(\w,\(\w,\(\w,\w\)\)\)\)\)", repl_asym_sextets_symbol, tree)
    return new_tree

def _process_dfs_asym_septet(df_asym_septets_true, dfs_asym_septets_new, num_resamples, asym_septet_dict, asym_sextet_dict, cell_dict, df_asym_sextets_true, df_non_asym_sextets_true, calc_expected=True):
    """Arranges observed counts for each asymmetric septet in all resamples and original trees into a combined DataFrame.
    
    Last column is analytically solved expected number of each asym_sextet.
        
    Args:
        df_asym_septet_true (DataFrame): DataFrame with number of each asymmetric sextet in original trees, indexed by `asym_septet_dict`.
        dfs_asym_septet_new (list): List with each entry as DataFrame of number of each asymmetric sextet in each set 
            of resampled trees, indexed by `asym_septet_dict`.
        num_resamples (int): Number of resample datasets.
        asym_septet_dict (dict): Keys are asymmetric septets, values are integers.
        asym_sextet_dict (dict): Keys are asym_sextets, values are integers.
        cell_dict (dict): Keys are cell types, values are integers.
        df_asym_sextets_true (DataFrame): DataFrame with number of each asymmetric sextet in original trees, indexed by `asym_sextet_dict`.
        df_non_asym_sextets_true (DataFrame): DataFrame with number of each cell fate in original trees, indexed by `cell_dict`.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        dfs_c (DataFrame): Indexed by values from `asym_septet_dict`.
            Last column is analytically solved expected number of each asymmetric sextet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    
    """
    
    dfs_list = [dfs_asym_septets_new[i] for i in range(num_resamples)] + [df_asym_septets_true]
    dfs_c = pd.concat(dfs_list, axis=1, sort=False)
    
    dfs_c.fillna(0, inplace=True)

    # for asymmetric septet df
    empty_indices = [i for i in range(0,len(asym_septet_dict)) if i not in dfs_c.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        df_to_append_list.append(df_to_append)
    dfs_c = pd.concat([dfs_c]+df_to_append_list, axis=0)
    dfs_c.sort_index(inplace=True)

    if calc_expected==False:
        return dfs_c
    
    # for non_asym_sextets df
    empty_indices = [i for i in range(0,len(cell_dict)) if i not in df_non_asym_sextets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_non_asym_sextets_true = pd.concat([df_non_asym_sextets_true]+df_to_append_list, axis=0)

    df_non_asym_sextets_true_norm = df_non_asym_sextets_true/df_non_asym_sextets_true.sum()
    df_non_asym_sextets_true_norm = df_non_asym_sextets_true_norm.rename({v: k for k, v in cell_dict.items()})
    
    # for asymmetric sextets df
    empty_indices = [i for i in range(0,len(asym_sextet_dict)) if i not in df_asym_sextets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_asym_sextets_true = pd.concat([df_asym_sextets_true]+df_to_append_list, axis=0)

    df_asym_sextets_true_norm = df_asym_sextets_true/df_asym_sextets_true.sum()
    df_asym_sextets_true_norm = df_asym_sextets_true_norm.rename({v: k for k, v in asym_sextet_dict.items()})
    
    expected_list = []
    for key in tqdm(asym_septet_dict.keys()):
        cell_1 = key[1]
        cell_2 = key[3:-1]
        #print(cell_1, cell_2)
        p_cell_1 = df_non_asym_sextets_true_norm.loc[cell_1].values[0]
        p_cell_2 = df_asym_sextets_true_norm.loc[cell_2].values[0]
        #print(p_cell_1, p_cell_2)
        expected = dfs_c.sum()[0]*p_cell_1*p_cell_2
        #print(expected)
        expected_list.append(expected)
        
    dfs_c = dfs_c.copy()
    dfs_c['expected'] = expected_list
    dfs_c.fillna(0, inplace=True)
    
    return dfs_c

def _make_df_asym_septets(all_trees_sorted, asym_septet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all asym_septets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        asym_septet_dict (dict): Keys are asym_septets, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): if True, then index of resulting DataFrame uses `asym_septet_dict` keys.
            
    Returns:
        df_asym_septets (DataFrame): Rows are asym_septets, column is resample number.
    """
    asym_septets = _flatten_asym_septets(all_trees_sorted)
    asym_septets_resample_index = [asym_septet_dict[i] for i in asym_septets]
    df_asym_septets = pd.DataFrame.from_dict(Counter(asym_septets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_asym_septets = df_asym_septets.rename({v: k for k, v in asym_septet_dict.items()})
    return df_asym_septets

def _make_df_non_asym_sextets(all_trees_sorted, cell_dict, resample, labels_bool=False):
    """Makes a DataFrame of all non_asym_sextets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        cell_dict (dict): Keys are cell types, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): If True, then index of resulting DataFrame uses `cell_dict` keys.
    
    Returns:
        df_non_asym_sextets (DataFrame): Rows are non_asym_sextets, column is resample number.
    """
    non_asym_sextets = _flatten_non_asym_sextets(all_trees_sorted)
    non_asym_sextets_resample_index = [cell_dict[i] for i in non_asym_sextets]
    df_non_asym_sextets = pd.DataFrame.from_dict(Counter(non_asym_sextets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_non_asym_sextets = df_non_asym_sextets.rename({v: k for k, v in cell_dict.items()})
    return df_non_asym_sextets

def resample_trees_asym_septets(all_trees_sorted, 
                            num_resamples=10000, 
                            replacement_bool=True,
                            cell_fates='auto',
                            calc_expected=True
                           ):
    """Performs resampling of tree, drawing with or without replacement, returning subtree dictionary and DataFrame containing 
    number of asymmetric septets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done via (1) replacing each asymmetric sextet with a randomly chosen asymmetric sextet across all trees and 
    (2) replacing every other cell with a randomly chosen non-asymmetric sextet cell across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        (tuple): Contains the following variables.
        - asym_septet_dict (dict): Keys are asym_septets, values are integers.
        - cell_fates (list): List where each entry is a string representing a cell fate.
        - dfs_c (DataFrame): Indexed by values from `asym_septet_dict`.
            Last column is analytically solved expected number of each asym_septet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    """
    # automatically determine cell fates if not explicitly provided
    if cell_fates == 'auto':
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted for i in sublist])))))
    
    # _make_subtree_dict functions can only handle 10 cell fates max
    if len(cell_fates)>10:
        print('warning, _make_subtree_dict functions can only handle 10 cell fates max!')
      
    asym_septet_dict = _make_asym_septet_dict(cell_fates)
    asym_sextet_dict = _make_asym_sextet_dict(cell_fates)
    cell_dict = _make_cell_dict(cell_fates)
    
    # store result for each rearrangement in dfs list
    dfs_asym_septets_new = []
    df_asym_septets_true = _make_df_asym_septets(all_trees_sorted, asym_septet_dict, 'observed', False)
    df_asym_sextets_true = _make_df_asym_sextets(all_trees_sorted, asym_sextet_dict, 'observed', False)
    df_non_asym_sextets_true = _make_df_non_asym_sextets(all_trees_sorted, cell_dict, 'observed', False)

    # rearrange leaves num_resamples times
    for resample in tqdm(range(0, num_resamples)):
        asym_sextets_true = _flatten_asym_sextets(all_trees_sorted)
        non_asym_sextets_true = _flatten_non_asym_sextets(all_trees_sorted)
        
        # shuffle if replacement=False
        if replacement_bool==False:
            random.shuffle(asym_sextets_true)
            random.shuffle(non_asym_sextets_true)
        
        # first, replace the doublet with a symbol
        new_trees_1 = [_replace_asym_sextets_symbol(i) for i in all_trees_sorted]
        # then, replace all other cells 
        new_trees_2 = [_replace_all(i, non_asym_sextets_true, replacement_bool) for i in new_trees_1]
        # then, replace the symbols
        new_trees_3 = [_replace_symbols(i, asym_sextets_true, replacement_bool) for i in new_trees_2]
        df_asym_septets_new = _make_df_asym_septets(new_trees_3, asym_septet_dict, resample, False)
        dfs_asym_septets_new.append(df_asym_septets_new)
        
    dfs_c = _process_dfs_asym_septet(df_asym_septets_true, dfs_asym_septets_new, num_resamples, asym_septet_dict, asym_sextet_dict, cell_dict, df_asym_sextets_true, df_non_asym_sextets_true, calc_expected)
    
    return (asym_septet_dict, cell_fates, dfs_c)

### for sextet analysis

def _make_sextet_dict(cell_fates):
    """Makes a dictionary of all possible sextets.
    
    Args:
        cell_fates (list): List with each entry as a cell fate.
    
    Returns:
        sextet_dict (dict): Keys are sextets, values are integers.
    """

    doublet_dict = _make_doublet_dict(cell_fates)
    quartet_dict = _make_quartet_dict(cell_fates)
    
    sextet_combinations = []
    for i in doublet_dict.keys():
        for j in quartet_dict.keys():
            sextet = f"({i},{j})"
            sextet_combinations.append(sextet)

    sextet_dict = {}
    for i, j in enumerate(sextet_combinations):
        sextet_dict[j] = i
    return sextet_dict


def _replace_quartets_blank(tree):
    """Erases all quartets in tree.
    
    Args:
        tree (string): tree in NEWICK format.
    
    Returns:
        new_tree (string): tree in NEWICK format without quartets.
    """
    def repl_quartets_blank(var):
        return ''
    new_tree = re.sub("\(\(\w,\w\),\(\w,\w\)\)", repl_quartets_blank, tree)
    return new_tree

# returns relavent subtrees
def _flatten_doublets_non_quartets(all_trees_sorted):
    """Returns all doublets outside quartets in list of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        doublets_non_quartets (list): List with each entry as a doublet_non_quartet (string).
    """
    x_quartets = [_replace_quartets_blank(i) for i in all_trees_sorted]
    doublets_non_quartets = []
    for i in x_quartets:
        doublets_non_quartets.extend(re.findall("\(\w,\w\)", i))
    return doublets_non_quartets

# returns relavent subtrees
def _flatten_sextets(all_trees_sorted):
    """Makes a list of all sextets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        sextets (list): List with each entry as an asymmetric quartet (string).
    """
    sextets = []
    for i in all_trees_sorted:
        sextets.extend(re.findall("\(\(\w,\w\),\(\(\w,\w\),\(\w,\w\)\)\)", i))
    return sextets

def _replace_quartets_symbol(tree):
    """Replaces all quartets in tree with "?".
    
    Args:
        tree (string): Tree in NEWICK format.
    
    Returns:
        new_tree (string): Tree in NEWICK format, quartets replaced with "?".
    """
    def repl_quartets_symbol(var):
        return '?'
    new_tree = re.sub("\(\(\w,\w\),\(\w,\w\)\)", repl_quartets_symbol, tree)
    return new_tree

def _process_dfs_sextet(df_sextets_true, dfs_sextets_new, num_resamples, sextet_dict, quartet_dict, doublet_dict, df_quartets_true, df_doublets_non_quartets_true, calc_expected=True):
    """Arranges observed counts for each sextet in all resamples and original trees into a combined DataFrame.
    
    Last column is analytically solved expected number of each sextet.
        
    Args:
        df_sextet_true (DataFrame): DataFrame with number of each sextet in original trees, indexed by `sextet_dict`.
        dfs_sextet_new (list): List with each entry as DataFrame of number of each sextet in each set 
            of resampled trees, indexed by `sextet_dict`.
        num_resamples (int): Number of resample datasets.
        sextet_dict (dict): Keys are sextets, values are integers.
        quartet_dict (dict): Keys are quartets, values are integers.
        doublet_dict (dict): Keys are doublets, values are integers.
        df_quartets_true (DataFrame): DataFrame with number of each quartet in original trees, indexed by `quartet_dict`.
        df_doublets_non_quartets_true (DataFrame): DataFrame with number of each non-quartet doublet in original trees, indexed by `doublet_dict`.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        dfs_c (DataFrame): Indexed by values from `sextet_dict`.
            Last column is analytically solved expected number of each sextet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    
    """
    
    dfs_list = [dfs_sextets_new[i] for i in range(num_resamples)] + [df_sextets_true]
    dfs_c = pd.concat(dfs_list, axis=1, sort=False)
    
    dfs_c.fillna(0, inplace=True)

    # for sextet df
    empty_indices = [i for i in range(0,len(sextet_dict)) if i not in dfs_c.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        df_to_append_list.append(df_to_append)
    dfs_c = pd.concat([dfs_c]+df_to_append_list, axis=0)
    dfs_c.sort_index(inplace=True)

    if calc_expected==False:
        return dfs_c
    
    # for doublets_non_quartet df
    empty_indices = [i for i in range(0,len(doublet_dict)) if i not in df_doublets_non_quartets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_doublets_non_quartets_true = pd.concat([df_doublets_non_quartets_true]+df_to_append_list, axis=0)

    df_doublets_non_quartets_true_norm = df_doublets_non_quartets_true/df_doublets_non_quartets_true.sum()
    df_doublets_non_quartets_true_norm = df_doublets_non_quartets_true_norm.rename({v: k for k, v in doublet_dict.items()})
    
    # for quartets df
    empty_indices = [i for i in range(0,len(quartet_dict)) if i not in df_quartets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_quartets_true = pd.concat([df_quartets_true]+df_to_append_list, axis=0)

    df_quartets_true_norm = df_quartets_true/df_quartets_true.sum()
    df_quartets_true_norm = df_quartets_true_norm.rename({v: k for k, v in quartet_dict.items()})
    
    expected_list = []
    for key in sextet_dict.keys():
        cell_1 = key[1:6]
        cell_2 = key[7:-1]
        #print(cell_1, cell_2)
        p_cell_1 = df_doublets_non_quartets_true_norm.loc[cell_1].values[0]
        p_cell_2 = df_quartets_true_norm.loc[cell_2].values[0]
        #print(p_cell_1, p_cell_2)
        expected = dfs_c.sum()[0]*p_cell_1*p_cell_2
        #print(expected)
        expected_list.append(expected)
        
    dfs_c = dfs_c.copy()
    dfs_c['expected'] = expected_list
    dfs_c.fillna(0, inplace=True)
    
    return dfs_c

def _make_df_sextets(all_trees_sorted, sextet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all sextets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        sextet_dict (dict): Keys are sextets, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): if True, then index of resulting DataFrame uses `sextet_dict` keys.
            
    Returns:
        df_sextets (DataFrame): Rows are sextets, column is resample number.
    """
    sextets = _flatten_sextets(all_trees_sorted)
    sextets_resample_index = [sextet_dict[i] for i in sextets]
    df_sextets = pd.DataFrame.from_dict(Counter(sextets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_sextets = df_sextets.rename({v: k for k, v in sextet_dict.items()})
    return df_sextets

def _make_df_doublets_non_quartets(all_trees_sorted, doublet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all doublets_non_quartets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        doublet_dict (dict): Keys are doublets, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): If True, then index of resulting DataFrame uses `doublet_dict` keys.
    
    Returns:
        df_doublets_non_quartets (DataFrame): Rows are doublets_non_quartets, column is resample number.
    """
    doublets_non_quartets = _flatten_doublets_non_quartets(all_trees_sorted)
    doublets_non_quartets_resample_index = [doublet_dict[i] for i in doublets_non_quartets]
    df_doublets_non_quartets = pd.DataFrame.from_dict(Counter(doublets_non_quartets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_doublets_non_quartets = df_doublets_non_quartets.rename({v: k for k, v in doublet_dict.items()})
    return df_doublets_non_quartets

def resample_trees_sextets(all_trees_sorted, 
                            num_resamples=10000, 
                            replacement_bool=True,
                            cell_fates='auto', 
                            calc_expected=True
                           ):
    """Performs resampling of tree, drawing with or without replacement, returning subtree dictionary and DataFrame containing 
    number of sextets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done via (1) replacing each quartet with a randomly chosen quartet across all trees and 
    (2) replacing every other doublet with a randomly chosen non-quartet doublet across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        (tuple): Contains the following variables.
        - sextet_dict (dict): Keys are sextets, values are integers.
        - cell_fates (list): List where each entry is a string representing a cell fate.
        - dfs_c (DataFrame): Indexed by values from `sextet_dict`.
            Last column is analytically solved expected number of each sextet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    """
    # automatically determine cell fates if not explicitly provided
    if cell_fates == 'auto':
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted for i in sublist])))))
    
    # _make_subtree_dict functions can only handle 10 cell fates max
    if len(cell_fates)>10:
        print('warning!')
      
    sextet_dict = _make_sextet_dict(cell_fates)
    quartet_dict = _make_quartet_dict(cell_fates)
    doublet_dict = _make_doublet_dict(cell_fates)
    
    # store result for each rearrangement in dfs list
    dfs_sextets_new = []
    df_sextets_true = _make_df_sextets(all_trees_sorted, sextet_dict, 'observed', False)
    df_quartets_true = _make_df_quartets(all_trees_sorted, quartet_dict, 'observed', False)
    df_doublets_non_quartets_true = _make_df_doublets_non_quartets(all_trees_sorted, doublet_dict, 'observed', False)

    # rearrange leaves num_resamples times
    for resample in tqdm(range(0, num_resamples)):
        quartets_true = _flatten_quartets(all_trees_sorted)
        doublets_non_quartets_true = _flatten_doublets_non_quartets(all_trees_sorted)
        
        # shuffle if replacement=False
        if replacement_bool==False:
            random.shuffle(quartets_true)
            random.shuffle(doublets_non_quartets_true)
        
        # first, replace the doublet with a symbol
        new_trees_1 = [_replace_quartets_symbol(i) for i in all_trees_sorted]
        # then, replace all other cells 
        new_trees_2 = [_replace_doublets(i, doublets_non_quartets_true, replacement_bool) for i in new_trees_1]
        # then, replace the symbols
        new_trees_3 = [_replace_symbols(i, quartets_true, replacement_bool) for i in new_trees_2]
        df_sextets_new = _make_df_sextets(new_trees_3, sextet_dict, resample, False)
        dfs_sextets_new.append(df_sextets_new)
        
    dfs_c = _process_dfs_sextet(df_sextets_true, dfs_sextets_new, num_resamples, sextet_dict, quartet_dict, doublet_dict, df_quartets_true, df_doublets_non_quartets_true, calc_expected)
    
    return (sextet_dict, cell_fates, dfs_c)

### for octet analysis

def _make_octet_dict(cell_fates):
    """Makes a dictionary of all possible octets.
    
    Args:
        cell_fates (list): List with each entry as a cell fate.
    
    Returns:
        octet_dict (dict): Keys are octets, values are integers.
    """

    quartet_dict = _make_quartet_dict(cell_fates)
    z = [sorted([i, j]) for i in list(quartet_dict.keys()) for j in list(quartet_dict.keys())]
    x = [f'({i[0]},{i[1]})' for i in z]

    # get rid of duplicates
    y = []
    for i in tqdm(x):
        if i not in y:
            y.append(i)
        
    octet_dict = {}
    for i, j in enumerate(y):
        octet_dict[j] = i
    return octet_dict

# returns relavent subtrees
def _flatten_octets(all_trees):
    """Makes a list of all octets in set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
    
    Returns:
        octets (list): List with each entry as a octet (string).
    """
    octets = []
    for i in all_trees:
        octets.extend(re.findall("\(\(\(\w,\w\),\(\w,\w\)\),\(\(\w,\w\),\(\w,\w\)\)\)", i))
    return octets


def _make_df_octets(all_trees_sorted, octet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all octets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        octet_dict (dict): Keys are octets, values are integers.
        resample (int): Resample number.
        labels_bool (bool, optional): if True, then index of resulting DataFrame uses `octet_dict` keys.
    
    Returns:
        df_octets (DataFrame): Rows are octets, column is resample number.
    """
    octets = _flatten_octets(all_trees_sorted)
    octets_resample_index = [octet_dict[i] for i in octets]
    df_octets = pd.DataFrame.from_dict(Counter(octets_resample_index), orient='index', columns=[f"{resample}"])
    if labels_bool == True:
        df_octets = df_octets.rename({v: k for k, v in octet_dict.items()})
    return df_octets


# Replace quartets drawing from quartets_true
def _replace_quartets(tree, quartets_true, replacement_bool):
    """Replaces all quartets in tree with a new quartet drawing from `quartets_true`.
    
    Args:
        tree (string): Tree in NEWICK format.
        quartets_true (list): List with each entry as a quartet (string).
        replacement_bool (bool): Draw with or without replacement from `quartets_true`.
    
    Returns:
        new_tree_sorted_octet (string): Tree in NEWICK format, quartets replaced with new quartets, 
            and all quartets/octets in alphabetical order.
    """
    if replacement_bool==False:
        def repl_quartet(var):
            return quartets_true.pop()
    elif replacement_bool==True:
        def repl_quartet(var):
            return random.choice(quartets_true)
    new_tree = re.sub("\(\(\w,\w\),\(\w,\w\)\)", repl_quartet, tree)
    new_tree_sorted_octet = sort_align_tree(new_tree)
    return new_tree_sorted_octet


def _process_dfs_octet(df_octets_true, dfs_octets_new, num_resamples, octet_dict, quartet_dict, df_quartets_true, calc_expected=True):
    """Arranges observed counts for each octet in all resamples and original trees into a combined DataFrame.
    
    Last column is analytically solved expected number of each octet.
        
    Args:
        df_octet_true (DataFrame): DataFrame with number of each octet in original trees, indexed by `octet_dict`.
        dfs_octet_new (list): List with each entry as DataFrame of number of each octet in each set 
            of resampled trees, indexed by `octet_dict`.
        num_resamples (int): Number of resample datasets.
        octet_dict (dict): Keys are octets, values are integers.
        quartet_dict (dict): Keys are quartets, values are integers.
        df_quartets_true (DataFrame): DataFrame with number of each quartet in original trees, indexed by `quartet_dict`.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        dfs_c (DataFrame): Indexed by values from `octet_dict`.
            Last column is analytically solved expected number of each octet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.
    
    """
    
    dfs_list = [dfs_octets_new[i] for i in range(num_resamples)] + [df_octets_true]
    dfs_c = pd.concat(dfs_list, axis=1, sort=False)
    
    dfs_c.fillna(0, inplace=True)

    # for octet df
    empty_indices = [i for i in range(0,len(octet_dict)) if i not in dfs_c.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        df_to_append_list.append(df_to_append)
    dfs_c = pd.concat([dfs_c]+df_to_append_list, axis=0)
    dfs_c.sort_index(inplace=True)

    if calc_expected==False:
        return dfs_c
    
    # for quartets df
    empty_indices = [i for i in range(0,len(quartet_dict)) if i not in df_quartets_true.index]
    df_to_append_list = []
    for i in tqdm(empty_indices):
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_to_append_list.append(df_to_append)
    df_quartets_true = pd.concat([df_quartets_true]+df_to_append_list, axis=0)

    df_quartets_true_norm = df_quartets_true/df_quartets_true.sum()
    df_quartets_true_norm = df_quartets_true_norm.rename({v: k for k, v in quartet_dict.items()})
    
    expected_list = []
    for key in octet_dict.keys():
        cell_1 = key[1:14]
        cell_2 = key[15:28]
        #print(cell_1, cell_2)
        p_cell_1 = df_quartets_true_norm.loc[cell_1].values[0]
        p_cell_2 = df_quartets_true_norm.loc[cell_2].values[0]
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


def resample_trees_octets(all_trees_sorted, 
                            num_resamples=10000, 
                            replacement_bool=True,
                            cell_fates='auto', 
                            calc_expected=True
                           ):
    """Performs resampling of tree, drawing with or without replacement, returning subtree dictionary and DataFrame containing 
    the number of octets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done via replacing each quartet with a randomly chosen quartet from across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted using the `sort_align_tree` function.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
        calc_expected (Boolean): Calculate expected count if True by multiplying the marginal probabilities of each sub-pattern by the total number of subtrees
    
    Returns:
        (tuple): Contains the following variables.
        - octet_dict (dict): Keys are octets, values are integers.
        - cell_fates (list): List where each entry is a string representing a cell fate.
        - dfs_c (DataFrame): Indexed by values from `octet_dict`.
            Last column is analytically solved expected number of each octet.
            Second to last column is observed number of occurences in the original dataset.
            Rest of columns are the observed number of occurences in the resampled sets.


    """
    # automatically determine cell fates if not explicitly provided
    if cell_fates == 'auto':
        cell_fates = sorted(list(np.unique(re.findall('[A-Z]', ''.join([i for sublist in all_trees_sorted for i in sublist])))))
    
    # _make_subtree_dict functions can only handle 10 cell fates max
    if len(cell_fates)>10:
        print('warning!')
      
    octet_dict = _make_octet_dict(cell_fates)
    quartet_dict = _make_quartet_dict(cell_fates)
    
    # store result for each rearrangement in dfs list
    dfs_octets_new = []
    df_octets_true = _make_df_octets(all_trees_sorted, octet_dict, 'observed', False)
    df_quartets_true = _make_df_quartets(all_trees_sorted, quartet_dict, 'observed', False)

    # rearrange leaves num_resamples times
    for resample in tqdm(range(0, num_resamples)):
        quartets_true = _flatten_quartets(all_trees_sorted)
        
        # shuffle if replacement=False
        if replacement_bool==False:
            random.shuffle(quartets_true)
        
        new_trees = [_replace_quartets(i, quartets_true, replacement_bool) for i in all_trees_sorted]
        df_octets_new = _make_df_octets(new_trees, octet_dict, resample, False)
        dfs_octets_new.append(df_octets_new)
        
    dfs_c = _process_dfs_octet(df_octets_true, dfs_octets_new, num_resamples, octet_dict, quartet_dict, df_quartets_true, calc_expected)
    
    return (octet_dict, cell_fates, dfs_c)

### for multi_dataset resampling

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
