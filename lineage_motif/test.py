### for asymmetric quintet analysis

def make_asym_quintet_dict(cell_fates):
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
    new_tree = re.sub("\(\w*,\(\w*,\(\w*,\w*\)\)\)", repl_asym_quartets_blank, tree)
    return new_tree

# returns relavent subtrees
def _flatten_non_asym_quartets(all_trees_sorted):
    """Returns all cells outside asymmetric quartets in list of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all asymmetric quintets in (x,(x,(x,(x,x)))) format, asymmetric quartets in (x,(x,(x,x))) format, 
            triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
    
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
            Trees are sorted to have all asymmetric quintets in (x,(x,(x,(x,x)))) format, asymmetric quartets in (x,(x,(x,x))) format, 
            triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
    
    Returns:
        asym_quintets (list): List with each entry as an asymmetric quartet (string).
    """
    asym_quintets = []
    for i in all_trees_sorted:
        asym_quintets.extend(re.findall("\(\w*,\(\w*,\(\w*,\(\w*,\w*\)\)\)\)", i))
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
    new_tree = re.sub("\(\w*,\(\w*,\(\w*,\w*\)\)\)", repl_asym_quartets_symbol, tree)
    return new_tree

def _process_dfs_asym_quintet(df_asym_quintets_true, dfs_asym_quintets_new, num_resamples, asym_quintet_dict, asym_quartet_dict, cell_dict, df_asym_quartets_true, df_non_asym_quartets_true):
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
    for i in empty_indices:
        num_zeros = num_resamples+1
        index_to_append = {i: [0]*num_zeros}
        df_to_append = pd.DataFrame(index_to_append)
        df_to_append = df_to_append.transpose()
        df_to_append.columns = dfs_c.columns
        dfs_c = pd.concat([dfs_c, df_to_append], axis=0)
    dfs_c.sort_index(inplace=True)
    
    # for non_asym_quartets df
    empty_indices = [i for i in range(0,len(cell_dict)) if i not in df_non_asym_quartets_true.index]
    for i in empty_indices:
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_non_asym_quartets_true = pd.concat([df_non_asym_quartets_true, df_to_append], axis=0)

    df_non_asym_quartets_true_norm = df_non_asym_quartets_true/df_non_asym_quartets_true.sum()
    df_non_asym_quartets_true_norm = df_non_asym_quartets_true_norm.rename({v: k for k, v in cell_dict.items()})
    
    # for asymmetric quartets df
    empty_indices = [i for i in range(0,len(asym_quartet_dict)) if i not in df_asym_quartets_true.index]
    for i in empty_indices:
        df_to_append = pd.DataFrame([0], index=[i], columns=[f'{num_resamples}'])
        df_asym_quartets_true = pd.concat([df_asym_quartets_true, df_to_append], axis=0)

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

def make_df_asym_quintets(all_trees_sorted, asym_quintet_dict, resample, labels_bool=False):
    """Makes a DataFrame of all asym_quintets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all asymmetric quintets in (x,(x,(x,(x,x)))) format, asymmetric quartets in (x,(x,(x,x))) format, 
            triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
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

def make_df_non_asym_quartets(all_trees_sorted, cell_dict, resample, labels_bool=False):
    """Makes a DataFrame of all non_asym_quartets in the set of trees.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all asymmetric quintets in (x,(x,(x,(x,x)))) format, asymmetric quartets in (x,(x,(x,x))) format, 
            triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
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
                            cell_fates='auto'
                           ):
    """Performs resampling of tree, drawing with or without replacement, returning subtree dictionary and DataFrame containing 
    number of asymmetric quartets across all resamples, the original trees, and the expected number (solved analytically).
    
    Resampling is done via (1) replacing each asymmetric quartet with a randomly chosen asymmetric quartet across all trees and 
    (2) replacing every other cell with a randomly chosen non-asymmetric quartet cell across all trees.
    If `cell_fates` not explicitly provided, use automatically determined cell fates based on tree dataset.
    
    Args:
        all_trees_sorted (list): List where each entry is a string representing a tree in NEWICK format. 
            Trees are sorted to have all asymmetric quintets in (x,(x,(x,(x,x)))) format, asymmetric quartets in (x,(x,(x,x))) format, 
            triplets in (x,(x,x)) format, and all doublets/quartets in alphabetical order.
        num_resamples (int, optional): Number of resample datasets.
        replacement_bool (bool, optional): Sample cells with or without replacement drawing from the pool of all cells.
        cell_fates (string or list, optional): If 'auto' (i.e. not provided by user), automatically determined 
            based on tree dataset. User can also provide list where each entry is a string representing a cell fate.
    
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
    
    # make_subtree_dict functions can only handle 10 cell fates max
    if len(cell_fates)>10:
        print('warning, make_subtree_dict functions can only handle 10 cell fates max!')
      
    asym_quintet_dict = make_asym_quintet_dict(cell_fates)
    asym_quartet_dict = make_asym_quartet_dict(cell_fates)
    cell_dict = make_cell_dict(cell_fates)
    
    # store result for each rearrangement in dfs list
    dfs_asym_quintets_new = []
    df_asym_quintets_true = make_df_asym_quintets(all_trees_sorted, asym_quintet_dict, 'observed', False)
    df_asym_quartets_true = make_df_asym_quartets(all_trees_sorted, asym_quartet_dict, 'observed', False)
    df_non_asym_quartets_true = make_df_non_asym_quartets(all_trees_sorted, cell_dict, 'observed', False)

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
        df_asym_quintets_new = make_df_asym_quintets(new_trees_3, asym_quintet_dict, resample, False)
        dfs_asym_quintets_new.append(df_asym_quintets_new)
        
    dfs_c = _process_dfs_asym_quintet(df_asym_quintets_true, dfs_asym_quintets_new, num_resamples, asym_quintet_dict, asym_quartet_dict, cell_dict, df_asym_quartets_true, df_non_asym_quartets_true)
    
    return (asym_quintet_dict, cell_fates, dfs_c)

