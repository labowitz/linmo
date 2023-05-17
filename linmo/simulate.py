"""
## linmo.simulate
Provides functions for simulating lineage trees.

This module contains the following functions:
- `simulate_tree` - Simulate tree based on provided transition matrix and progenitor/cell type labels.

"""
# +
# packages for both analysis and plotting
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import numba

@numba.njit
def _sample_discrete(probs):
    """Randomly sample an index with probability given by probs."""
    # Generate random number
    q = np.random.rand()
    # Find index
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1

def simulate_tree(transition_matrix, starting_progenitor, labels):
    '''Simulate tree based on provided transition matrix and progenitor/cell type labels.
    Progenitors are represented by lowercase letters, cell types are represented by uppercase letters.
    
    Args:
        transition_matrix (array): matrix where rows represent original state, column represents state to transition into.
            Rows should sum to 1 for progenitors.
        starting_progenitor (string): string with starting progenitor
        labels (string): string with progenitor/cell type labels that correspond to the rows of the provided transition matrix.
    
    Returns:
        tree_input (string): new tree in NEWICK format after simulated division until no progenitor cells remaining.
    '''
    tree_input = starting_progenitor
    # continue dividing if tree contains progenitors
    while re.findall('[a-z]', tree_input) != []: 
        tree_input = _divide(tree_input, transition_matrix, labels)
    return tree_input

def _divide(tree_input, transition_matrix, labels):
    '''Simulates a division in all progenitor cells
    
    Args:
        tree_input (string): tree in NEWICK format
        transition_matrix (array): matrix where rows represent original state, column represents state to transition into.
            Rows should sum to 1 for progenitors.
        labels (string): string with progenitor/cell type labels that correspond to the rows of the provided transition matrix.
    
    Returns:
        tree_input (string): new tree in NEWICK format after simulated division
    '''
    for i in re.findall('[a-z]', tree_input):
        i_escape = re.escape(i)
        i_escape_index = list(labels).index(i_escape)
        transition_probs = transition_matrix[i_escape_index]
        #print(i_escape, i_escape_index, transition_probs)
        
        # substitute the first match with a branch. count=1 ensures that only one progenitor is replaced
        tree_input = re.sub(i_escape, _branch(transition_probs, labels), tree_input, count=1)
    return tree_input

def _branch(transition_probs, labels):
    '''
    Generates a branch with two cells whose identity are chosen using provided `transition_probs`.
    
    Args:
        transition_probs (array): probabilities of generating the progenitors or cell fates indexed as in `labels`.
        labels (string): string with progenitor/cell type labels.
    
    Returns:
        branch_input: string of newly drawn branch. Should have the form (x,x)
    '''
    cell_1 = labels[_sample_discrete(transition_probs)]
    cell_2 = labels[_sample_discrete(transition_probs)]
    branch_input = f'({cell_1},{cell_2})'
    return branch_input