"""
Document Processing Module

This module handles the processing of XML documents, including:
- Document loading and counting
- Structure and content measure calculations
- TF-IDF computation
"""

import fnmatch
import os
from . import preprocessing as prepr
import math

# Module level variables
docs = []
nb_of_docs = 0

def get_docs():
    """
    Get list of XML documents from the DocumentsDB directory.
    
    Returns:
        tuple: (list of document names, number of documents)
    """
    doc_list = []
    doc_count = 0
    db_path = os.path.join(os.getcwd(), "DocumentsDB")
    
    for filename in os.listdir(db_path):
        doc_list.append(filename)
        doc_count += 1
    print(doc_list)
    return doc_list, doc_count

def compute_measures(arr, measure_type):
    """
    Generic function to compute document measures.
    
    Args:
        arr: List of document names
        measure_type: Type of measure to compute ('struct', 'struct_cont', 
                                                'content', 'content_tags')
    Returns:
        dict: Dictionary containing IDF weights
    """
    measure_functions = {
        'struct': prepr.all_paths_weights,
        'struct_cont': prepr.struc_AND_content,
        'content': prepr.contentONLY,
        'content_tags': prepr.tagsANDContent
    }
    
    dict_idf = {}
    dict_tf = {}
    
    for file in arr:
        path = os.path.join(os.getcwd(), "DocumentsDB", file)
        weights_tf = measure_functions[measure_type](path)
        if measure_type == 'struct':
            weights_tf = weights_tf[0]  # Handle tuple return
        dict_tf[file] = weights_tf
    
    temp_keys = []
    temp_values = []
    
    for key, value in dict_tf.items():
        expanded_arr = expand_arr(value)
        temp_keys.append(key)
        temp_values.append(expanded_arr)
    
    contents = compute_content_matrix(temp_values)
    
    for arr in temp_values:
        for i in range(len(arr)):
            word = arr[i][0] if measure_type != 'struct_cont' else [arr[i][0], arr[i][1]]
            n = count_occurrences(contents, word)
            idf = math.log10(nb_of_docs/n)
            arr[i][2 if measure_type != 'struct_cont' else 3] = idf

    for i in range(len(temp_values)):
        dict_idf[temp_keys[i]] = temp_values[i]

    return dict_idf

def run():
    """
    Run the complete document processing pipeline.
    
    Returns:
        tuple: (structure measures, structure+content measures, 
               content measures, content+tags measures)
    """
    global docs, nb_of_docs
    docs, nb_of_docs = get_docs()
    
    return (
        compute_measures(docs, 'struct'),
        compute_measures(docs, 'struct_cont'),
        compute_measures(docs, 'content'),
        compute_measures(docs, 'content_tags')
    )

def expand_arr(arr):
    """
    Expand array to include space for IDF values.
    
    Args:
        arr: Input array of [term, weight] or [term, weight, extra]
        
    Returns:
        list: Expanded array with space for IDF
    """
    to_return = []
    if(len(arr[0]) == 2):
        for elem in arr:
            to_return.append([elem[0], elem[1], 0])
    if(len(arr[0]) == 3):
        for elem in arr:
            to_return.append([elem[0], elem[1], elem[2], 0])
    return to_return

def compute_content_matrix(matrix):
    """
    Convert matrix to content-only format.
    
    Args:
        matrix: Input matrix of terms and weights
        
    Returns:
        list: Matrix with only content terms
    """
    content_matrix = []
    if(len(matrix[0][0]) == 3):
        for elem in matrix:
            vec = []
            for item in elem:
                vec.append(item[0])
            content_matrix.append(vec)
    if(len(matrix[0][0]) == 4):
        for elem in matrix:
            vec = []
            for item in elem:
                vec.append([item[0], item[1]])
            content_matrix.append(vec)
    return content_matrix

def count_occurrences(mat, word):
    """
    Count occurrences of a word in a matrix.
    
    Args:
        mat: Matrix to search in
        word: Word to search for
        
    Returns:
        int: Number of occurrences
    """
    n = 0
    for elem in mat:
        if(word in elem):
            n += 1
    return n

def compute_struct_measures_to_compare(arr, nb_of_docs):
    """
    Compute structural measures for comparison purposes.
    
    Args:
        arr: List of document names
        nb_of_docs: Number of documents
        
    Returns:
        dict: Dictionary containing IDF weights for structural elements
    """
    dict_idf = {}
    dict_tf = {}
    for file in arr:
        path = os.path.join(os.getcwd(), "DocumentsDB", file)
        weights_tf = prepr.structureONLY(path)[0]
        dict_tf[file] = weights_tf
    
    temp_keys = []
    temp_values = []
    
    for key, value in dict_tf.items():
        expanded_arr = expand_arr(value)
        temp_keys.append(key)
        temp_values.append(expanded_arr)
    
    contents = compute_content_matrix(temp_values)
    
    for arr in temp_values:
        for i in range(len(arr)):
            word = arr[i][0]
            n = count_occurrences(contents, word)
            idf = math.log10(nb_of_docs/n)
            arr[i][2] = idf

    for i in range(len(temp_values)):
        dict_idf[temp_keys[i]] = temp_values[i]

    return dict_idf