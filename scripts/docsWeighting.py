"""
Document Weighting Module

This module provides functions for calculating different weighting schemes (TF and IDF)
for document terms. It handles both structure-only and structure-content combined weights.
"""

import copy
from typing import Dict, List, Any

def IDFWeights(doc_dict: Dict[str, List[List[Any]]]) -> Dict[str, List[List[Any]]]:
    """
    Calculate IDF weights from a document dictionary by removing TF weights.
    
    Args:
        doc_dict: Dictionary with document terms and their weights.
                 Format: {doc_name: [[term, tf, idf], ...]} or 
                        {doc_name: [[term, path, tf, idf], ...]}
    
    Returns:
        Dict[str, List[List[Any]]]: Dictionary with only IDF weights.
                                   Removes element at index 1 for 3-element lists,
                                   or element at index 2 for 4-element lists.
    """
    dict_copy = copy.deepcopy(doc_dict)
    for key, value in dict_copy.items():
        if len(value[0]) == 3:  # Structure or content only format
            for elem in value:
                del elem[1]  # Remove TF weight
        elif len(value[0]) == 4:  # Structure and content combined format
            for elem in value:
                del elem[2]  # Remove TF weight
    return dict_copy

def TFWeights(doc_dict: Dict[str, List[List[Any]]]) -> Dict[str, List[List[Any]]]:
    """
    Calculate TF weights from a document dictionary by removing IDF weights.
    
    Args:
        doc_dict: Dictionary with document terms and their weights.
                 Format: {doc_name: [[term, tf, idf], ...]} or 
                        {doc_name: [[term, path, tf, idf], ...]}
    
    Returns:
        Dict[str, List[List[Any]]]: Dictionary with only TF weights.
                                   Removes element at index 2 for 3-element lists,
                                   or element at index 3 for 4-element lists.
    """
    dict_copy = copy.deepcopy(doc_dict)
    for key, value in dict_copy.items():
        if len(value[0]) == 3:  # Structure or content only format
            for elem in value:
                del elem[2]  # Remove IDF weight
        elif len(value[0]) == 4:  # Structure and content combined format
            for elem in value:
                del elem[3]  # Remove IDF weight
    return dict_copy