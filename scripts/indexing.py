"""
Document Indexing Module

This module handles the indexing of XML documents by:
- Computing TF-IDF scores
- Selecting most important terms
- Creating inverted indices for efficient retrieval
"""

import numpy as np
from typing import Dict, List, Tuple, Any

def tf_idf(doc_dict: Dict[str, List[List[Any]]], selector: int) -> Dict[str, List[List[Any]]]:
    """
    Compute TF-IDF scores for terms in documents.
    
    Args:
        doc_dict: Dictionary containing document terms and weights
        selector: Type of processing (0: content/structure only, 1: structure+content)
    
    Returns:
        Dict[str, List[List[Any]]]: Dictionary with computed TF-IDF scores
    """
    result_dict = {}
    
    if selector == 0:  # Content only or structure only
        for doc_id in doc_dict:
            arr = []
            for term in doc_dict[doc_id]:
                arr.extend([[term[0], term[1]*term[2]]])  # term * tf*idf
            result_dict[doc_id] = arr
            
    elif selector == 1:  # Structure and content combined
        for doc_id in doc_dict:
            arr = []
            for term in doc_dict[doc_id]:
                arr.extend([[term[0]+"-"+term[1], term[2]*term[3]]])  # term-path * tf*idf
            result_dict[doc_id] = arr
            
    return result_dict

def indexing(array: List[List[Any]], ratio: float) -> List[str]:
    """
    Select terms with highest TF-IDF scores.
    
    Args:
        array: List of [term, score] pairs
        ratio: Percentage of terms to keep (0.0 to 1.0)
    
    Returns:
        List[str]: Selected terms with highest scores
    """
    size = len(array)
    indexing_size = int(round(size * ratio))
    selected_terms = []
    count = 0
    
    # Extract scores for sorting
    scores = [term[1] for term in array]
    
    # Select terms with highest scores
    while count < indexing_size:
        max_index = scores.index(max(scores))
        selected_terms.extend([array[max_index][0]])
        scores[max_index] = 0
        count += 1
        
    return selected_terms

def indexing_extended(doc_dict: Dict[str, List[List[Any]]], ratio: float, selector: int) -> Dict[str, List[str]]:
    """
    Create document-term index with highest TF-IDF terms.
    
    Args:
        doc_dict: Dictionary of document terms and weights
        ratio: Percentage of terms to keep
        selector: Type of processing (0: content/structure only, 1: structure+content)
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping documents to their important terms
    """
    tfidf_dict = tf_idf(doc_dict, selector)
    result_dict = {}
    
    for doc_id in tfidf_dict:
        result_dict[doc_id] = indexing(tfidf_dict[doc_id], ratio)
        
    return result_dict

def word_indexing(doc_dict: Dict[str, List[List[Any]]], ratio: float, selector: int) -> Dict[str, List[str]]:
    """
    Create inverted index mapping terms to documents.
    
    Args:
        doc_dict: Dictionary of document terms and weights
        ratio: Percentage of terms to keep
        selector: Type of processing (0: content/structure only, 1: structure+content)
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping terms to list of documents containing them
    """
    doc_term_dict = indexing_extended(doc_dict, ratio, selector)
    
    # Get unique terms across all documents
    all_terms = []
    for doc_id in doc_term_dict:
        all_terms.extend(doc_term_dict[doc_id])
    unique_terms = np.unique(all_terms).tolist()
    
    # Create inverted index
    inverted_index = {}
    for term in unique_terms:
        docs = []
        for doc_id in doc_term_dict:
            if term in doc_term_dict[doc_id]:
                docs.append(doc_id)
        inverted_index[term] = docs
        
    return inverted_index

def run(ratio: float, doc_dict: Tuple[Dict, Dict, Dict, Dict]) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Run complete indexing pipeline for all document representations.
    
    Args:
        ratio: Percentage of terms to keep in index
        doc_dict: Tuple of dictionaries (structure, structure+content, content, content+tags)
    
    Returns:
        Tuple[Dict, Dict, Dict, Dict]: Inverted indices for each document representation
    """
    docs_struct, docs_struct_cont, docs_cont, docs_cont_tags = doc_dict
    
    return (
        word_indexing(docs_struct, ratio, 0),
        word_indexing(docs_struct_cont, ratio, 1),
        word_indexing(docs_cont, ratio, 0),
        word_indexing(docs_cont_tags, ratio, 0)
    )
