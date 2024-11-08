"""
Indexing and Similarity Calculation Module

This module handles document indexing and similarity calculations between documents,
supporting different types of document representations:
- Structure only
- Structure and content combined
- Content only
- Content with tags
"""

from . import indexing
from . import preprocessing as prepr
import copy
from typing import List, Dict, Any, Optional

# Global index dictionaries
indexed_struct: Dict = {}
indexed_struct_cont: Dict = {}
indexed_cont: Dict = {}
indexed_cont_tags: Dict = {}

def start(percent_index: float, doc_dict: Dict) -> None:
    """
    Initialize indexing with given percentage and document dictionary.
    
    Args:
        percent_index: Percentage of terms to keep in index (0.0 to 1.0)
        doc_dict: Dictionary containing document terms and weights
    """
    index_results = indexing.run(percent_index, doc_dict)
    global indexed_struct, indexed_struct_cont, indexed_cont, indexed_cont_tags
    indexed_struct = index_results[0]
    indexed_struct_cont = index_results[1]
    indexed_cont = index_results[2]
    indexed_cont_tags = index_results[3]

def check_if_indexed(doc: str, selector: int, union_or_inter: str) -> List[str]:
    """
    Check if document terms are in index and return matching documents.
    
    Args:
        doc: Document path or query string
        selector: Type of processing:
                 1 -> structure only
                 2 -> structure and content
                 3 -> query as sentence without tags
                 4 -> query as sentence with tags
        union_or_inter: Union ('U') or intersection ('I') of results
    
    Returns:
        List[str]: List of matching document names
    """
    query_vec: List = []
    documents_list: List[List[str]] = []
    tags_vec: List[str] = []
    indexed_terms: List[str] = []

    # Process query based on selector
    if selector == 1:
        query_vec = prepr.all_paths_weights(doc)[0]
        tags_vec = get_tags(query_vec, selector)
        indexed_terms = indexed_struct.keys()
        print(indexed_terms)
        for elem in tags_vec:
            if elem in indexed_terms:
                documents_list.append(indexed_struct[elem])

    elif selector == 2:
        query_vec = prepr.struc_AND_content(doc)
        tags_vec = get_tags(query_vec, selector)
        indexed_terms = indexed_struct_cont.keys()
        print(indexed_terms)
        for elem in tags_vec:
            if elem in indexed_terms:
                documents_list.append(indexed_struct_cont[elem])

    elif selector == 3:
        query_vec = prepr.queryproc(doc)
        tags_vec = get_tags(query_vec, selector)
        indexed_terms = indexed_cont.keys()
        print(indexed_terms)
        for elem in tags_vec:
            if elem in indexed_terms:
                documents_list.append(indexed_cont[elem])

    elif selector == 4:
        query_vec = prepr.queryproc(doc)
        tags_vec = get_tags(query_vec, selector)
        indexed_terms = indexed_cont_tags.keys()
        print(indexed_terms)
        for elem in tags_vec:
            if elem in indexed_terms:
                documents_list.append(indexed_cont[elem])

    if len(documents_list) == 0:
        return []

    # Combine results using union or intersection
    if union_or_inter == 'U':
        return get_union_docs(documents_list)
    elif union_or_inter == 'I':
        return get_intersection_docs(documents_list)
    
    return []

def get_tags(vect: List[List[Any]], selector: int) -> List[str]:
    """
    Extract tags from vector based on selector type.
    
    Args:
        vect: Vector containing terms and weights
        selector: Type of processing (1-4)
    
    Returns:
        List[str]: List of extracted tags
    """
    to_return = []
    if selector in [1, 3, 4]:
        for elem in vect:
            to_return.append(elem[0])
    elif selector == 2:
        for elem in vect:
            temp = elem[0] + '-' + elem[1]
            to_return.append(temp)
    return to_return

def get_union_docs(docs_list: List[List[str]]) -> List[str]:
    """
    Get union of document lists.
    
    Args:
        docs_list: List of document lists to union
    
    Returns:
        List[str]: Union of all document lists
    """
    to_return = []
    for elem in docs_list:
        for item in elem:
            if item not in to_return:
                to_return.append(item)
    return to_return

def get_intersection_docs(docs_list: List[List[str]]) -> List[str]:
    """
    Get intersection of document lists.
    
    Args:
        docs_list: List of document lists to intersect
    
    Returns:
        List[str]: Intersection of all document lists
    """
    if not docs_list:
        return []
        
    to_return = copy.deepcopy(docs_list[0])
    for i in range(1, len(docs_list)):
        to_return = [doc for doc in to_return if doc in docs_list[i]]
    return to_return
