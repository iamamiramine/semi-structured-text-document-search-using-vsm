"""
XML Document Preprocessing Module

This module handles the preprocessing of XML documents, including:
- XML parsing and structure analysis
- Term extraction and vectorization
- Content and path processing
- Query preprocessing
"""

from lxml import etree
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, List, Tuple, Any, Union

def preproc(doc: str) -> pd.DataFrame:
    """
    Preprocess an XML document to extract terms and structure.
    
    Args:
        doc: Path to XML document
        
    Returns:
        pd.DataFrame: DataFrame containing terms and their paths with term frequencies
    """
    tree = etree.parse(doc)
    root = tree.getroot()
    text_list = []
    cv = CountVectorizer(stop_words='english')
    
    # Process root node
    for element in tree.iter():
        if element == root:
            text_list.append(root.text)
            df = pd.DataFrame({'tag': [root.tag], 'text': text_list})

            # Process root attributes
            for key in root.attrib:
                df_attr = pd.DataFrame({
                    'tag': [f"{root.tag}/{key}"], 
                    'text': root.attrib.get(key)
                })
                df = pd.concat([df, df_attr])
        else:
            # Process other nodes
            inner_list = []
            inner_list.append(element.text)
            df2 = pd.DataFrame({
                'tag': [tree.getpath(element)], 
                'text': inner_list
            })

            # Process node attributes
            for key in element.attrib:
                df_attr = pd.DataFrame({
                    'tag': [f"{tree.getpath(element)}/{key}"], 
                    'text': element.attrib.get(key)
                })
                df2 = pd.concat([df2, df_attr])

            df = pd.concat([df, df2])

    # Create term frequency matrix
    cv_matrix = cv.fit_transform(df['text'])
    df_tf = pd.DataFrame(
        cv_matrix.toarray(), 
        index=df['tag'].values, 
        columns=cv.get_feature_names_out()
    )

    return df_tf

def struc_AND_content(doc: str) -> List[List[Any]]:
    """
    Extract structure and content information from document.
    
    Args:
        doc: Path to XML document
        
    Returns:
        List[List[Any]]: List of [path, term, frequency] triplets
    """
    df = preproc(doc)
    df = df.fillna(0).astype(int)
    doc_dict = df.to_dict('index')
    
    # Convert dictionary to list format
    term_list = []
    for path, terms in doc_dict.items():
        for term, freq in terms.items():
            if freq != 0:
                term_list.append([path, term, freq])
                
    return term_list

def contentONLY(doc: str) -> List[List[Any]]:
    """
    Extract content-only information from document.
    
    Args:
        doc: Path to XML document
        
    Returns:
        List[List[Any]]: List of [term, frequency] pairs
    """
    tf_array = struc_AND_content(doc)
    term_dict = {}
    
    # Aggregate term frequencies across all paths
    for item in tf_array:
        term = item[1]
        freq = item[2]
        term_dict[term] = term_dict.get(term, 0) + freq
    
    return [[term, freq] for term, freq in term_dict.items()]

def structureONLY(doc: str) -> Tuple[List[List[Any]], List[List[Any]]]:
    """
    Extract structural information from document.
    
    Args:
        doc: Path to XML document
        
    Returns:
        Tuple[List[List[Any]], List[List[Any]]]: 
            - List of [path, frequency] pairs
            - List of [tag, frequency] pairs
    """
    tree = etree.parse(doc)
    root = tree.getroot()
    
    # Extract paths and tags
    paths_arr = []
    tags = []
    for element in tree.iter():
        tags.append(element.tag.casefold())
        
        # Build path from root
        curr_elem = element
        path = [curr_elem]
        while curr_elem.getparent() is not None:
            curr_elem = curr_elem.getparent()
            path.append(curr_elem)
        paths_arr.append(list(reversed(path)))
    
    # Process paths
    paths = []
    for path_elements in paths_arr:
        if len(path_elements) > 1:
            path = "/".join(elem.tag for elem in path_elements)
            paths.append(path.casefold())
    
    # Count frequencies
    path_weights = []
    for path in paths:
        if path != "0":
            count = paths.count(path)
            path_weights.append([path, count])
            paths = ["0" if p == path else p for p in paths]
    
    tag_weights = []
    for tag in tags:
        if tag != "0":
            count = tags.count(tag)
            tag_weights.append([tag, count])
            tags = ["0" if t == tag else t for t in tags]
    
    return path_weights, tag_weights

def preprocq(query: str) -> pd.DataFrame:
    """
    Preprocess a query string.
    
    Args:
        query: Query string to process
        
    Returns:
        pd.DataFrame: DataFrame containing query terms and frequencies
    """
    cv = CountVectorizer(stop_words='english')
    df = pd.DataFrame({'tag': "query", 'text': [query]})
    cv_matrix = cv.fit_transform(df['text'])
    return pd.DataFrame(
        cv_matrix.toarray(), 
        index=df['tag'].values, 
        columns=cv.get_feature_names_out()
    )

def toArr(df: pd.DataFrame) -> List[List[Any]]:
    """
    Convert DataFrame to array format.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        List[List[Any]]: List of [path, term, frequency] triplets
    """
    df = df.fillna(0).astype(int)
    doc_dict = df.to_dict('index')
    
    term_list = []
    for path, terms in doc_dict.items():
        for term, freq in terms.items():
            if freq != 0:
                term_list.append([path, term, freq])
                
    return term_list

def toQuery(tf_array: List[List[Any]]) -> List[List[Any]]:
    """
    Convert TF array to query format.
    
    Args:
        tf_array: List of term frequencies
        
    Returns:
        List[List[Any]]: List of [term, frequency] pairs
    """
    term_dict = {}
    for item in tf_array:
        term = item[1]
        freq = item[2]
        term_dict[term] = term_dict.get(term, 0) + freq
    
    return [[term, freq] for term, freq in term_dict.items()]

def queryproc(query: str) -> List[List[Any]]:
    """
    Process a query string into term frequencies.
    
    Args:
        query: Query string to process
        
    Returns:
        List[List[Any]]: List of [term, frequency] pairs
    """
    df = preprocq(query)
    arr = toArr(df)
    return toQuery(arr)

def tagsANDContent(doc: str) -> List[List[Any]]:
    """
    Combine content and tag information.
    
    Args:
        doc: Path to XML document
        
    Returns:
        List[List[Any]]: Combined list of content and tag information
    """
    return contentONLY(doc) + all_paths_weights(doc)[1]

def all_paths_weights(doc: str) -> Tuple[List[List[Any]], List[List[Any]]]:
    """
    Extract all path weights and tag weights from document.
    
    This function combines path analysis and tag frequency counting,
    including attribute paths in the XML structure.
    
    Args:
        doc: Path to XML document
        
    Returns:
        Tuple[List[List[Any]], List[List[Any]]]: 
            - List of [path, frequency] pairs including attributes
            - List of [tag, frequency] pairs
    """
    tree = etree.parse(doc)
    root = tree.getroot()
    
    # Extract paths and tags
    paths_arr = []
    tags = []
    
    for element in tree.iter():
        tags.append(element.tag.casefold())
        
        # Handle attributes
        for attb in element.attrib:
            tags.append(attb)
        
        # Build path from root
        curr_elem = element
        path = [curr_elem]
        while curr_elem.getparent() is not None:
            curr_elem = curr_elem.getparent()
            path.append(curr_elem)
        paths_arr.append(list(reversed(path)))
    
    # Process paths including attributes
    paths = []
    for path_elements in paths_arr:
        if len(path_elements) > 1:
            path = "/".join(elem.tag for elem in path_elements)
            paths.append(path.casefold())
            
            # Add attribute paths
            for elem in path_elements:
                for attb in elem.attrib:
                    attr_path = f"{path}/{attb}"
                    paths.append(attr_path.casefold())
    
    # Count frequencies
    path_weights = []
    for path in paths:
        if path != "0":
            count = paths.count(path)
            path_weights.append([path, count])
            paths = ["0" if p == path else p for p in paths]
    
    tag_weights = []
    for tag in tags:
        if tag != "0":
            count = tags.count(tag)
            tag_weights.append([tag, count])
            tags = ["0" if t == tag else t for t in tags]
    
    return path_weights, tag_weights
