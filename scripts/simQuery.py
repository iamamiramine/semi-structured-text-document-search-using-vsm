"""
Query Similarity Module

This module handles similarity computations between queries and documents:
- Content-based similarity
- Tag-based similarity
- Support for different weighting schemes (TF, IDF, TF-IDF)
- Support for different similarity measures (Cosine, PCC)
"""

from . import preprocessing as prepr
from . import indexingSimCalc as indexDocs
from . import docsWeighting as docswght
import math
import numpy
from typing import List, Dict, Any, Union, Tuple

def contentsOnly(query: str,
                weighting_selector: str,
                similarity_measure: str,
                is_indexed: bool,
                union_or_inter: str,
                dict_cont: Dict[str, List[List[Any]]]) -> List[List[Any]]:
    """
    Compute similarity between query and documents using content only.
    
    Args:
        query: Query string
        weighting_selector: Weight scheme ('TF', 'IDF', 'TFIDF')
        similarity_measure: Similarity measure ('C' for cosine, 'P' for PCC)
        is_indexed: Whether to use indexed terms
        union_or_inter: Union or intersection of terms ('U' or 'I')
        dict_cont: Dictionary of document content
        
    Returns:
        List[List[Any]]: Sorted list of [document_id, similarity_score] pairs
    """
    print(weighting_selector, similarity_measure, is_indexed, union_or_inter)
    similarity_array = []
    query_cont = prepr.queryproc(query)

    if is_indexed:
        if weighting_selector == 'TF':
            all_docs = docswght.TFWeights(dict_cont)
        elif weighting_selector == 'IDF':
            all_docs = docswght.IDFWeights(dict_cont)
        elif weighting_selector == 'TFIDF':
            all_docs = dict_cont
            
        list_indexed_docs = indexDocs.checkIfIndexed(query, 3, union_or_inter)
        documents_cont = {
            doc_id: all_docs[doc_id] for doc_id in list_indexed_docs
        }
    else:
        if weighting_selector == 'TF':
            documents_cont = docswght.TFWeights(dict_cont)
        elif weighting_selector == 'IDF':
            documents_cont = docswght.IDFWeights(dict_cont)
        elif weighting_selector == 'TFIDF':
            documents_cont = dict_cont

    for doc_id, value in documents_cont.items():
        if similarity_measure == 'C':
            similarity_array.append([doc_id, simCosine(query_cont, value)])
        elif similarity_measure == 'P':
            similarity_array.append([doc_id, simPCC(query_cont, value)])

    similarity_array = Sort(similarity_array)
    similarity_array.reverse()
    return similarity_array

def weights(doc_a: List[List[Any]], doc_b: List[List[Any]]) -> List[List[Any]]:
    """
    Compute weight matrix for two documents.
    
    Args:
        doc_a: First document's terms and weights
        doc_b: Second document's terms and weights
        
    Returns:
        List[List[Any]]: Matrix of [term, weight_a, weight_b] for each term
    """
    # Get unique terms from both documents
    total = []
    for a in doc_a:
        total.extend([a[0]])
    for b in doc_b:
        total.extend([b[0]])
    total = numpy.unique(total).tolist()
    
    # Get weights for each term
    weights_a = []
    weights_b = []
    for term in total:
        occ_a = next((a[1] for a in doc_a if a[0] == term), 0)
        weights_a.append(occ_a)
        occ_b = next((b[1] for b in doc_b if b[0] == term), 0)
        weights_b.append(occ_b)
    
    return [[term, w_a, w_b] for term, w_a, w_b in zip(total, weights_a, weights_b)]

def simCosine(doc_a: List[List[Any]], doc_b: List[List[Any]]) -> float:
    """
    Compute cosine similarity between two documents.
    
    Args:
        doc_a: First document's terms and weights
        doc_b: Second document's terms and weights
        
    Returns:
        float: Cosine similarity score
    """
    total = []
    for a in doc_a:
        total.extend([a[0]])
    for b in doc_b:
        total.extend([b[0]])
    total = numpy.unique(total).tolist()
    
    # Calculate numerator (sum of products)
    mat = weights(doc_a, doc_b)
    sum_of_prod = sum(i[1] * i[2] for i in mat)
    
    # Calculate weights for denominator
    weights_a = [next((a[1] for a in doc_a if a[0] == term), 0) for term in total]
    weights_b = [next((b[1] for b in doc_b if b[0] == term), 0) for term in total]
    
    # Calculate denominator
    sum_of_w1_squared = sum(w * w for w in weights_a)
    sum_of_w2_squared = sum(w * w for w in weights_b)
    denominator = math.sqrt(sum_of_w1_squared * sum_of_w2_squared)
    
    if denominator == 0:
        return 0
    return sum_of_prod / denominator

def simPCC(doc_a: List[List[Any]], doc_b: List[List[Any]]) -> float:
    """
    Compute PCC (Pearson Correlation Coefficient) similarity between two documents.
    
    Args:
        doc_a: First document's terms and weights
        doc_b: Second document's terms and weights
        
    Returns:
        float: PCC similarity score
    """
    total = []
    for a in doc_a:
        total.extend([a[0]])
    for b in doc_b:
        total.extend([b[0]])
    total = numpy.unique(total).tolist()
    
    # Get weights and calculate means
    weights_a = [next((a[1] for a in doc_a if a[0] == term), 0) for term in total]
    weights_b = [next((b[1] for b in doc_b if b[0] == term), 0) for term in total]
    
    avg_a = sum(weights_a) / len(weights_a)
    avg_b = sum(weights_b) / len(weights_b)
    
    # Calculate PCC components
    sum_of_prod_pcc = sum((wa - avg_a) * (wb - avg_b) for wa, wb in zip(weights_a, weights_b))
    sum_of_w1_squared_pcc = sum((w - avg_a) ** 2 for w in weights_a)
    sum_of_w2_squared_pcc = sum((w - avg_b) ** 2 for w in weights_b)
    
    denominator = math.sqrt(sum_of_w1_squared_pcc * sum_of_w2_squared_pcc)
    
    if sum_of_prod_pcc == 0 and denominator == 0:
        return 1
    elif denominator == 0:
        return 0
        
    return sum_of_prod_pcc / denominator

def Sort(sub_li: List[List[Any]]) -> List[List[Any]]:
    """
    Sort list by second element.
    
    Args:
        sub_li: List to sort
        
    Returns:
        List[List[Any]]: Sorted list
    """
    return sorted(sub_li, key=lambda x: x[1])

def tagsContentSim(query: str,
                  weighting_selector: str,
                  similarity_measure: str,
                  is_indexed: bool,
                  union_or_inter: str,
                  dict_content_tags: Dict[str, List[List[Any]]]) -> List[List[Any]]:
    """
    Compute similarity between query and documents using tag-based similarity.
    
    Args:
        query: Query string
        weighting_selector: Weight scheme ('TF', 'IDF', 'TFIDF')
        similarity_measure: Similarity measure ('C' for cosine, 'P' for PCC)
        is_indexed: Whether to use indexed terms
        union_or_inter: Union or intersection of terms ('U' or 'I')
        dict_content_tags: Dictionary of document tags and content
        
    Returns:
        List[List[Any]]: Sorted list of [document_id, similarity_score] pairs
    """
    print(weighting_selector, similarity_measure, is_indexed, union_or_inter)
    similarity_array = []
    query_cont = prepr.queryproc(query)

    if is_indexed:
        if weighting_selector == 'TF':
            all_docs = docswght.TFWeights(dict_content_tags)
        elif weighting_selector == 'IDF':
            all_docs = docswght.IDFWeights(dict_content_tags)
        elif weighting_selector == 'TFIDF':
            all_docs = dict_content_tags
            
        list_indexed_docs = indexDocs.checkIfIndexed(query, 4, union_or_inter)
        documents_cont_tags = {
            doc_id: all_docs[doc_id] for doc_id in list_indexed_docs
        }
    else:
        if weighting_selector == 'TF':
            documents_cont_tags = docswght.TFWeights(dict_content_tags)
        elif weighting_selector == 'IDF':
            documents_cont_tags = docswght.IDFWeights(dict_content_tags)
        elif weighting_selector == 'TFIDF':
            documents_cont_tags = dict_content_tags

    for doc_id, value in documents_cont_tags.items():
        if similarity_measure == 'C':
            similarity_array.append([doc_id, simCosine(query_cont, value)])
        elif similarity_measure == 'P':
            similarity_array.append([doc_id, simPCC(query_cont, value)])

    similarity_array = Sort(similarity_array)
    similarity_array.reverse()
    return similarity_array
