"""
XML Document Similarity Module

This module computes similarity between XML documents using various measures:
- Structure similarity
- Content similarity
- Combined structure and content similarity
Supports different weighting schemes (TF, IDF, TF-IDF) and similarity measures
(Cosine, PCC).
"""

from . import preprocessing as prepr
from . import docsWeighting as docswght
from . import indexingSimCalc as indexDocs
import math
import copy
import numpy
from typing import List, Dict, Any, Tuple, Union

def structureContentSim(doc: str, 
                       weighting_selector: str,
                       similarity_measure: str,
                       is_indexed: bool,
                       union_or_inter: str,
                       dict_struct_cont: Dict) -> List[List[Any]]:
    """
    Compute similarity between documents using both structure and content.
    
    Args:
        doc: Query document path
        weighting_selector: Weight scheme ('TF', 'IDF', 'TFIDF')
        similarity_measure: Similarity measure ('C' for cosine, 'P' for PCC)
        is_indexed: Whether to use indexed terms
        union_or_inter: Union or intersection of terms ('U' or 'I')
        dict_struct_cont: Dictionary of document structure and content
        
    Returns:
        List[List[Any]]: Sorted list of [document_id, similarity_score] pairs
    """
    print(doc, weighting_selector, similarity_measure, is_indexed, union_or_inter)
    similarity_array = []
    query_struct_cont = prepr.struc_AND_content(doc)

    if is_indexed:
        if weighting_selector == 'TF':
            all_docs = docswght.TFWeights(dict_struct_cont)
        elif weighting_selector == 'IDF':
            all_docs = docswght.IDFWeights(dict_struct_cont)
        elif weighting_selector == 'TFIDF':
            all_docs = dict_struct_cont
            
        list_indexed_docs = indexDocs.checkIfIndexed(doc, 2, union_or_inter)
        documents_struct_cont = {
            doc_id: all_docs[doc_id] for doc_id in list_indexed_docs
        }
    else:
        if weighting_selector == 'TF':
            documents_struct_cont = docswght.TFWeights(dict_struct_cont)
        elif weighting_selector == 'IDF':
            documents_struct_cont = docswght.IDFWeights(dict_struct_cont)
        elif weighting_selector == 'TFIDF':
            documents_struct_cont = dict_struct_cont

    for doc_id, value in documents_struct_cont.items():
        extended_vec = getExtendedV(query_struct_cont, value)
        query_vect = extendV(query_struct_cont, extended_vec)
        doc_vect = extendV(value, extended_vec)

        if similarity_measure == 'C':
            similarity_array.append([doc_id, simCosineStrucCont(query_vect, doc_vect)])
        elif similarity_measure == 'P':
            similarity_array.append([doc_id, simPCCStrucCont(query_vect, doc_vect)])

    similarity_array = Sort(similarity_array)
    similarity_array.reverse()
    return similarity_array

def structureSim(doc: str,
                 weighting_selector: str,
                 similarity_measure: str,
                 is_indexed: bool,
                 union_or_inter: str,
                 dict_struct: Dict,
                 sel_to_compare: int) -> List[List[Any]]:
    """
    Compute similarity between documents using structure only.
    
    Args:
        doc: Query document path
        weighting_selector: Weight scheme ('TF', 'IDF', 'TFIDF')
        similarity_measure: Similarity measure ('C' for cosine, 'P' for PCC)
        is_indexed: Whether to use indexed terms
        union_or_inter: Union or intersection of terms ('U' or 'I')
        dict_struct: Dictionary of document structure
        sel_to_compare: Selector for comparison method (1 or other)
        
    Returns:
        List[List[Any]]: Sorted list of [document_id, similarity_score] pairs
    """
    print(doc, weighting_selector, similarity_measure, is_indexed, union_or_inter)
    similarity_array = []

    # Get query structure based on selector
    if sel_to_compare == 1:
        query_struct = prepr.structureONLY(doc)[0]
    else:
        query_struct = prepr.all_paths_weights(doc)[0]

    if is_indexed:
        if weighting_selector == 'TF':
            all_docs = docswght.TFWeights(dict_struct)
        elif weighting_selector == 'IDF':
            all_docs = docswght.IDFWeights(dict_struct)
        elif weighting_selector == 'TFIDF':
            all_docs = dict_struct
            
        list_indexed_docs = indexDocs.checkIfIndexed(doc, 1, union_or_inter)
        documents_struct = {
            doc_id: all_docs[doc_id] for doc_id in list_indexed_docs
        }
    else:
        if weighting_selector == 'TF':
            documents_struct = docswght.TFWeights(dict_struct)
        elif weighting_selector == 'IDF':
            documents_struct = docswght.IDFWeights(dict_struct)
        elif weighting_selector == 'TFIDF':
            documents_struct = dict_struct

    for doc_id, value in documents_struct.items():
        if similarity_measure == 'C':
            similarity_array.append([doc_id, simCosineStruc(query_struct, value)])
        elif similarity_measure == 'P':
            similarity_array.append([doc_id, simPCCStruc(query_struct, value)])

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
    total = []
    for a in doc_a:
        total.extend([a[0]])
    for b in doc_b:
        total.extend([b[0]])
    total = numpy.unique(total).tolist()
    
    weights_a = []
    weights_b = []
    
    for term in total:
        occ_a = next((a[1] for a in doc_a if a[0] == term), 0)
        weights_a.append(occ_a)
        occ_b = next((b[1] for b in doc_b if b[0] == term), 0)
        weights_b.append(occ_b)
    
    return [[term, w_a, w_b] for term, w_a, w_b in zip(total, weights_a, weights_b)]

def simCosineStruc(doc_a: List[List[Any]], doc_b: List[List[Any]]) -> float:
    """
    Compute cosine similarity between two documents using structure.
    
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
    
    sum_of_prod = 0
    mat = weights(doc_a, doc_b)
    for i in mat:
        sum_of_prod += (i[1] * i[2])
        
    weights_a = [next((a[1] for a in doc_a if a[0] == term), 0) for term in total]
    weights_b = [next((b[1] for b in doc_b if b[0] == term), 0) for term in total]
    
    sum_of_w1_squared = sum(w * w for w in weights_a)
    sum_of_w2_squared = sum(w * w for w in weights_b)
    
    denominator = math.sqrt(sum_of_w1_squared * sum_of_w2_squared)
    if denominator == 0:
        return 0
        
    return sum_of_prod / denominator

def simPCCStruc(doc_a: List[List[Any]], doc_b: List[List[Any]]) -> float:
    """
    Compute PCC similarity between two documents using structure.
    
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
    
    weights_a = [next((a[1] for a in doc_a if a[0] == term), 0) for term in total]
    weights_b = [next((b[1] for b in doc_b if b[0] == term), 0) for term in total]
    
    avg_a = sum(weights_a) / len(weights_a)
    avg_b = sum(weights_b) / len(weights_b)
    
    sum_of_prod_pcc = sum((wa - avg_a) * (wb - avg_b) for wa, wb in zip(weights_a, weights_b))
    sum_of_w1_squared_pcc = sum((w - avg_a) ** 2 for w in weights_a)
    sum_of_w2_squared_pcc = sum((w - avg_b) ** 2 for w in weights_b)
    
    denominator = math.sqrt(sum_of_w1_squared_pcc * sum_of_w2_squared_pcc)
    
    if sum_of_prod_pcc == 0 and denominator == 0:
        return 1
    elif denominator == 0:
        return 0
        
    return sum_of_prod_pcc / denominator

def simCosineStrucCont(vec1: List[List[Any]], vec2: List[List[Any]]) -> float:
    """
    Compute cosine similarity between two documents using structure and content.
    
    Args:
        vec1: First document's vector
        vec2: Second document's vector
        
    Returns:
        float: Cosine similarity score
    """
    if len(vec1[0]) == 3:
        array_vd1 = toArray(vec1)
        array_vd2 = toArray(vec2)
        numerator_sim = 0

        for i in range(len(vec1)):
            item = vec1[i]
            name = array_vd1[i]
            lst = list_duplicates_of(array_vd1, name)
            for j in range(len(lst)):
                sim = WagnerFisher(item[0], vec2[lst[j]][0])
                numerator_sim += item[2] * vec2[lst[j]][2] * sim

        denominator_sim = math.sqrt(getCard(vec1) * getCard(vec2))
        if denominator_sim == 0:
            return 0
            
        return numerator_sim / denominator_sim
    return 0

def simPCCStrucCont(vec1: List[List[Any]], vec2: List[List[Any]]) -> float:
    """
    Compute PCC similarity between two documents using structure and content.
    
    Args:
        vec1: First document's vector
        vec2: Second document's vector
        
    Returns:
        float: PCC similarity score
    """
    if len(vec1[0]) == 3:
        array_vd1 = toArray(vec1)
        array_vd2 = toArray(vec2)
        numerator_sim = 0
        mean_vd1 = getMean(vec1)
        mean_vd2 = getMean(vec2)

        for i in range(len(vec1)):
            item = vec1[i]
            name = array_vd1[i]
            lst = list_duplicates_of(array_vd1, name)
            for j in range(len(lst)):
                sim = WagnerFisher(item[0], vec2[lst[j]][0])
                numerator_sim += (item[2] - mean_vd1) * (vec2[lst[j]][2] - mean_vd2) * sim

        denominator1 = sum((elem[2] - mean_vd1) ** 2 for elem in vec1)
        denominator2 = sum((elem[2] - mean_vd2) ** 2 for elem in vec2)
        denominator_sim = math.sqrt(denominator1 * denominator2)

        if numerator_sim == 0 and denominator_sim == 0:
            return 1
        elif denominator_sim == 0:
            return 0
            
        return numerator_sim / denominator_sim
    return 0

def getMean(vector: List[List[Any]]) -> float:
    """
    Calculate mean of weights in vector.
    
    Args:
        vector: Vector containing weights
        
    Returns:
        float: Mean weight
    """
    weights = [elem[2] for elem in vector]
    return sum(weights) / len(weights)

def getCard(vector: List[List[Any]]) -> float:
    """
    Calculate cardinality (sum of squared weights) of vector.
    
    Args:
        vector: Vector containing weights
        
    Returns:
        float: Cardinality
    """
    return sum(elem[2] ** 2 for elem in vector)

def list_duplicates_of(seq: List[Any], item: Any) -> List[int]:
    """
    Find all indices of duplicates in a sequence.
    
    Args:
        seq: Sequence to search
        item: Item to find
        
    Returns:
        List[int]: List of indices where item appears
    """
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def toArray(vector: List[List[Any]]) -> List[Any]:
    """
    Extract second elements from vector.
    
    Args:
        vector: Input vector
        
    Returns:
        List[Any]: List of second elements
    """
    return [elem[1] for elem in vector]

def WagnerFisher(lst1: str, lst2: str) -> float:
    """
    Compute Wagner-Fisher similarity between two strings.
    
    Args:
        lst1: First string
        lst2: Second string
        
    Returns:
        float: Similarity score
    """
    lst1 = lst1[1:]
    lst2 = lst2[1:]
    string_a = lst1.split('/')
    string_b = lst2.split('/')
    m = len(string_a)
    n = len(string_b)

    dist = [[0 for x in range(n + 1)] for y in range(m + 1)]

    for i in range(1, m + 1):
        dist[i][0] = dist[i-1][0] + len(string_a[i-1])
    for j in range(1, n + 1):
        dist[0][j] = dist[0][j-1] + len(string_b[j-1])
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dist[i][j] = min(
                dist[i-1][j-1] + costUpdate(string_a[i-1], string_b[j-1]),
                dist[i-1][j] + len(string_a[i-1]),
                dist[i][j-1] + len(string_b[j-1])
            )

    return 1 / (1 + dist[m][n])

def costUpdate(word_a: str, word_b: str) -> int:
    """
    Compute update cost between two words.
    
    Args:
        word_a: First word
        word_b: Second word
        
    Returns:
        int: Update cost
    """
    if word_a == word_b:
        return 0
    elif len(word_a) == len(word_b):
        return 1
    else:
        return abs(len(word_a) - len(word_b))

def getExtendedV(vector1: List[List[Any]], vector2: List[List[Any]]) -> List[List[Any]]:
    """
    Create extended vector containing all terms from both vectors.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        List[List[Any]]: Extended vector
    """
    extended_vector = []
    for elem in vector1:
        item = [elem[0], elem[1]]
        extended_vector.append(item)
    for elem in vector2:
        item = [elem[0], elem[1]]
        if not checkIn(item, extended_vector):
            extended_vector.append(item)
    return extended_vector

def extendV(vector: List[List[Any]], ex_v: List[List[Any]]) -> List[List[Any]]:
    """
    Extend vector to match dimensions of extended vector.
    
    Args:
        vector: Vector to extend
        ex_v: Extended vector to match
        
    Returns:
        List[List[Any]]: Extended vector with weights
    """
    to_return = [[item[0], item[1], 0] for item in ex_v]

    for elem in vector:
        try:
            ind = getIndex(elem, to_return)
            to_return[ind][2] = elem[2]
        except:
            print(vector)

    return to_return

def checkIn(elem: List[Any], vector: List[List[Any]]) -> bool:
    """
    Check if element exists in vector.
    
    Args:
        elem: Element to check
        vector: Vector to search in
        
    Returns:
        bool: True if element exists in vector
    """
    return elem in vector

def getIndex(elem: List[Any], vector: List[List[Any]]) -> int:
    """
    Get index of element in vector.
    
    Args:
        elem: Element to find
        vector: Vector to search in
        
    Returns:
        int: Index of element
    """
    item = copy.deepcopy(elem)
    try:
        item[2] = 0
    except:
        print(elem)

    if len(item) == 4:
        item.pop()
    return vector.index(item)

def Sort(sub_li: List[List[Any]]) -> List[List[Any]]:
    """
    Sort list by second element.
    
    Args:
        sub_li: List to sort
        
    Returns:
        List[List[Any]]: Sorted list
    """
    if len(sub_li) == 0:
        return sub_li
    return sorted(sub_li, key=lambda x: x[1])