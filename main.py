#!/usr/bin/env python3
"""
XML Document Processing and Similarity Pipeline

This script runs the XML document processing pipeline, which includes:
- Document preprocessing
- TF-IDF calculation
- Indexing
- Similarity calculations

Usage:
    python main.py --input_dir /path/to/dataset --output_dir /path/to/output [options]
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add scripts directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts import (
    docsProc,
    indexing,
    similarityXML,
    simQuery,
    indexingSimCalc,
    preprocessing
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='XML Document Processing Pipeline')
    parser.add_argument('--input_dir', required=True, 
                       help='Input directory containing XML documents')
    parser.add_argument('--output_dir', required=True, 
                       help='Output directory for results')
    parser.add_argument('--index_ratio', type=float, default=0.5,
                       help='Indexing ratio (default: 0.5)')
    parser.add_argument('--weighting', choices=['TF', 'IDF', 'TFIDF'],
                       default='TFIDF', help='Weighting scheme to use')
    parser.add_argument('--similarity', choices=['C', 'P'],
                       default='C', help='Similarity measure (C=Cosine, P=PCC)')
    return parser.parse_args()

def setup_directories(args: argparse.Namespace) -> None:
    """
    Create output directory if it doesn't exist and validate input directory.
    
    Args:
        args: Command line arguments
        
    Raises:
        FileNotFoundError: If input directory doesn't exist
        NotADirectoryError: If input path is not a directory
    """
    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

def validate_documents(input_dir: str) -> None:
    """
    Validate XML documents in input directory.
    
    Args:
        input_dir: Path to input directory
        
    Raises:
        ValueError: If no XML files found or invalid XML detected
    """
    xml_files = list(Path(input_dir).glob('*.xml'))
    if not xml_files:
        raise ValueError(f"No XML files found in {input_dir}")
    
    # Validate each XML file
    for xml_file in xml_files:
        try:
            preprocessing.preproc(str(xml_file))
        except Exception as e:
            raise ValueError(f"Invalid XML file {xml_file}: {str(e)}")

def process_documents(args: argparse.Namespace) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Process documents and compute measures.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple[Dict, Dict, Dict, Dict]: Document measures for different representations
    """
    logger.info("Processing documents...")
    try:
        return docsProc.run()
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise

def compute_indexing(args: argparse.Namespace, docs_data: Tuple[Dict, Dict, Dict, Dict]) -> Dict:
    """
    Compute document indexing.
    
    Args:
        args: Command line arguments
        docs_data: Document measures
        
    Returns:
        Dict: Indexing results
    """
    logger.info("Computing document indexing...")
    try:
        # Initialize indexing
        indexingSimCalc.start(args.index_ratio, docs_data)
        return indexing.run(args.index_ratio, docs_data)
    except Exception as e:
        logger.error(f"Error computing indexing: {str(e)}")
        raise

def save_results(results: Dict, output_path: str) -> None:
    """
    Save processing results to file.
    
    Args:
        results: Results to save
        output_path: Path to output file
    """
    logger.info(f"Saving results to {output_path}")
    try:
        with open(output_path, 'w') as f:
            f.write(str(results))
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main() -> None:
    """Main pipeline execution function."""
    try:
        # Parse arguments and setup
        args = parse_args()
        setup_directories(args)
        
        # Change to input directory
        original_dir = os.getcwd()
        os.chdir(args.input_dir)
        
        try:
            # Validate input documents
            validate_documents(args.input_dir)
            
            # Process documents
            docs_data = process_documents(args)
            
            # Compute indexing
            indexing_results = compute_indexing(args, docs_data)
            
            # Save results
            output_path = os.path.join(args.output_dir, 'results.txt')
            save_results(indexing_results, output_path)
            
            logger.info("Pipeline completed successfully")
            
        finally:
            # Restore original directory
            os.chdir(original_dir)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 