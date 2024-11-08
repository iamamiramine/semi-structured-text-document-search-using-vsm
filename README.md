# XML Document Processing Pipeline

## Overview
A Python-based pipeline for analyzing XML documents through structural and content-based similarity measures. The system supports multiple document representation schemes, weighting methods, and similarity computation approaches.

## Features

### Document Processing
- XML structure analysis and path extraction
- Content term extraction and vectorization
- Combined structure and content analysis
- Tag-based document representation

### Similarity Measures
- Cosine similarity
- Pearson Correlation Coefficient (PCC)
- Wagner-Fisher similarity for structural comparison
- Support for:
  - Structure-only comparison
  - Content-only comparison
  - Combined structure and content comparison
  - Tag-based similarity

### Weighting Schemes
- Term Frequency (TF)
- Inverse Document Frequency (IDF)
- Combined TF-IDF
- Support for both structural and content weights

### Indexing
- Configurable indexing ratio
- Inverted index creation
- Term importance scoring
- Support for:
  - Structure-based indexing
  - Content-based indexing
  - Combined indexing
  - Tag-based indexing

## Installation

### Requirements
```bash
pip install -r requirements.txt
```
Required packages:
- lxml>=4.9.0
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=0.24.0

### Project 
project_root/
├── scripts/
│ ├── init.py
│ ├── docsProc.py # Document processing
│ ├── docsWeighting.py # Weight calculations
│ ├── indexing.py # Document indexing
│ ├── indexingSimCalc.py # Similarity indexing
│ ├── preprocessing.py # XML preprocessing
│ ├── similarityXML.py # XML similarity
│ └── simQuery.py # Query similarity
├── main.py # Pipeline entry point
└── README.md

## Data Structures

### Document Representations

1. Structure-only Format:

```python
[
[path, frequency], # e.g., ["/root/child", 2]
[tag, frequency] # e.g., ["paragraph", 3]
]
```

2. Content-only Format:

```python
[
[term, frequency] # e.g., ["word", 5]
]
```

3. Structure-Content Format:

```python
[
[path, term, tf, idf] # e.g., ["/root/child", "word", 2, 0.5]
]
```

4. Tag-Content Format:
```
python
[
[term, frequency], # Content terms
[tag, frequency] # XML tags
]
```

### Index Formats

1. Document-Term Index:

```
python
{
"doc1.xml": ["term1", "term2", ...],
"doc2.xml": ["term3", "term4", ...]
}
```

2. Inverted Index:

```
python
{
"term1": ["doc1.xml", "doc3.xml"],
"term2": ["doc2.xml", "doc4.xml"]
}
```


## Usage

### Command Line Interface
```bash
python main.py --input_dir /path/to/dataset --output_dir /path/to/output [options]
```


### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--input_dir` | Input directory containing XML documents | Required |
| `--output_dir` | Output directory for results | Required |
| `--index_ratio` | Indexing ratio (0.0 to 1.0) | 0.5 |
| `--weighting` | Weighting scheme (TF, IDF, TFIDF) | TFIDF |
| `--similarity` | Similarity measure (C=Cosine, P=PCC) | C |

### Example

```bash
python main.py \
--input_dir ./data/xml_docs \
--output_dir ./results \
--index_ratio 0.7 \
--weighting TFIDF \
--similarity C
```

## Pipeline Flow

### 1. Document Loading
- XML file validation
- Directory structure verification
- Document counting

### 2. Preprocessing
- XML parsing
- Structure extraction
- Content extraction
- Term vectorization

### 3. Weight Computation
- Term Frequency (TF) calculation
- Inverse Document Frequency (IDF) calculation
- Combined TF-IDF weights
- Path and tag weighting

### 4. Indexing
- Term selection based on weights
- Index creation
- Optimization
- Multiple representation support

### 5. Similarity Computation
- Structure comparison (Wagner-Fisher)
- Content comparison (Cosine/PCC)
- Combined similarity measures
- Query processing

## Error Handling

### Input Validation
- XML file format checking
- Directory existence verification
- Parameter range validation
- File permission checking

### Processing Errors
- XML parsing errors
- Memory management
- Computation errors
- Index integrity

### Output Handling
- Directory creation
- File writing permissions
- Result validation
- Error logging

## Performance Considerations

### Memory Management
- Streaming document processing
- Efficient data structures
- Memory-efficient indexing
- Optimized matrix operations

### Processing Speed
- Vectorized operations
- Efficient similarity computations
- Optimized path analysis
- Parallel processing where possible

### Scalability
- Configurable indexing ratio
- Memory-efficient processing
- Modular architecture
- Extensible design

## Limitations

### Technical Constraints
- Memory requirements scale with document size
- Processing time increases with XML complexity
- Index size grows with vocabulary size
- Query processing overhead

### Functional Limitations
- Limited to XML documents
- English language focus
- Single-machine processing
- Sequential execution

## Future Enhancements

### Performance Improvements
- Parallel processing support
- Distributed computing capability
- Memory optimization
- Caching mechanisms

### Feature Additions
- Additional similarity measures
- Multi-language support
- Real-time processing
- Advanced querying capabilities

### Integration Options
- Database backend support
- REST API development
- Cloud deployment support
- Monitoring and analytics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

Amir Amine
Jinan Itaoui
Yara Aoun

## Acknowledgments

- XML processing libraries contributors
- Scientific computing package maintainers
- Testing and feedback contributors
