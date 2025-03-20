# Performance Trade-offs in Retrieval Augmented Generation Systems for Multi-Hop Question Answering

## Overview
This project implements Retrieval-Augmented Generation (RAG) using different rankers, rerankers, and large language models (LLMs). First stage is designed to retrieve top-10 relevant documents and second stage generates text using a combination of the best ranker and LLMs. The performance is evaluated on multi-hop QA tasks, and results of several combinations are compared and analysed to explore the performance trade-offs in RAG. 

## File Descriptions
Here is a brief overview of the files included in this project:

- **report.pdf**: Detailed report explaining the methodology, experiments, and results of the project.
- **retrieval/**:
    - **rankerA.py, rankerB.py, rankerC.py**: Python scripts implementing different ranker models used in the retrieval stage.
    - **rerankerA.py, rerankerB.py, rerankerC.py**: Python scripts implementing different reranker models to enhance the retrieval results.
- **RAGs/**:
    - **RAGA.py, RAGB.py, RAGC.py, RAGD.py**: Python scripts implementing the RAG models A, B, C, D respectively, using LLMs A, B, C and D. They operate on documents retrieved by the best ranker.
    - **RAGE.py, RAGF.py, RAGG.py, RAGH.py**: Python scripts implementing the RAG models E, F, G, and H, respectively, using LLMs E, F, G, and H. They operate on documents retrieved by the second-best ranker.
- **MyRAGEval.py**: Script used to evaluate the performance of RAG models.
- **MyRetEval.py**: Script used to evaluate the performance of rankers and rerankers in the retrieval stage.
- **requirements.txt**:  List of dependencies required to reproduce the project.
- **output/**: Directory with output files for each ranker, reranker, and RAG model in JSON.
    - **rankerA.json, rankerB.json, rankerC.json**: Results of retrieval using ranker models.
    - **rerankerA.json, rerankerB.json, rerankerC.json**: Results of of retrieval using reranker models.
    - **RAGA.json to RAGH.json**: Results of text generation using RAG models.
- **evaluation_result/**:
    - **ranker_eval.txt**:  Results of rankers evaluation (for reference)
    - **rag_eval.txt**: Results of RAGs evaluation (for reference)


## Installation
1. Git clone this repository

2. Installed the dependencies:  
   `pip install -r requirements.txt`
   
## Running the Models
1. Use `tmux` to restore running progress from a particular terminal. 
2. (Optional) Use `script` to store outputs.
3. First Stage - Run the rankers and rerankers using the same identifiers (A, B, or C) to retrieve documents  
   `python rankerA.py`  
   `python rerankerA.py`

    Do the same for other rankers and rerankers.

    Note: Reranker A corresponds to Ranker A, and so forth for other identifiers.

4. Second Stage - Run the RAG models to generate answer:  
   `python RAGA.py` 

    Do the same for other RAG models. 

5. Evaluation - Evaluate the retreival models:  
   `python MyRetEval.py`
   
   Evaluate the RAG models:  
   `python MyRAGEval.py`

6. Please find the results above in the output/ folder as JSON files. 