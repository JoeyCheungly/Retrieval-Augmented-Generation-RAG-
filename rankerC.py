import json, os
from tqdm import tqdm 
from copy import deepcopy
from typing import Any, Generator, List, Dict, Optional

import openai

from llama_index import (
  ServiceContext,
  OpenAIEmbedding,
  PromptHelper,
  VectorStoreIndex,
  set_global_service_context
)
from llama_index.extractors import BaseExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.llms import OpenAI
from llama_index.text_splitter import SentenceSplitter
from llama_index.embeddings import HuggingFaceEmbedding,VoyageEmbedding,InstructorEmbedding
from llama_index.postprocessor import FlagEmbeddingReranker
from llama_index.schema import QueryBundle,MetadataMode
from llama_index.schema import Document

# This is the staging flag. Set to False if you want to run on the real
# collection.
STAGING=False
#STAGING=True

def save_list_to_json(lst, filename):
  """ Save Files """
  with open(filename, 'w') as file:
    json.dump(lst, file)

def wr_dict(filename,dic):
  """ Write Files """
  try:
    if not os.path.isfile(filename):
      data = []
      data.append(dic)
      with open(filename, 'w') as f:
        json.dump(data, f)
    else:      
      with open(filename, 'r') as f:
        data = json.load(f)
        data.append(dic)
      with open(filename, 'w') as f:
        json.dump(data, f)
  except Exception as e:
    print("Save Error:", str(e))
  return
            
def rm_file(file_path):
  """ Delete Files """
  if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File {file_path} removed successfully.")

def _depth_first_yield(json_data: Any, levels_back: int, collapse_length: 
                       Optional[int], path: List[str], ensure_ascii: bool = False,
                      ) -> Generator[str, None, None]:
  """ Do depth first yield of all of the leaf nodes of a JSON.
      Combines keys in the JSON tree using spaces.
      If levels_back is set to 0, prints all levels.
      If collapse_length is not None and the json_data is <= that number
      of characters, then we collapse it into one line.
  """
  if isinstance(json_data, (dict, list)):
    # only try to collapse if we're not at a leaf node
    json_str = json.dumps(json_data, ensure_ascii=ensure_ascii)
    if collapse_length is not None and len(json_str) <= collapse_length:
      new_path = path[-levels_back:]
      new_path.append(json_str)
      yield " ".join(new_path)
      return
    elif isinstance(json_data, dict):
      for key, value in json_data.items():
        new_path = path[:]
        new_path.append(key)
        yield from _depth_first_yield(value, levels_back, collapse_length, new_path)
    elif isinstance(json_data, list):
      for _, value in enumerate(json_data):
        yield from _depth_first_yield(value, levels_back, collapse_length, path)
    else:
      new_path = path[-levels_back:]
      new_path.append(str(json_data))
      yield " ".join(new_path)


# The two classes are used to parse the json corpus and queries.
class JSONReader():
  """JSON reader.
     Reads JSON documents with options to help suss out relationships between nodes.
  """
  def __init__(self, is_jsonl: Optional[bool] = False,) -> None:
    """Initialize with arguments."""
    super().__init__()
    self.is_jsonl = is_jsonl

  def load_data(self, input_file: str) -> List[Document]:
    """Load data from the input file."""
    documents = []
    with open(input_file, 'r') as file:
      load_data = json.load(file)
    for data in load_data:
      metadata = {"title": data['title'], 
                  "published_at": data['published_at'],
                  "source":data['source']}
      documents.append(Document(text=data['body'], metadata=metadata))
    return documents
    

class CustomExtractor(BaseExtractor):
  async def aextract(self, nodes) -> List[Dict]:
    metadata_list = [
      {
        "title": (node.metadata["title"]),
        "source": (node.metadata["source"]),      
        "published_at": (node.metadata["published_at"])
      } for node in nodes
    ]
    return metadata_list


def gen_stage_0(corpus, queries, rank_model_name, rerank, 
                rerank_model_name, output_name):
  openai.api_key = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")
  openai.base_url = "your_api_base"
  voyage_api_key = os.environ.get("VOYAGE_API_KEY", "your_voyage_api_key")
  cohere_api_key = os.environ.get("COHERE_API_KEY", "your_cohere_api_key")
  model_name = rank_model_name 
  def_llm = "gpt-3.5-turbo-1106"
  topk = 10
  chunk_size = 256
  context_window = 2048
  num_output = 256
  save_file = output_name
  model_name = rank_model_name 
  llm = OpenAI(model=def_llm, temperature=0, max_tokens=context_window)

  print('Remove save file if exists.')
  rm_file(save_file)

  # Most likely you only need the HuggingFaceEmbedding, but I try to 
  # account # for many other possibilities.
  if 'text' in model_name:
    # "text-embedding-ada-002" “text-search-ada-query-001”
    embed_model = OpenAIEmbedding(model = model_name,embed_batch_size=10)
  elif 'Cohere' in model_name:
    embed_model = CohereEmbedding(
      cohere_api_key=cohere_api_key,
      model_name="embed-english-v3.0",
      input_type="search_query",
    )
  elif 'voyage-02' in model_name:
    embed_model = VoyageEmbedding(
      model_name='voyage-02', voyage_api_key=voyage_api_key
    )
  elif 'instructor' in model_name:
    embed_model = InstructorEmbedding(model_name=model_name)
  else:
    embed_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)

  # Create a service context 
  text_splitter = SentenceSplitter(chunk_size=chunk_size)

  prompt_helper = PromptHelper(
    context_window=context_window,
    num_output=num_output,
    chunk_overlap_ratio=0.1,
    chunk_size_limit=None,
  )

  service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    text_splitter=text_splitter,
    prompt_helper=prompt_helper,
  )

  set_global_service_context(service_context)

  # Now read the corpus json file.
  # It should print the first record for debugging purposes
  reader = JSONReader()
  data = reader.load_data(corpus)
  print('Corpus Data')
  print('--------------------------')
  print(data[0])
  print('--------------------------')

  # Now initialise the model. Download it if it is not cached.
  # Finetune the embeddings for our corpus. 
  print('Initialising pipeline')    
  transformations = [text_splitter,CustomExtractor()] 
  pipeline = IngestionPipeline(transformations=transformations)
  nodes = pipeline.run(documents=data)
  nodes_see = deepcopy(nodes)
  print("LLM sees:\n",(nodes_see)[0].get_content(metadata_mode=MetadataMode.LLM))
  print('Finished Loading...')

  index = VectorStoreIndex(nodes, show_progress=True)
  print('Vector Store Created ...')

  # Now we are finally ready to parse the queries.
  with open(queries, 'r') as file:
    query_data = json.load(file)

  print('Query Data')
  print('--------------------------')
  print(query_data[0])
  print('--------------------------')

  if rerank:
    print('Reranker enabled')
    rerank_postprocessors = FlagEmbeddingReranker(model=rerank_model_name, top_n=topk)

  # Run the retrieval. This will take a while.
  retrieval_save_list = []
  print("Running Retrieval ...")
  for data in tqdm(query_data):
    query = data['query']   
    if rerank:
      nodes_score = index.as_retriever(similarity_top_k=20).retrieve(query)
      nodes_score = rerank_postprocessors.postprocess_nodes(
                      nodes_score, query_bundle=QueryBundle(query_str=query)
                    )
    else:
      nodes_score = index.as_retriever(similarity_top_k=topk).retrieve(query)

    retrieval_list = []
    for ns in nodes_score:
      dic = {}
      dic['text'] = ns.get_content(metadata_mode=MetadataMode.LLM)
      dic['score'] = ns.get_score()
      retrieval_list.append(dic)

    save = {}
    save['query'] = data['query']   
    save['answer'] = data['answer']   
    save['question_type'] = data['question_type'] 
    save['retrieval_list'] = retrieval_list
    save['gold_list'] = data['evidence_list']   
    retrieval_save_list.append(save)

  print('Retieval complete. Saving Results')
  with open(save_file, 'w') as json_file:
    json.dump(retrieval_save_list, json_file)


if __name__ == '__main__':
  if STAGING:
    corpus = "data/sample-corpus.json"
    queries = "data/sample-rag.json"
  else:
    corpus = "data/corpus.json"
    queries = "data/rag.json"

  rank_model_name = "BAAI/bge-large-en-v1.5"
  output_name = "output/rankerC.json"
    
  # For Reranking, Set to True and comment out output_name above and uncomment the
  # three lines below the rerank Boolean.
  # Reranking requires both a rank and rerank model to be defined.
  rerank = False

  # Run the main loop.
  gen_stage_0(corpus, queries, rank_model_name, rerank, None, output_name)
