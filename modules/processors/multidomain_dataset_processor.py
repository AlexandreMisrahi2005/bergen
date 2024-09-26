from ..dataset_processor import *
import datasets
import json

from tqdm import tqdm
from hydra.utils import instantiate
import requests  
import pandas as pd

from urllib.parse import unquote
from bs4 import BeautifulSoup


class BIOASQ11B_Ragged(Processor):
    """BIOASQ benchmark, dataset from Ragged paper"""

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'BIOASQ11B_Ragged'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        hf_name = "jenhsia/ragged"
        dataset = datasets.load_dataset(hf_name, 'bioasq', num_proc=self.num_proc)[self.split]
        # ['id', 'input', 'output', 'question_type']
        dataset = dataset.map(lambda example: {'label': [dictt["answer"] for dictt in example["output"] if dictt["answer"] is not None]})
        dataset = dataset.map(lambda example: {'label': [" ".join(example['label'])] if example['question_type'] == 'list' else example['label']})  # concatenate list question types to compute recall on all labels (see metrics computation)
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['question_type', 'output'])
        return dataset
    
class PubMed2023_Ragged(Processor):
    """PubMed abstracts, dataset from Ragged paper"""

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'PubMed-2023_Ragged'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        hf_name = "jenhsia/ragged"
        dataset = datasets.load_dataset(hf_name, 'pubmed', num_proc=self.num_proc)[self.split].select(range(100))

        concatenated_data = {}
        for row in tqdm(dataset):
            real_id, field_type = row['id'].split('_')

            if real_id not in concatenated_data:
                concatenated_data[real_id] = {"title": "", "content": ""}
            
            if field_type == '0':
                concatenated_data[real_id]["title"] = row['contents']
            elif field_type == '1':
                concatenated_data[real_id]["content"] = row['contents']
                
        concatenated_rows = []
        for real_id, fields in tqdm(concatenated_data.items()):
            title = fields["title"]
            content = fields["content"]
            concatenated_content = f"{title}: {content}" if content else title
            concatenated_rows.append({"id": real_id, "content": concatenated_content})

        dataset = datasets.Dataset.from_list(concatenated_rows)

        return dataset

def process_APIBench_gorilla(apibench_file):
    def download_file(url):
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        data = []
        for line in response.text.strip().split('\n'):
            data.append(json.loads(line))
        return data

    def index(data):
        """
        data: list of python dictionaries
        """
        for i,_ in enumerate(data):
            data[i]['id'] = i
            if 'performance' in data[i]:
                del data[i]['performance']
        return data

    api_data = download_file(apibench_file)
    
    print('Processing...')
    indexed_api_bench = index(api_data)
    tmp_df = pd.DataFrame(data=indexed_api_bench)

    def get_instruction(x):
        tmp = x.split('###Instruction:')
        if len(tmp) == 2:
            tmp = tmp[1].split('###Output:')
            return tmp[0].strip().replace('\n', '\\')
        else:
            tmp = x.split('### Instruction:')
            if len(tmp) == 2:
                tmp = tmp[1].split('###Output:')
                if len(tmp) >= 2:
                    return tmp[0].strip().replace('\n', '\\')
                else:
                    tmp = tmp[0].split('### Output:')
                    if len(tmp) >= 2:
                        return tmp[0].strip().replace('\n', '\\')
            else:
                return None
        
    tmp_df['content'] = tmp_df['code'].apply(get_instruction)
    def listify_label(row):
        row['label'] = [row['label']]
        return row
    tmp_df['label'] = tmp_df['api_call']
    tmp_df = tmp_df.drop(['code', 'provider', 'api_data'], axis=1).dropna()
    api_bench_dataset = datasets.Dataset.from_pandas(tmp_df)
    if "__index_level_0__" in api_bench_dataset.column_names:
        api_bench_dataset = api_bench_dataset.remove_columns(["api_call", "__index_level_0__"]).cast_column('id', datasets.Value('string')).map(listify_label)
    else:
        api_bench_dataset = api_bench_dataset.remove_columns(["api_call"]).cast_column('id', datasets.Value('string')).map(listify_label)
    print('Done.')
    return api_bench_dataset

class APIBench_gorilla_HF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'APIBench_gorilla_HF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        apibench_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/apibench/huggingface_eval.json'
        api_bench_dataset = process_APIBench_gorilla(apibench_file)
        return api_bench_dataset
    
class APIBench_gorilla_TF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'APIBench_gorilla_TF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        apibench_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/apibench/tensorflow_eval.json'
        api_bench_dataset = process_APIBench_gorilla(apibench_file)
        return api_bench_dataset
    
class APIBench_gorilla_TH(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'APIBench_gorilla_TH'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        apibench_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/apibench/torchhub_eval.json'
        api_bench_dataset = process_APIBench_gorilla(apibench_file)
        return api_bench_dataset


def process_API_gorilla(api_file):
    def download_file(url):
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        data = []
        for line in response.text.strip().split('\n'):
            data.append(json.loads(line))
        return data
    
    def index(data):
        """
        data: list of python dictionaries
        """
        for i,_ in enumerate(data):
            data[i]['id'] = i
            if 'performance' in data[i]:
                del data[i]['performance']
        return data
    
    def convert_to_string(x):
        if isinstance(x, list):  # Check if the element is a list
            if len(x) > 0:
                if isinstance(x[0], dict):
                    return str(x)
            return ','.join(x)   # Join the list elements with commas
        return str(x)                 # If it's not a list, return the original string

    api_data = download_file(api_file)
    print('Processing...')
    indexed_api = index(api_data) # create indices
    tmp_df = pd.DataFrame(data=indexed_api)
    tmp_df['api_arguments'] = tmp_df['api_arguments'].apply(convert_to_string)
    tmp_df['python_environment_requirements'] = tmp_df['python_environment_requirements'].apply(convert_to_string)
    tmp_df['example_code'] = tmp_df['example_code'].apply(convert_to_string)
    tmp_df['functionality'] = tmp_df['functionality'].apply(convert_to_string)
    api_dataset = datasets.Dataset.from_pandas(tmp_df).map(lambda row: {'content':'\n'.join([f"{key}: {value};" for key,value in row.items() if key != 'id'])})
    api_dataset = api_dataset.remove_columns([column for column in api_dataset.column_names if column not in ['id', 'content']]).cast_column('id', datasets.Value('string'))
    print('Done.')
    return api_dataset

class API_gorilla_HF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'API_gorilla_HF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        """
        self.split should be one of ['huggingface', 'torchhub', 'tensorflowhub']
        """
        api_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/api/huggingface_api.jsonl'
        api_dataset = process_API_gorilla(api_file)
        return api_dataset
    

class API_gorilla_TF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'API_gorilla_TF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        """
        self.split should be one of ['huggingface', 'torchhub', 'tensorflowhub']
        """
        api_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/api/tensorflowhub_api.jsonl'
        api_dataset = process_API_gorilla(api_file)
        return api_dataset
    
class API_gorilla_TH(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'API_gorilla_TH'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        """
        self.split should be one of ['huggingface', 'torchhub', 'tensorflowhub']
        """

        api_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/api/torchhub_api.jsonl'
        api_dataset = process_API_gorilla(api_file)
        return api_dataset
    

class CodeRAGBench_HumanEval(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = "CodeRAGBench_HumanEval"
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        # load from the CodeRAGBench HF repo
        hf_name = "code-rag-bench/humaneval"
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
        dataset = dataset.rename_column("task_id", "id").rename_column("prompt", "content").rename_column("canonical_solution", "label")
        def listify_label(row):
            row['label'] = [row['label']]
            return row
        dataset = dataset.map(listify_label)
        return dataset


class CodeRAGBench_MBPP(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = "CodeRAGBench_MBPP"
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        # load from the CodeRAGBench HF repo
        hf_name = "code-rag-bench/mbpp"
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
        dataset = dataset.rename_column("task_id", "id").rename_column("text", "content").rename_column("code", "label")
        dataset = dataset.remove_columns([column for column in dataset.column_names if column not in ['id', 'content', 'label']])
        def listify_label(row):
            row['label'] = [row['label']]
            return row
        dataset = dataset.map(listify_label)
        return dataset
    
def CRB_cat_title_content(x, content_colname="content", title_colname="title"):
    if title_colname is None:
        x["content"] = f"{x[content_colname]}"
    else:
        x["content"] = f"{x[title_colname]}: {x[content_colname]}"
    return x

class CodeRAGBench_programming_solutions(Processor):
    """Contains oracle docs for HumanEval and MBPP"""
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_programming_solutions'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/programming-solutions", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"text"}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_online_tutorials(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_online_tutorials'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/online-tutorials", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"text"}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_library_documentation(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_library_documentation'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/library-documentation", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"doc_content", "title_colname":"doc_id"}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_stackoverflow(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_stackoverflow'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/stackoverflow-posts", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset

class CodeRAGBench_gitrepospython(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_gitrepospython'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/github-repos-python", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_gitrepos(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_gitrepos'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/github-repos", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset

class SyllabusQA(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = "SyllabusQA"
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        assert self.split in ['train', 'val', 'test'], "Wrong dataset split, should be one of 'train', 'val', or 'test'."
        url = f"https://raw.githubusercontent.com/umass-ml4ed/SyllabusQA/main/data/dataset_split/{self.split}.csv"
        df = pd.read_csv(url)
        def merge_syllabusname_question(row):
            row['content'] = row['syllabus_name'] + ": " + row['question']
            return row
        dataset = datasets.Dataset.from_pandas(df).map(merge_syllabusname_question).rename_column("answer", "label").remove_columns([
            'answer_span_1',
            'answer_span_2',
            'answer_span_3',
            'answer_span_4',
            'answer_span_5',
            'reasoning_step_1',
            'reasoning_step_2',
            'reasoning_step_3',
            'reasoning_step_4',
            'reasoning_step_5',
            ])
        def listify_label(row):
            row['label'] = [row['label']]
            return row
        dataset = dataset.map(listify_label)
        return dataset
    
class SyllabusQA_syllabi(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = "SyllabusQA_syllabi"
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        """
        Important notice: 
        Due to requests feature, sometimes this function fails with ```datasets.arrow_writer.SchemaInferenceError: Please pass `features` or at least one example when writing data```
        This happens when the request fails to retrieve all the txt files (and retrieves none of them) resulting in an empty Dataset
        Just run the function until it works... BUT make sure to remove the directory SyllabusQA_syllabi_train created in datasets/
        """

        url = "https://github.com/umass-ml4ed/SyllabusQA/tree/main/syllabi/syllabi_redacted/text"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # parse the HTML to find all .txt file links
        base_url = "https://raw.githubusercontent.com/umass-ml4ed/SyllabusQA/main/syllabi/syllabi_redacted/text/"
        files = []

        # get file links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.txt'):
                file_name = href.split('/')[-1]  # Get file name
                files.append(base_url + file_name)
        
        # duplicates
        files = list(set(files))
        print(len(files), 'files found.')

        # download each .txt
        syllabi = []
        for file_url in files:
            file_name = file_url.split('/')[-1]
            response = requests.get(file_url)
            content = response.content.decode('MacRoman')
            syllabi.append({'file_name': file_name, 'content': content})
            print(f"Downloaded {file_name}.")
        print("Done.")

        def chunk_text(text, title, max_size=1000, overlap=200):
            """
            Chunks the given text into parts with a maximum size and overlap, prepending the title to each chunk.
            
            Args:
            - text: The text to chunk.
            - title: The title of the syllabus to prepend to each chunk.
            - max_size: Maximum size of each chunk (default is 1000 characters, same as in https://arxiv.org/pdf/2403.14666).
            - overlap: Overlap between adjacent chunks (default is 200 characters, same as in https://arxiv.org/pdf/2403.14666).
            
            Returns:
            - A list of dictionaries with chunk 'id' and 'content' keys.
            """
            chunks = []
            start = 0
            chunk_id = 0
            while start < len(text):
                end = start + max_size
                chunk = text[start:end]
                chunk = title + ": " + chunk  # Prepend the title
                chunks.append({'id': f"{title}_{chunk_id}", 'content': chunk})
                start = end - overlap
                chunk_id += 1

            return chunks

        # chunk
        all_chunks = []
        for file in syllabi:
            title, text = unquote(file['file_name'].split('/')[-1].strip('.txt')), file['content']
            chunks = chunk_text(text, title)
            all_chunks.extend(chunks)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(all_chunks))
        return dataset

class NarrativeQA(Processor):
    def __init__(self, *args, **kwargs):
        dataset_name = 'NarrativeQA'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        ds = datasets.load_dataset("deepmind/narrativeqa", num_proc=self.num_proc)[self.split]

        def map_fn(example):
            example["id"] = example["document"]["id"]
            example['content'] = f"{example['document']['summary']['title']}: {example['question']['text'].lower()}"
            example['label'] = [example["answers"][1]["text"]]
            return example
        
        ds = ds.map(map_fn, num_proc=self.num_proc)
        return ds

class NarrativeQA_docs(Processor):
    def __init__(self, *args, **kwargs):
        self.chunk_size, self.overlap, self.type = None, None, None
        if "type" in kwargs:
            self.type = kwargs["type"]
            del kwargs['type']
        if "chunk_size" in kwargs and "overlap" in kwargs:
            self.chunk_size = kwargs["chunk_size"]
            self.overlap = kwargs["overlap"]
            del kwargs['chunk_size']
            del kwargs['overlap']
        dataset_name = f"NarrativeQA_docs_CHS_{self.chunk_size}_OVLP_{self.overlap}" if self.chunk_size is not None and self.overlap is not None else "NarrativeQA_docs"
        dataset_name = "NarrativeQA_docs_summary" if self.type == "summary" else "NarrativeQA_docs"
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        ds = datasets.load_dataset("deepmind/narrativeqa", num_proc=self.num_proc)[self.split]

        def chunk_text(text, chunk_size=100, overlap=20):
            # chunk according to words
            tokens = text.split()
            num_tokens = len(tokens)
            
            chunks = []
            for i in range(0, num_tokens, chunk_size - overlap):
                chunk_tokens = tokens[i:i + chunk_size]
                
                # If this is the last chunk and there's still some remaining text, capture it fully
                if i + chunk_size >= num_tokens:
                    chunk_tokens = tokens[i:]
                    
                chunk = ' '.join(chunk_tokens)
                chunks.append(chunk)
                
                if len(chunk_tokens) < chunk_size:
                    break
            return chunks
                
        new_data = {"id": [], "content": []}
        ids_seen = set()
        for row in ds:
            doc_id = row["document"]["id"]
            if doc_id in ids_seen:
                continue
            ids_seen.add(doc_id)
            if self.type is not None and self.type == "summary":
                text = f"{row['document']['summary']['title']}: {row['document']['summary']['text']}"
                new_id = f"{doc_id}"
                new_data["id"].append(new_id)
                new_data["content"].append(text)
            else:
                text = row["document"]["text"]
                chunks = chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)
                for chunk_idx, chunk in enumerate(chunks):
                    new_id = f"{doc_id}_{chunk_idx + 1}"
                    new_data["id"].append(new_id)
                    chunk = f"{row['document']['summary']['title']}: {chunk}"
                    new_data["content"].append(chunk)
        chunked_dataset = datasets.Dataset.from_dict(new_data)
                
        return chunked_dataset
    

class MultiQA(Processor):
    """
    This dataset contains a combination of the following QA datasets:

        < QUERY >                     |     < DOCS >
        nq open                       |     odqa-wiki-corpora-100w-karpukhin
        msmarco 2.1 (first 100k)      |     ms-marco HF dir: irds/msmarco-passage; on bergen: ms-marco_full
        adverserial qa                |     squad -> wikipedia -> kilt-100w
        hotpotqa                      |     kilt-100w
        wikiqa                        |     kilt-100w
        sciq                          |     kilt-100w
        asqa                          |     kilt-100w
        triviaqa                      |     kilt-100w
        freebase_qa                   |     freebase extract from https://github.com/kelvin-jiang/FreebaseQA but not sure how to integrate this ; not done
        squad_v1.1                    |     wikipedia -> kilt-100w
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'MultiQA'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        ds = datasets.load_dataset("dmrau/multi_qa", num_proc=self.num_proc)[self.split]
        return ds
