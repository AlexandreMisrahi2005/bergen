import os
import json
from tqdm import tqdm
from collections import defaultdict
import spacy
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
import argparse

overwrite_exp = True
nlp = None

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 'yes', 'y', '1'}:
        return True
    elif value.lower() in {'false', 'no', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}.")

parser = argparse.ArgumentParser()
parser.add_argument("experiment_path", type=str, help="Path to the experiment output file")
parser.add_argument("--debug", type=str_to_bool, default=False, help="Debug mode (true/false)")
args = parser.parse_args()

experiment_path = args.experiment_path
DEBUG = args.debug

claim_extractor_model_path = "Babelscape/t5-base-summarization-claim-extractor"
claim_extractor_model_path = "chentong00/propositionizer-wiki-flan-t5-large"
nli_model_path = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
claim_extractor_batch_size = 64 # real batch size is 2x this as for each example we have 2 inputs: gt and response
entailement_max_batch_size = 128

suffix = ""
if DEBUG:
    suffix = "_debug"
if DEBUG and claim_extractor_model_path == "Babelscape/t5-base-summarization-claim-extractor":
    suffix = "_babelscapet5base_debug"
if DEBUG and claim_extractor_model_path == "chentong00/propositionizer-wiki-flan-t5-large":
    suffix = "_flant5_debug"

if not overwrite_exp and os.path.exists(experiment_path.replace("eval_dev_out.json", f"ragchecker_dev_out{suffix}.json")):
    raise OSError(f"Output file {experiment_path.replace('eval_dev_out.json', f'ragchecker_dev_out{suffix}.json')} already exists.")

def extract_documents(experiment_output_file):
    for j,line in enumerate(experiment_output_file):
        documents = []
        for i in range(1, 6):
            documents.append(line["instruction"][0].split(f"Document {i}: ")[1].split("\n")[0])
        experiment_output_file[j]["chunks"] = documents
        if DEBUG and j == 2:
            experiment_output_file = experiment_output_file[:2]
            break
    return experiment_output_file
if DEBUG:
    print("extracting docs...")
with open(experiment_path) as f:
    experiment_output_file = json.load(f)
experiment_output_file = extract_documents(experiment_output_file)


assert torch.cuda.is_available()

print("loading claim extractor model...")
with torch.no_grad():
    tokenizer = T5Tokenizer.from_pretrained(claim_extractor_model_path)
    model = T5ForConditionalGeneration.from_pretrained(claim_extractor_model_path).to("cuda")

    print("extracting claims...")
    for i in tqdm(range(0, len(experiment_output_file), claim_extractor_batch_size)):
        if DEBUG and i == 0:
            print("    question: ", experiment_output_file[i]["question"])
            print("    gt label: ", experiment_output_file[i]["label"])
            print("    response: ", experiment_output_file[i]["response"])
        # by default we assume that each example has only one label
        # assert all(len(experiment_output_file[j]["label"]) == 1 for j in range(i, min(i+claim_extractor_batch_size, len(experiment_output_file))))
        inputs = [experiment_output_file[j]["label"][0] for j in range(i, min(i+claim_extractor_batch_size, len(experiment_output_file)))] + \
                 [experiment_output_file[j]["response"] for j in range(i, min(i+claim_extractor_batch_size, len(experiment_output_file)))]
        try:
            tok_input = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True).to("cuda")
            model_out = model.generate(**tok_input).cpu()
        except torch.OutOfMemoryError as e:
            print("CUDA OOM. Reduce claim_extractor_batch_size")
            raise e
        gt_and_response_claims = tokenizer.batch_decode(model_out, skip_special_tokens=True)
        for j in range(i, min(i+claim_extractor_batch_size, len(experiment_output_file))):
            experiment_output_file[j]["gt_claims"] = gt_and_response_claims[j - i]
            experiment_output_file[j]["response_claims"] = gt_and_response_claims[j - i + len(inputs)//2]
        if DEBUG and i == 0:
            print("    gt claims extracted: ", gt_and_response_claims[0])
            print("    response claims extracted: ", gt_and_response_claims[min(claim_extractor_batch_size, len(experiment_output_file))])

    del model
    del tokenizer
    torch.cuda.empty_cache()

print("sentencizing claims...")

### from https://github.com/amazon-science/RefChecker/blob/bd516aff77b7d69a3e4c899dc9826794c5de4ab7/refchecker/utils.py#L25
def sentencize(text):
    """Split text into sentences"""
    global nlp
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return list(set([str(sent) for sent in doc.sents]))

for i in range(len(experiment_output_file)):
    experiment_output_file[i]["response_claims_sents"] = sentencize(experiment_output_file[i]["response_claims"])
    experiment_output_file[i]["gt_claims_sents"] = sentencize(experiment_output_file[i]["gt_claims"])

if DEBUG:
    print(f"    {len(experiment_output_file[0]['response_claims_sents'])} response claims: ", experiment_output_file[0]["response_claims_sents"])
    print(f"    {len(experiment_output_file[0]['gt_claims_sents'])} gt claims: ", experiment_output_file[0]["gt_claims_sents"])

print("loading nli model...")
with torch.no_grad():
    tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_path).to("cuda")

    print("generating entailment results...")
    for i in tqdm(range(len(experiment_output_file))):
        max_length = 256
        
        premises = []
        hypotheses = []
        # define all premise-hypothesis pairs we need to compute
        # are response claims entailed by the ground-truth?
        for j, response_claim in enumerate(experiment_output_file[i]["response_claims_sents"]):
            premises.append(experiment_output_file[i]["label"][0])
            hypotheses.append(response_claim)
        # are ground truth claims entailed by the response?
        for j, gt_claim in enumerate(experiment_output_file[i]["gt_claims_sents"]):
            premises.append(gt_claim)
            hypotheses.append(experiment_output_file[i]["response"])
        # are ground truth claims entailed by each chunk?
        for j, gt_claim in enumerate(experiment_output_file[i]["gt_claims_sents"]):
            for k, chunk in enumerate(experiment_output_file[i]["chunks"]):
                premises.append(chunk)
                hypotheses.append(gt_claim)
        # are response claims entailed by each chunk?
        for j, response_claim in enumerate(experiment_output_file[i]["response_claims_sents"]):
            for k, chunk in enumerate(experiment_output_file[i]["chunks"]):
                premises.append(chunk)
                hypotheses.append(response_claim)

        assert len(premises) == len(hypotheses)

        if len(premises) > entailement_max_batch_size:
            batch_premises = []
            batch_hypotheses = []
            for j in range(0, len(premises), entailement_max_batch_size):
                batch_premises.append(premises[j:j+entailement_max_batch_size])
                batch_hypotheses.append(hypotheses[j:j+entailement_max_batch_size])
        else:
            batch_premises = [premises]
            batch_hypotheses = [hypotheses]

        # print(f"{len(premises)} pairs will be inferred in {len(batch_premises)} batch(es)")
        try:
            outputs = []
            for j in range(len(batch_premises)):
                tokenized_input_seq_pairs = tokenizer.batch_encode_plus(list(zip(batch_premises[j], batch_hypotheses[j])),
                                                                        max_length=max_length,
                                                                        padding=True,
                                                                        truncation=True,
                                                                        return_token_type_ids=True, 
                                                                        return_attention_mask=True,
                                                                        return_tensors="pt"
                                                                    ).to("cuda")
                input_ids = tokenized_input_seq_pairs['input_ids']
                attention_mask = tokenized_input_seq_pairs['attention_mask']
                token_type_ids = tokenized_input_seq_pairs['token_type_ids']
                batch_outputs = model(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=None)

                outputs.append(batch_outputs.logits.cpu())
            outputs = torch.cat(outputs, dim=0)

        except torch.OutOfMemoryError as e:
            print("CUDA OOM. Reduce entailement_max_batch_size")
            raise e
        
        # if DEBUG and i == 0:
        #     print("outputs.shape = ", outputs.shape) # (bs, 3) <--> (bs, (entailment, neutral, contradiction))

        predicted_probability = torch.softmax(outputs, dim=1).cpu().detach().numpy()

        # if DEBUG and i == 0:
        #     # print the first 10 pairs
        #     for j in range(10):
        #         print(f"    pair {j}: ", premises[j], " | ", hypotheses[j], " | ", predicted_probability[j])

        # keep a dictionary with keys like "<hypothesis_i>_isentailedby_<premise_j>" e.g. "response_claim_0_isentailedby_gt"
        entailment_dict = {}

        for j in range(len(premises)):
            if j < len(experiment_output_file[i]["response_claims_sents"]):
                key = f"response_claim_{j}_isentailedby_gt"
            elif j < len(experiment_output_file[i]["gt_claims_sents"]) + len(experiment_output_file[i]["response_claims_sents"]):
                key = f"gt_claim_{j-len(experiment_output_file[i]['response_claims_sents'])}_isentailedby_response"
            elif j < len(experiment_output_file[i]["gt_claims_sents"]) + len(experiment_output_file[i]["response_claims_sents"]) + len(experiment_output_file[i]["chunks"]) * len(experiment_output_file[i]["gt_claims_sents"]):
                offset = len(experiment_output_file[i]["gt_claims_sents"]) + len(experiment_output_file[i]["response_claims_sents"])
                chunk_offset = (j - offset) // len(experiment_output_file[i]["chunks"])
                chunk_idx = (j - offset) % len(experiment_output_file[i]["chunks"])
                key = f"gt_claim_{chunk_offset}_isentailedby_chunk_{chunk_idx}"
            else:
                offset = len(experiment_output_file[i]["gt_claims_sents"]) + len(experiment_output_file[i]["response_claims_sents"]) + len(experiment_output_file[i]["chunks"]) * len(experiment_output_file[i]["gt_claims_sents"])
                chunk_offset = (j - offset) // len(experiment_output_file[i]["chunks"])
                chunk_idx = (j - offset) % len(experiment_output_file[i]["chunks"])
                key = f"response_claim_{chunk_offset}_isentailedby_chunk_{chunk_idx}"

            entailment_dict[key] = {
                "premise": premises[j],
                "hypothesis": hypotheses[j],
                "probabilities": predicted_probability[j].tolist(),
                "predicted_label": np.argmax([predicted_probability[j][1] + predicted_probability[j][2], predicted_probability[j][0]]).item()
            }

        # add some metadata
        entailment_dict["num_chunks"] = len(experiment_output_file[i]["chunks"])
        entailment_dict["num_gt_claims"] = len(experiment_output_file[i]["gt_claims_sents"])
        entailment_dict["num_response_claims"] = len(experiment_output_file[i]["response_claims_sents"])

        assert len(premises) == len(entailment_dict) - 3

        # Store the entailment dictionary in the experiment output file
        experiment_output_file[i]["entailment_results"] = entailment_dict

    del model
    del tokenizer
    torch.cuda.empty_cache()

if DEBUG:
    print(f"{len(experiment_output_file[0]['entailment_results'])} entailment results: ", sorted(list(experiment_output_file[0]["entailment_results"].keys())))
    for key in sorted(list(experiment_output_file[0]["entailment_results"].keys())):
        print(key.split("_isentailedby_")[0], " = ", experiment_output_file[0]["entailment_results"][key]["hypothesis"], "\nis entailed by\n", experiment_output_file[0]["entailment_results"][key]["premise"], "\nwith probability [entailed, neutral, contradiction] =", experiment_output_file[0]["entailment_results"][key]["probabilities"])


class RAGCheckerMetrics:
    def __init__(self, experiment_output_file):
        self.experiment_output_file = experiment_output_file
        self.qid2metrics = {}
        self.mean_metrics = defaultdict(dict)

    def compute_metrics(self):
        for i in tqdm(range(len(self.experiment_output_file))):
            entailment_results = self.experiment_output_file[i]["entailment_results"]
            if entailment_results["num_gt_claims"] == 0 or entailment_results["num_response_claims"] == 0:
                print(f"Skipping example id == {self.experiment_output_file[i]['q_id']} as it has no gt claims or response claims")
                continue
            out = {
                "overall_metrics": {},
                "retriever_metrics": {},
                "generator_metrics": {}
            }
            out["overall_metrics"]["precision"] = self.precision(entailment_results)
            out["overall_metrics"]["recall"] = self.recall(entailment_results)
            out["retriever_metrics"]["claim_recall"] = self.claim_recall(entailment_results)
            out["retriever_metrics"]["context_precision"] = self.context_precision(entailment_results)
            out["generator_metrics"]["faithfulness"] = self.faithfulness(entailment_results)
            out["generator_metrics"]["relevant_noise_sensitivity"] = self.relevant_noise_sensitivity(entailment_results)
            out["generator_metrics"]["irrelevant_noise_sensitivity"] = self.irrelevant_noise_sensitivity(entailment_results)
            out["generator_metrics"]["hallucination"] = self.hallucination(entailment_results)
            out["generator_metrics"]["self_knowledge"] = self.self_knowledge(entailment_results)
            out["generator_metrics"]["context_utilization"] = self.context_utilization(entailment_results)
            self.qid2metrics[self.experiment_output_file[i]["q_id"]] = out
        # Compute mean metrics
        for metric_type in out.keys():
            for metric in out[metric_type].keys():
                self.mean_metrics[metric_type][metric] = np.mean([v[metric_type][metric] for v in self.qid2metrics.values()]).astype(float)
    
    @staticmethod
    def compute_relevant_chunks(entailment_results):
        """Indexes of chunks that entail at least one of the gt claims"""
        relevant_chunks = set()
        for k in range(entailment_results["num_chunks"]):
            for j in range(entailment_results["num_gt_claims"]):
                key = f"gt_claim_{j}_isentailedby_chunk_{k}"
                if entailment_results[key]["predicted_label"] == 1:  # entailment
                    relevant_chunks.add(k)
        return relevant_chunks

    @staticmethod
    def compute_irrelevant_chunks(entailment_results):
        """Indexes of chunks that entail none of the gt claims"""
        all_chunks = set(range(entailment_results["num_chunks"]))
        relevant_chunks = RAGCheckerMetrics.compute_relevant_chunks(entailment_results)
        return all_chunks - relevant_chunks
    
    @staticmethod
    def precision(entailment_results):
        """Number of response claims entailed by the ground-truth divided by the total number of response claims"""
        entailed_count = sum(
            entailment_results[f"response_claim_{j}_isentailedby_gt"]["predicted_label"] == 1
            for j in range(entailment_results["num_response_claims"])
        )
        return entailed_count / entailment_results["num_response_claims"]

    @staticmethod
    def recall(entailment_results):
        """Number of gt claims entailed by the response divided by the total number of gt claims"""
        entailed_count = sum(
            entailment_results[f"gt_claim_{j}_isentailedby_response"]["predicted_label"] == 1
            for j in range(entailment_results["num_gt_claims"])
        )
        return entailed_count / entailment_results["num_gt_claims"]

    @staticmethod
    def claim_recall(entailment_results):
        """Number of gt claims entailed by at least one of the chunks divided by the total number of gt claims"""
        entailed_count = sum(
            any(entailment_results[f"gt_claim_{j}_isentailedby_chunk_{k}"]["predicted_label"] == 1 
                for k in range(entailment_results["num_chunks"]))
            for j in range(entailment_results["num_gt_claims"])
        )
        return entailed_count / entailment_results["num_gt_claims"]

    @staticmethod
    def context_precision(entailment_results):
        """Number of relevant chunks divided by the total number of chunks"""
        relevant_chunks = RAGCheckerMetrics.compute_relevant_chunks(entailment_results)
        return len(relevant_chunks) / entailment_results["num_chunks"]

    @staticmethod
    def faithfulness(entailment_results):
        """Number of response claims entailed by at least one chunk, divided by the total number of response claims"""
        entailed_count = sum(
            any(entailment_results[f"response_claim_{j}_isentailedby_chunk_{k}"]["predicted_label"] == 1
                for k in range(entailment_results["num_chunks"]))
            for j in range(entailment_results["num_response_claims"])
        )
        return entailed_count / entailment_results["num_response_claims"]

    @staticmethod
    def relevant_noise_sensitivity(entailment_results):
        """Number of response claims not entailed by the ground truth and entailed by relevant chunks, divided by the total number of response claims"""
        relevant_chunks = RAGCheckerMetrics.compute_relevant_chunks(entailment_results)
        noisy_count = sum(
            entailment_results[f"response_claim_{j}_isentailedby_gt"]["predicted_label"] == 0 and
            any(entailment_results[f"response_claim_{j}_isentailedby_chunk_{k}"]["predicted_label"] == 1 for k in relevant_chunks)
            for j in range(entailment_results["num_response_claims"])
        )
        return noisy_count / entailment_results["num_response_claims"]

    @staticmethod
    def irrelevant_noise_sensitivity(entailment_results):
        """Number of response claims not entailed by the ground truth and entailed by irrelevant chunks, divided by the total number of response claims"""
        irrelevant_chunks = RAGCheckerMetrics.compute_irrelevant_chunks(entailment_results)
        noisy_count = sum(
            entailment_results[f"response_claim_{j}_isentailedby_gt"]["predicted_label"] == 0 and
            any(entailment_results[f"response_claim_{j}_isentailedby_chunk_{k}"]["predicted_label"] == 1 for k in irrelevant_chunks)
            for j in range(entailment_results["num_response_claims"])
        )
        return noisy_count / entailment_results["num_response_claims"]

    @staticmethod
    def hallucination(entailment_results):
        """Number of response claims not entailed by the ground truth and not entailed by any of the chunks, divided by the total number of response claims"""
        hallucinated_count = sum(
            entailment_results[f"response_claim_{j}_isentailedby_gt"]["predicted_label"] == 0 and
            all(entailment_results[f"response_claim_{j}_isentailedby_chunk_{k}"]["predicted_label"] == 0 for k in range(entailment_results["num_chunks"]))
            for j in range(entailment_results["num_response_claims"])
        )
        return hallucinated_count / entailment_results["num_response_claims"]

    @staticmethod
    def self_knowledge(entailment_results):
        """Number of response claims entailed by the ground truth, and not entailed by any of the chunks, divided by the total number of response claims"""
        self_knowledge_count = sum(
            entailment_results[f"response_claim_{j}_isentailedby_gt"]["predicted_label"] == 1 and
            all(entailment_results[f"response_claim_{j}_isentailedby_chunk_{k}"]["predicted_label"] == 0 for k in range(entailment_results["num_chunks"]))
            for j in range(entailment_results["num_response_claims"])
        )
        return self_knowledge_count / entailment_results["num_response_claims"]

    @staticmethod
    def context_utilization(entailment_results):
        """Number of gt claims entailed by any of the chunks and entailed by the response, divided by the total number of gt claims that are entailed by any of the chunks"""
        entailed_chunks = [
            any(entailment_results[f"gt_claim_{j}_isentailedby_chunk_{k}"]["predicted_label"] == 1
                for k in range(entailment_results["num_chunks"]))
            for j in range(entailment_results["num_gt_claims"])
        ]
        entailed_by_response = [
            entailment_results[f"gt_claim_{j}_isentailedby_response"]["predicted_label"] == 1
            for j in range(entailment_results["num_gt_claims"])
        ]
        entailed_and_utilized = sum(e and r for e, r in zip(entailed_chunks, entailed_by_response))
        total_entailed = sum(entailed_chunks)
        return entailed_and_utilized / total_entailed if total_entailed > 0 else 0


print("computing metrics...")
metrics = RAGCheckerMetrics(experiment_output_file)
metrics.compute_metrics()
print("\n\n=== RAG-CHECKER METRICS ===")
print(dict(metrics.mean_metrics), "\n\n")

for i in range(len(experiment_output_file)):
    experiment_output_file[i]["ragchecker_metrics"] = metrics.qid2metrics.get(experiment_output_file[i]["q_id"], None)

with open(experiment_path.replace("eval_dev_out.json", f"ragchecker_dev_out{suffix}.json"), "w") as f:
    if DEBUG:
        columns = ["q_id", "response", "instruction", "label", "question", "ranking_label", "M", "EM", "F1", "Precision", "Recall", "Recall_char3gram", "Rouge-1", "Rouge-2", "Rouge-L", "chunks", "response_claims_sents", "gt_claims_sents", "entailment_results", "ragchecker_metrics"]
    else:
        columns = ["q_id", "response", "instruction", "label", "question", "ranking_label", "M", "EM", "F1", "Precision", "Recall", "Recall_char3gram", "Rouge-1", "Rouge-2", "Rouge-L", "ragchecker_metrics"]
    json.dump([{k: v for k, v in line.items() if k in columns} for line in experiment_output_file], f, indent=4)

with open(experiment_path.replace("eval_dev_out.json", f"ragchecker_dev_metrics{suffix}.json"), "w") as f:
    json.dump(metrics.mean_metrics, f, indent=4)

print("hackathon.py done.")