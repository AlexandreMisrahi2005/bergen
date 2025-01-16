from tqdm import tqdm
import datasets
import regex
import string
from collections import Counter, defaultdict

def normalize(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_single(prediction: str, ground_truth: str, tokenfun=lambda x: x.split()):
    prediction_tokens = tokenfun(normalize(prediction))
    ground_truth_tokens = tokenfun(normalize(ground_truth))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

class BioASQMetrics():
    def __init__(self, dataset_dir="datasets/BIOASQ12B_", split="dev"):
        # split is either 'dev' or 'train'
        # build index to question type mapping
        self.index_to_question_type = {}
        self.index_to_original_references = {}
        original_dataset = datasets.load_from_disk(dataset_dir + split)
        for item in original_dataset:
            self.index_to_question_type[item['id']] = item['type'] # one of 'yesno', 'list', 'factoid'
            self.index_to_original_references[item['id']] = item['label_originalformat'] # useful for list types 
    
    def __call__(self, predictions, references, ids):
        yesno_correct = 0
        yesno_total = 0
        yesno_yes_tp = 0
        yesno_yes_fp = 0
        yesno_yes_fn = 0
        yesno_no_tp = 0
        yesno_no_fp = 0
        yesno_no_fn = 0

        factoid_Scorrect = 0
        factoid_Lcorrect = 0
        factoid_total = 0

        list_precision = 0
        list_recall = 0
        list_f1 = 0
        list_total = 0

        metrics_per_question = [defaultdict(int) for _ in range(len(ids))]
        
        for idx, id in tqdm(enumerate(ids)):
            if self.index_to_question_type[id] == 'yesno':
                # yesno question
                # check if the prediction is correct
                if references[idx][0].lower() == 'yes':
                    yesno_total += 1
                    if 'yes' in predictions[idx].lower():
                        yesno_correct += 1
                        yesno_yes_tp += 1
                        metrics_per_question[idx].update({'yesno': 1})
                    else:
                        yesno_yes_fn += 1
                        yesno_no_fp += 1
                        metrics_per_question[idx].update({'yesno': 0})

                elif references[idx][0].lower() == 'no':
                    yesno_total += 1
                    if 'yes' in predictions[idx].lower():
                        yesno_no_fn += 1
                        yesno_yes_fp += 1
                        metrics_per_question[idx].update({'yesno': 0})
                    else:
                        yesno_correct += 1
                        yesno_no_tp += 1
                        metrics_per_question[idx].update({'yesno': 1})
                else:
                    print("WARNING: unknown yesno question type = ", references[idx][0])
        
            elif self.index_to_question_type[id] == 'factoid':
                # factoid question
                if references[idx][0].lower() in predictions[idx].lower():
                    factoid_Scorrect += 1
                    metrics_per_question[idx].update({'factoid_strict': 1})
                elif any(ref.lower() in predictions[idx].lower() for ref in references[idx]):
                    factoid_Lcorrect += 1
                    metrics_per_question[idx].update({'factoid_lenient': 1})
                else:
                    metrics_per_question[idx].update({'factoid_lenient': 0})
                factoid_total += 1

            elif self.index_to_question_type[id] == 'list':
                # references[idx] will be a list of strings where each string is a list of items
                # we need to compute mean precision, recall, F1
                # thus we start by flattening the golden list
                # flattened_ref = defaultdict(list)
                # num_elements = None
                # for i,ref in enumerate(references[idx]):
                #     elements = [e.strip() for e in ref.split(',')]
                #     num_elements = len(elements) if num_elements is None else num_elements
                #     assert num_elements == len(elements), "All references should have the same number of elements id =" + id + " reference =" + str(references[idx]) + " i = " + str(i)
                #     for i,e in enumerate(elements):
                #         flattened_ref[i].append(e.lower())


                flattened_ref = self.index_to_original_references[id]
                num_elements = len(flattened_ref)

                # now we compute the precision, recall, F1
                tp = 0 # the number of elements mentionned both in the predictions and the references (or any of the synonyms)
                fn = 0
                for i in range(num_elements):
                    if any(e in predictions[idx].lower() for e in flattened_ref[i]):
                        tp += 1
                    else:
                        fn += 1
                fp = len(predictions[idx].split(',')) - tp
                list_precision += tp / (tp + fp) if (tp + fp) > 0 else 0
                list_recall += tp / (tp + fn) if (tp + fn) > 0 else 0
                list_f1 += 2 * list_precision * list_recall / (list_precision + list_recall) if (list_precision + list_recall) > 0 else 0
                list_total += 1
                metrics_per_question[idx].update({'list_recall': list_recall, 'list_precision': list_precision, 'list_f1': list_f1})

                # f1, precision, recall = [max(values) for values in zip(*[f1_single(predictions[idx], gt, tokenfun=lambda x: x.split(',')) for gt in self.index_to_original_references[id]])]
                # list_precision += precision
                # list_recall += recall
                # list_f1 += f1
                # list_total += 1
                # metrics_per_question[idx].update({'list_recall': recall, 'list_precision': precision, 'list_f1': f1})

        yesno_F1yes = 2 * yesno_yes_tp / (2 * yesno_yes_tp + yesno_yes_fp + yesno_yes_fn) if (2 * yesno_yes_tp + yesno_yes_fp + yesno_yes_fn) > 0 else 0
        yesno_F1no = 2 * yesno_no_tp / (2 * yesno_no_tp + yesno_no_fp + yesno_no_fn) if (2 * yesno_no_tp + yesno_no_fp + yesno_no_fn) > 0 else 0
        yesno_maF1 = (yesno_F1yes + yesno_F1no) / 2

        out_metrics = {
            "yesno_acc": yesno_correct / yesno_total if yesno_total > 0 else 0,
            "yesno_F1yes": yesno_F1yes, 
            "yesno_F1no": yesno_F1no, 
            "yesno_maF1": yesno_maF1,
            "factoid_Sacc": factoid_Scorrect / factoid_total if factoid_total > 0 else 0,
            "factoid_Lacc": factoid_Lcorrect / factoid_total if factoid_total > 0 else 0,
            "factoid_MRR": factoid_Lcorrect / factoid_total if factoid_total > 0 else 0, # approximation of MRR since we cannot systematically interpret a ranked output from LLMs so we perform MRR with top-1 result only which is the LLM output
            "list_precision": list_precision / list_total if list_total > 0 else 0,
            "list_recall": list_recall / list_total if list_total > 0 else 0,
            "list_f1": list_f1 / list_total if list_total > 0 else 0
            }
        return out_metrics, metrics_per_question
    