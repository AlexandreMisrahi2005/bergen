from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import torch
import gc
import re
import numpy as np
from hydra.utils import instantiate
from torch.distributions import Categorical
from difflib import SequenceMatcher
from collections import defaultdict

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from models.generators.DiffTransformer.llama_lora_diff_transformer_model import LlamaLoraDiffTransformerForCausalLM

def plot_combined_attention(
    attentions, input_ids, prompt_len, tokenizer,
    magic_start, magic_end, query_start, query_end,
    save_path="figs/combined_attention_map.html"
):
    """
    Plots a single bar chart for combined normalized attention weights.

    Parameters:
    - attentions: Full attention matrix (no positive/negative decomposition).
    - input_ids: Token IDs.
    - prompt_len: Length of the prompt tokens.
    - tokenizer: Tokenizer for decoding.
    - magic_start, magic_end, query_start, query_end: Grouping positions.
    - save_path: Path to save the plot.
    """
    # Slice the attention matrix
    sliced_combined_attentions = attentions[prompt_len:, :prompt_len].cpu().float().numpy()

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    prompt_tokens = tokens[:prompt_len]
    generated_tokens = tokens[prompt_len:]

    # --- Individual Token Attention ---
    # DataFrame for individual attention
    individual_combined_df = pd.DataFrame(
        sliced_combined_attentions,
        index=generated_tokens,
        columns=prompt_tokens
    )
    individual_avg_combined_attention = individual_combined_df.mean(axis=0)

    # --- Grouped Token Attention ---
    def compute_grouped_attention(sliced_attention):
        """Helper to compute grouped attention scores."""
        bos_attention = sliced_attention[:, :1].sum(axis=1)
        context_1_attention = sliced_attention[:, 1:magic_start].sum(axis=1)
        magic_attention = sliced_attention[:, magic_start:magic_end].sum(axis=1)
        context_2_attention = sliced_attention[:, magic_end:query_start].sum(axis=1)
        query_attention = sliced_attention[:, query_start:].sum(axis=1)
        return np.stack([bos_attention, context_1_attention, magic_attention, context_2_attention, query_attention], axis=1)

    grouped_combined_attentions = compute_grouped_attention(sliced_combined_attentions)

    grouped_tokens = ["BOS", "CONTEXT 1", "MAGIC NUMBER", "CONTEXT 2", "QUERY"]

    # DataFrame for grouped attention
    grouped_combined_df = pd.DataFrame(grouped_combined_attentions, index=generated_tokens, columns=grouped_tokens)
    grouped_avg_combined_attention = grouped_combined_df.mean(axis=0)

    # --- Create Subplots ---
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Normalized Attention for Individual Tokens",
            "Normalized Attention for Grouped Tokens"
        ],
        vertical_spacing=0.3
    )

    # Add Plot 1: Individual Token Attention
    fig.add_trace(
        go.Bar(
            x=prompt_tokens,
            y=individual_avg_combined_attention.values,
            marker=dict(color="blue"),
            name="Individual Attention"
        ),
        row=1, col=1
    )

    # Add Plot 2: Grouped Token Attention
    fig.add_trace(
        go.Bar(
            x=grouped_tokens,
            y=grouped_avg_combined_attention.values,
            marker=dict(color="purple"),
            name="Grouped Attention"
        ),
        row=2, col=1
    )

    # Annotate total combined attention values on the grouped plot
    total_combined_attention = grouped_avg_combined_attention.sum()
    fig.add_annotation(
        x=0.5, y=0.3,
        xref="paper", yref="paper",
        text=f"Total Combined Attention: {total_combined_attention:.4f}",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="center",
        bgcolor="lightyellow",
        bordercolor="black"
    )

    # Update layout
    fig.update_layout(
        height=1000,
        title="Attention Analysis: Detailed vs Grouped Tokens",
        xaxis=dict(title="Prompt Tokens (Detailed)", tickangle=45),
        xaxis2=dict(title="Prompt Tokens (Grouped)", tickangle=45),
        yaxis=dict(title="Normalized Attention"),
        yaxis2=dict(title="Normalized Attention"),
        font=dict(size=10)
    )

    # Save the figure as an HTML file
    # fig.write_html(save_path)
    fig.write_image(save_path, format='png')
    print(f"Combined attention analysis saved to {save_path}")

    return fig

def plot_pos_neg_attention_with_and_without_groups(
    attentions, input_ids, prompt_len, tokenizer,
    magic_start, magic_end, query_start, query_end,
    pos_attentions=None, neg_attentions=None,
    save_path="figs/normalized_attention_maps.html"
):
    """
    Plots three bar charts for normalized attention weights:
    1. Individual token attention with positive/negative decomposition.
    2. Grouped token attention with positive/negative decomposition.
    3. Combined attention for grouped tokens.
    """

    if pos_attentions is None and neg_attentions is None:
        return plot_combined_attention(
            attentions, input_ids, prompt_len, tokenizer,
            magic_start, magic_end, query_start, query_end,
            save_path=save_path
        )
    else:
        # Slice the attention matrix
        sliced_pos_attentions = pos_attentions[prompt_len:, :prompt_len].cpu().float().numpy()
        sliced_neg_attentions = neg_attentions[prompt_len:, :prompt_len].cpu().float().numpy()
        sliced_combined_attentions = attentions[prompt_len:, :prompt_len].cpu().float().numpy()

        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        prompt_tokens = tokens[:prompt_len]
        generated_tokens = tokens[prompt_len:]

        # --- Individual Token Attention ---
        individual_pos_df = pd.DataFrame(sliced_pos_attentions, index=generated_tokens, columns=prompt_tokens)
        individual_neg_df = pd.DataFrame(sliced_neg_attentions, index=generated_tokens, columns=prompt_tokens)
        individual_combined_df = pd.DataFrame(sliced_combined_attentions, index=generated_tokens, columns=prompt_tokens)

        individual_avg_pos_attention = individual_pos_df.mean(axis=0)
        individual_avg_neg_attention = individual_neg_df.mean(axis=0)
        individual_avg_combined_attention = individual_combined_df.mean(axis=0)

        # --- Grouped Token Attention ---
        def compute_grouped_attention(sliced_attention):
            """Helper to compute grouped attention scores."""
            bos_attention = sliced_attention[:, :1].sum(axis=1)
            context_1_attention = sliced_attention[:, 1:magic_start].sum(axis=1)
            magic_attention = sliced_attention[:, magic_start:magic_end].sum(axis=1)
            context_2_attention = sliced_attention[:, magic_end:query_start].sum(axis=1)
            query_attention = sliced_attention[:, query_start:].sum(axis=1)
            return np.stack([bos_attention, context_1_attention, magic_attention, context_2_attention, query_attention], axis=1)

        grouped_pos_attentions = compute_grouped_attention(sliced_pos_attentions)
        grouped_neg_attentions = compute_grouped_attention(sliced_neg_attentions)
        grouped_combined_attentions = compute_grouped_attention(sliced_combined_attentions)

        grouped_tokens = ["BOS", "CONTEXT 1", "MAGIC NUMBER", "CONTEXT 2", "QUERY"]

        grouped_pos_df = pd.DataFrame(grouped_pos_attentions, index=generated_tokens, columns=grouped_tokens)
        grouped_neg_df = pd.DataFrame(grouped_neg_attentions, index=generated_tokens, columns=grouped_tokens)
        grouped_combined_df = pd.DataFrame(grouped_combined_attentions, index=generated_tokens, columns=grouped_tokens)

        grouped_avg_pos_attention = grouped_pos_df.mean(axis=0)
        grouped_avg_neg_attention = grouped_neg_df.mean(axis=0)
        grouped_avg_combined_attention = grouped_combined_df.mean(axis=0)

        # --- Create Subplots ---
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                "Normalized Attention for Individual Tokens (Positive/Negative)",
                "Normalized Attention for Grouped Tokens (Positive/Negative)",
                "Normalized Combined Attention for Grouped Tokens"
            ],
            vertical_spacing=0.2,
        )

        # Plot 1: Individual Token Attention
        fig.add_trace(
            go.Bar(
                x=prompt_tokens,
                y=individual_avg_pos_attention.values,
                marker=dict(color="blue"),
                name="Positive Attention (Individual)"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                x=prompt_tokens,
                y=-individual_avg_neg_attention.values,
                marker=dict(color="red"),
                name="Negative Attention (Individual)"
            ),
            row=1, col=1
        )

        # Plot 2: Grouped Token Attention
        fig.add_trace(
            go.Bar(
                x=grouped_tokens,
                y=grouped_avg_pos_attention.values,
                marker=dict(color="green"),
                name="Positive Attention (Grouped)"
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(
                x=grouped_tokens,
                y=-grouped_avg_neg_attention.values,
                marker=dict(color="orange"),
                name="Negative Attention (Grouped)"
            ),
            row=2, col=1
        )

        # Plot 3: Combined Attention for Grouped Tokens
        fig.add_trace(
            go.Bar(
                x=grouped_tokens,
                y=grouped_avg_combined_attention.values,
                marker=dict(color="purple"),
                name="Combined Attention (Grouped)"
            ),
            row=3, col=1
        )

        # Annotate total attention values (on the third plot)
        total_combined_attention = grouped_avg_combined_attention.sum()
        fig.add_annotation(
            x=0.5, y=0.2,
            xref="paper", yref="paper",
            text=f"Total Positive Attention: {grouped_avg_pos_attention.sum():.4f}",
            showarrow=False,
            font=dict(size=12, color="black"),
            align="center",
            bgcolor="lightyellow",
            bordercolor="black"
        )
        fig.add_annotation(
            x=0.5, y=0.15,
            xref="paper", yref="paper",
            text=f"Total Negative Attention: {grouped_avg_neg_attention.sum():.4f}",
            showarrow=False,
            font=dict(size=12, color="black"),
            align="center",
            bgcolor="lightyellow",
            bordercolor="black"
        )

        # Update layout
        fig.update_layout(
            height=1200,
            title="Attention Analysis: Positive, Negative, and Combined (Grouped Tokens)",
            barmode="relative",  # Relative stacking for positive and negative
            xaxis=dict(title="Prompt Tokens (Individual)", tickangle=45),
            xaxis2=dict(title="Prompt Tokens (Grouped)", tickangle=45),
            xaxis3=dict(title="Prompt Tokens (Grouped)", tickangle=45),
            yaxis=dict(title="Normalized Attention"),
            yaxis2=dict(title="Normalized Attention"),
            yaxis3=dict(title="Normalized Attention"),
            font=dict(size=10)
        )

        # Save the figure as an HTML file
        # fig.write_html(save_path)
        fig.write_image(save_path, format='png')
        # print(f"Attention analysis saved to {save_path}")

        return fig


class LLM_att():
    def __init__(self, generator_config, prompt):
        # generator_config['init_args']['attn_implementation'] = 'sdpa'
        self.llm = instantiate(generator_config['init_args'], prompt=prompt)
        #breakpoint()
        #self.llm = Generate(**generator_config, prompt=prompt, flash_att=False) if generator_config != None else None   

    def collate_fn(self, sample):
        # detect if the sample is a needle in haystack test instance, match the phrase "(The magic number is xx)"
        nih = False
        nih_res = re.search(r" \(The magic number is (.*?)\) ", sample['instruction'])
        # print("nih_res", nih_res, nih_res.group(0))
        if nih_res:
            # print("nih detected")
            # extract whole phrase from text and tokenize it
            magic_phrase = nih_res.group(0)
            # tokenize it to know exactly the corresponding sequence of tokens
            magic_tokenized = self.llm.tokenizer([magic_phrase], is_split_into_words=True, add_special_tokens=False, return_tensors="pt")
            if self.llm.tokenizer.decode(magic_tokenized['input_ids'][0][-1] == ' '):  # very edge case, if we tokenize the phrase only (with the extra space at the end which is needed to get proper tokenization), then we might get an extra space token that doesn't match the entire instruction tokenization (usually the space is part of the next token but not standalone)
                magic_tokenized['input_ids'] = magic_tokenized['input_ids'][:,:-1]
            nih = True
        #decompotes prompt into different subparts, and keep trace of subparts position (to quantify attention at these subparts)
        instr_subset = {}
        for prompt_el in ['system', 'context', 'system_without_docs', "user", 'user_without_docs', 'language_instruction']:
            #prompt_val = str(self.llm.model.prompt._content[prompt_el])
            try:
                prompt_val = str(self.llm.prompt[prompt_el])
            except:                    
                continue
            prompt_pos = sample['instruction'].find(prompt_val)
            prompt_el = prompt_el.replace("user", "context").replace("_without_docs", "").replace('language_instruction', 'system')
            if prompt_pos > 0:
                instr_subset[(prompt_pos, prompt_pos+len(prompt_val))] = prompt_el
            else:
                match = SequenceMatcher(None, sample['instruction'], prompt_val).find_longest_match()    
                if match.b <3:                
                    instr_subset[(match.a, -1)] = prompt_el
        instr_subset[(0,1)] = "bos"        
        qpos = sample['instruction'].find(sample['question'])
        if qpos>0:
            instr_subset[(qpos, qpos+len(sample['question']))] = 'question'
        substrings = []
        substrings_types = []
        start_pos = 0
        i = 0
        curr_label = 'bos'
        # print("instr_subset", instr_subset)
        for pos in sorted(instr_subset):
            if pos[0]>=start_pos:
                substrings.append(sample['instruction'][start_pos:pos[0]])
                substrings_types.append(curr_label)
            if pos[1]>0:
                substrings.append(sample['instruction'][pos[0]:pos[1]])   
                substrings_types.append(instr_subset[pos])   
                start_pos = pos[1]
                curr_label = 'other'
            else:
                start_pos = pos[0]
                curr_label  = instr_subset[pos]
        
        substrings.append(sample['instruction'][start_pos:-1])
        substrings_types.append(curr_label)
        
        tokenized = self.llm.tokenizer(substrings+[sample['candidate']], is_split_into_words=True, add_special_tokens=False, return_tensors="pt")
        # print("tokens: ", [self.llm.tokenizer.decode(i) for i in tokenized['input_ids'][0]])

        #tokenized_tensor = self.llm.tokenizer(substrings+[sample['candidate']], add_special_tokens=False, )
        prompt_tokenized = self.llm.tokenizer(substrings, is_split_into_words=True, add_special_tokens=False, return_tensors="pt")
        #breakpoint()
        prompt_len = prompt_tokenized.input_ids.size(1)
        if nih:
            # print("finding nih positions")
            # find start end positions of magic_tokenized in the prompt
            magic_start = -1
            magic_end = -1
            # print("prompt_len", prompt_len)
            # print("magic_tokenized.input_ids.size(1)", magic_tokenized.input_ids.size(1))
            # print("magic_tokenized input ids: ", magic_tokenized['input_ids'])
            # print("magic_tokenized tokens: ", [self.llm.tokenizer.decode(i) for i in magic_tokenized['input_ids'][0]])
            # print("magic phrase: ", magic_phrase)
            # print("prompt_tokenized input ids: ", prompt_tokenized['input_ids'])
            # print("prompt_tokenized tokens: ", [self.llm.tokenizer.decode(i) for i in prompt_tokenized['input_ids'][0]])
            for i in range(prompt_len - magic_tokenized.input_ids.size(1)):

                if torch.all(magic_tokenized['input_ids'] == prompt_tokenized['input_ids'][:, i:i+magic_tokenized.input_ids.size(1)]):
                    magic_start = i
                    magic_end = i+magic_tokenized.input_ids.size(1)
                    # print("found magic tokenized start/end", magic_start, magic_end)
                    break
            # find start position of query
            # tokenize sample['question']
            question_start = -1
            question_end = -1
            question_tokenized = self.llm.tokenizer([sample['question']], is_split_into_words=True, add_special_tokens=False, return_tensors="pt")
            for i in range(prompt_len - question_tokenized.input_ids.size(1)):
                if torch.all(question_tokenized['input_ids'] == prompt_tokenized['input_ids'][:, i:i+question_tokenized.input_ids.size(1)]):
                    question_start = i
                    question_end = i+question_tokenized.input_ids.size(1)
                    # print("found question tokenized start/end", question_start, question_end)
                    break
        out_nih = (magic_start, magic_end, question_start, question_end) if nih and question_start > 0 and magic_start > 0 else None
        return tokenized, prompt_len, substrings_types, out_nih
    
    @torch.no_grad()
    def __call__(self, predictions, references, questions, instructions):
        def get_att_var(attention):
            # equation 2 from https://arxiv.org/pdf/2205.10828.pdf
            gen_len, prefix_len = attention.shape
            prefix_pos = torch.arange(0, prefix_len).to('cuda')
            mu = torch.matmul(prefix_pos.float(), attention.transpose(1,0).float())
            x = [torch.matmul(attention[i,:].float(), (mu[i]-prefix_pos)**2) for i in range(gen_len)]
            var = torch.mean(torch.stack(x))
            #for i in range(0, gen_len):
            #    var += torch.sum(attention[:,i]*(mu[i]-prefix_pos)**2)
            return var
        def get_att_entropy(attention):       
            """
            reflects how  peaky is the attention over prefix tokens, averaged over all generation tokens 
            higher values --> attention is more concentrated, lower values --> attention is more distributede
            https://aclanthology.org/I17-1004.pdf
            """     
            gen_len, prefix_len = attention.shape
            #need to normalize att over prefix to make it proper distribution
            attn_norm = attention/torch.sum(attention, axis=1, keepdim=True)
            entr_per_token = torch.sum(- attn_norm*torch.log(attn_norm), axis=1)
            return torch.mean(entr_per_token)

        def get_att_confidence(attention):  
            """
            maximum attention on the previx token averaged across all generated tokens 
            (could be lower for longer generations)
            """
            gen_len, prefix_len = attention.shape
            conf = torch.mean(torch.max(attention, axis=1)[0])

            return conf

        def get_att_coverage_conf(attention):  
            """
            maximum attention on the prefix token summed across all generated tokens 
            (should not penalize longer generations)
            """
            gen_len, prefix_len = attention.shape
            conf = torch.sum(torch.max(attention, axis=1)[0])

            return conf
        def get_att_coverage1(attention):  
            """
            |I| - prefix tokens length, |J| - generation length
            coverage = \avg_{j \in J} (\sum_{i \in I} \alpha_{ij})^2
            - reflects how much attention each generated token puts on the prefix, averaged across all generated tokens 
            taken from  https://arxiv.org/pdf/2105.14940
            """
        
            gen_len, prefix_len = attention.shape
            cov = torch.mean(torch.sum(attention, axis=1)**2)
            return cov

        def get_att_coverage(attention):  
            """
            |I| - prefix tokens length, |J| - generation lenght
            coverage = \sum_{i \in I} (\sum_{j \in J} \alpha_{ij})^2
            - reflects how much attention each token in the prefix has recieved from generated tokens
            summed over all prefix tokens --> overall coverage
            """
        
            gen_len, prefix_len = attention.shape
            cov = torch.sum(torch.sum(attention, axis=0)**2)
            return cov

        assert len(predictions) == len(references) == len(questions) == len(instructions)
        examples = [{'question': questions[i], 'candidate': predictions[i], 'instruction': instructions[i]}  for i in range(len(predictions))]
        batch_size = self.llm.batch_size
        batch_size = 1
        layer = -1
        scores = []
        samples=1000
        if samples == -1:
            samples = len(examples)
        for j in tqdm(range(0, min(len(examples), samples)), desc=' Compute attention-based metrics'):
        #for i in tqdm(range(0, len(examples)), desc=' Compute attention-based metrics'):
            # Extract batch
            batch_inputs, prompt_len, input_types, nih_output  = self.collate_fn(examples[j])

            batch_input_ids = batch_inputs['input_ids'].to('cuda')
            batch_attention_masks = batch_inputs['attention_mask'].to('cuda')

            output = self.llm.model.generate(input_ids=batch_input_ids, attention_mask=batch_attention_masks, max_new_tokens=1, do_sample=False, output_attentions=True, output_hidden_states=True, return_dict_in_generate=True, temperature=None, top_p=None)
            #hidden states: (batch_size, layers, [bsize, seq_len, hidden_size])
            sequences = output['sequences']
            generated_ids = sequences[:, prompt_len:-1]
            #decoded = self.llm.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            att_by_cat = defaultdict(float)
            layers = list(range(1, self.llm.model.model.config.num_hidden_layers, 3)) + [-1]

            if nih_output is not None:
                (start_nih, end_nih, query_start, query_end) = nih_output
                attentions = output['attentions'][0][-1][0] # take attention from last layer
                #avg across heads
                attentions = torch.mean(attentions, axis=0).squeeze(axis=0)# for att in full_attentions[0]
                pos_attentions, neg_attentions = None, None
                if isinstance(self.llm.model, LlamaLoraDiffTransformerForCausalLM) and j == 0:
                    # split attentions into pos and neg attentions
                    attentions, pos_attentions, neg_attentions = attentions[0], attentions[1], attentions[2]
                
                # hidden_states = output['hidden_states'][0][layer-1][0]
                if layer == -1:
                    layer = "last"
                # hidden_states_norm = torch.norm(hidden_states, dim=1)
                # prompt_hidden_states = hidden_states_norm[:prompt_len]
                #gen_hidden_states = hidden_states_norm[prompt_len:]
                #prompt_to_gen_att_mh = full_attentions[:, prompt_len:, :prompt_len]
                prompt_to_gen_att = attentions[prompt_len:, :prompt_len]

                prompt_to_gen_magic = attentions[prompt_len:, start_nih:end_nih]
                att_by_cat["att_last_magic"] = torch.mean(torch.sum(prompt_to_gen_magic, axis=1)).float().to('cpu').numpy().item()

                assert batch_input_ids.shape[0] == 1
                if j == 0:
                    plot_pos_neg_attention_with_and_without_groups(attentions, batch_input_ids.squeeze(), prompt_len, self.llm.tokenizer, start_nih, end_nih, query_start, query_end, pos_attentions=pos_attentions, neg_attentions=neg_attentions, save_path=f"figs/split_grouped_normalized_attention_maps/{self.llm.model_name.replace('experiments/','').replace('/', '_')}_sample{j}.png")
            elif not nih_output:
                print(f"WARNING: NIH output is None for example {j} \nreference={references[j]} \ninstruction={instructions[j]} ")

            for layer in layers:
                attentions = output['attentions'][0][layer][0]  
                if isinstance(self.llm.model, LlamaLoraDiffTransformerForCausalLM):
                    attentions = attentions[0]                                      
                #avg across heads
                attentions = torch.mean(attentions, axis=0).squeeze(axis=0)# for att in full_attentions[0]
                hidden_states = output['hidden_states'][0][layer-1][0]
                if layer == -1:
                    layer = "last"                
                hidden_states_norm = torch.norm(hidden_states, dim=1)
                prompt_hidden_states = hidden_states_norm[:prompt_len]
                #gen_hidden_states = hidden_states_norm[prompt_len:]
                #prompt_to_gen_att_mh = full_attentions[:, prompt_len:, :prompt_len]
                prompt_to_gen_att = attentions[prompt_len:, :prompt_len]

                #att_by_cat['att_prefix'] = torch.mean(torch.sum(prompt_to_gen_att, axis=1)).float().to('cpu').numpy()
                for i, cat in enumerate(input_types):
                    if not i+1 in batch_inputs.word_ids():
                        continue  
                    span = batch_inputs.word_to_tokens(i+1)
                    if span.start >= prompt_len:
                        continue
                    att_by_cat[f"att_{layer}_{cat}"] += torch.mean(torch.sum(prompt_to_gen_att[:, span.start:span.end], axis=1)).float().to('cpu').numpy()
                    att_by_cat[f"att_{layer}_{cat}_norm"] += torch.mean(torch.matmul(prompt_hidden_states[span.start:span.end], prompt_to_gen_att[:, span.start:span.end].t())).float().to('cpu').numpy()
                    att_by_cat[f"cov_{layer}_{cat}"] += get_att_coverage(prompt_to_gen_att[:, span.start:span.end]).float().to('cpu').numpy()
                    att_by_cat[f"cov1_{layer}_{cat}"] += get_att_coverage1(prompt_to_gen_att[:, span.start:span.end]).float().to('cpu').numpy()                    
                    att_by_cat[f"entropy_{layer}_{cat}"] += get_att_entropy(prompt_to_gen_att[:, span.start:span.end]).float().to('cpu').numpy()
                    att_by_cat[f"conf_{layer}_{cat}"] += get_att_confidence(prompt_to_gen_att[:, span.start:span.end]).float().to('cpu').numpy()
                    att_by_cat[f"conf1_{layer}_{cat}"] += get_att_coverage_conf(prompt_to_gen_att[:, span.start:span.end]).float().to('cpu').numpy()
                entropy= get_att_entropy(prompt_to_gen_att).float().to('cpu').numpy()
                entropy_nobos= get_att_entropy(prompt_to_gen_att[:, 1:]).float().to('cpu').numpy()
                att_by_cat[f"att_{layer}_prefix"] = np.sum([att_by_cat[x] for x in att_by_cat if not "norm" in x and f'{layer}' in x and 'att' in x])
                att_by_cat[f"att_{layer}_prefix_norm"] = np.sum([att_by_cat[x] for x in att_by_cat if "norm" in x and f'{layer}' in x and 'att' in x])
 
                #att_by_cat[f'var_{layer}'] += get_att_var(prompt_to_gen_att).float().to('cpu').numpy()
                #att_by_cat[f'var_{layer}_nobos'] += get_att_var(prompt_to_gen_att[:, 1:]).float().to('cpu').numpy()
                att_by_cat[f'entropy_{layer}'] += entropy     
                att_by_cat[f'entropy_{layer}_nobos'] += entropy_nobos
                att_by_cat[f'coverage_{layer}'] += get_att_coverage(prompt_to_gen_att).float().to('cpu').numpy()
                att_by_cat[f'coverage_{layer}_nobos'] += get_att_coverage(prompt_to_gen_att[:, 1:]).float().to('cpu').numpy()
                att_by_cat[f'conf_{layer}'] += get_att_confidence(prompt_to_gen_att).float().to('cpu').numpy()
                att_by_cat[f'conf_{layer}_nobos'] += get_att_confidence(prompt_to_gen_att[:, 1:]).float().to('cpu').numpy()
                att_by_cat[f'conf1_{layer}'] += get_att_coverage_conf(prompt_to_gen_att).float().to('cpu').numpy()
                att_by_cat[f'conf1_{layer}_nobos'] += get_att_coverage_conf(prompt_to_gen_att[:, 1:]).float().to('cpu').numpy()
                
                del attentions
                del hidden_states
                del prompt_hidden_states
                del prompt_to_gen_att
                torch.cuda.empty_cache()
                gc.collect()                    
            del output
            del batch_attention_masks
            del batch_input_ids
            del batch_inputs
            torch.cuda.empty_cache()
            gc.collect()
            scores.append(att_by_cat)    
            
        torch.cuda.empty_cache()
        d = {cat: np.mean([score[cat] for score in scores]) for cat in att_by_cat.keys()}
        return d, scores

