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

def plot_attention_map_with_bars(attentions, input_ids, prompt_len, tokenizer, save_path="figs/attention_map.html"):
    """
    Plots an attention map and a bar chart for average attention weights using Plotly.

    Parameters:
    - attentions (torch.Tensor): The attention matrix of shape (seq_len, seq_len).
    - input_ids (list[int]): List of token IDs for the input sequence.
    - prompt_len (int): Length of the prompt tokens.
    - tokenizer: The tokenizer to decode token IDs into strings.
    - save_path (str): Path to save the generated plot (default: "attention_map_with_bars.html").
    """
    # Slice the attention matrix: attentions from generated tokens to prompt tokens
    sliced_attentions = attentions[prompt_len:, :prompt_len].cpu().float().numpy()

    # Decode token IDs into strings
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    prompt_tokens = tokens[:prompt_len]
    generated_tokens = tokens[prompt_len:]
    print("prompt tokens: ", prompt_tokens)
    print("generated tokens: ", generated_tokens)

    # Convert data into a DataFrame for Plotly
    df = pd.DataFrame(
        sliced_attentions,
        index=[t for t in generated_tokens],  # Add "Gen" prefix for clarity
        columns=[t for t in prompt_tokens]  # Add "Prompt" prefix for clarity
    )
    print(f"Sliced attentions shape: {sliced_attentions.shape}")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print(f"Generated tokens: {len(generated_tokens)}")
    print(df.head())

    # Compute average attention over generated tokens
    avg_attention = df.mean(axis=0)  # Average along generated tokens axis

    # Create a subplot layout
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Attention Map: Generated Tokens → Prompt Tokens",
            "Attn map, Averaged over the generated tokens"
        ),
        vertical_spacing=0.3
    )

    # Add the heatmap (attention map)
    fig.add_trace(
        go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale="Viridis",
            colorbar=dict(title="Attention Weight")
        ),
        row=1, col=1
    )

    # Add the bar chart (average attention weights)
    fig.add_trace(
        go.Bar(
            x=df.columns,
            y=avg_attention.values,
            marker=dict(color="blue")
        ),
        row=2, col=1
    )

    # Update layout for the figure
    fig.update_layout(
        height=800,  # Adjust the height of the figure
        title="Attention Map and Average Attention Weights",
        xaxis=dict(title="Prompt Tokens", tickangle=45),
        xaxis2=dict(title="Prompt Tokens", tickangle=45),  # Separate x-axis for the bar chart
        yaxis=dict(title="Generated Tokens"),
        yaxis2=dict(title="Average Attention Weight"),
        font=dict(size=10)
    )

    # Save the figure as an HTML file
    fig.write_html(save_path)
    print(f"Attention map and bar chart saved to {save_path}")

    return fig



def plot_attention_map(attentions, input_ids, prompt_len, tokenizer, save_path="figs/attention_map.html"):
    """
    Plots an attention map highlighting attention from generated tokens to prompt tokens.
    
    Parameters:
    - attentions (torch.Tensor): The attention matrix of shape (seq_len, seq_len).
    - input_ids (list[int]): List of token IDs for the input sequence.
    - prompt_len (int): Length of the prompt tokens.
    - tokenizer: The tokenizer to decode token IDs into strings.
    """
    # Slice the attention matrix: attentions from generated tokens to prompt tokens
    sliced_attentions = attentions[prompt_len:, :prompt_len].cpu().float().numpy()

    # Decode token IDs into strings
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    prompt_tokens = tokens[:prompt_len]
    generated_tokens = tokens[prompt_len:]
    print("generated tokens: ", generated_tokens)

    # Convert data into a DataFrame for Plotly
    df = pd.DataFrame(
        sliced_attentions,
        index=[t for t in generated_tokens],  # Add "Gen" prefix for clarity
        columns=[t for t in prompt_tokens]  # Add "Prompt" prefix for clarity
    )

    # Create a Plotly heatmap
    fig = px.imshow(
        df,
        labels=dict(x="Prompt Tokens", y="Generated Tokens", color="Attention Weight"),
        title="Attention Map: Generated Tokens → Prompt Tokens",
        color_continuous_scale="Viridis"
    )

    # Update layout for better visualization
    fig.update_layout(
        xaxis=dict(tickangle=45),  # Rotate x-axis labels
        font=dict(size=10),       # Adjust font size
        title=dict(font_size=16)  # Title font size
    )

    # Save the figure as an HTML file
    fig.write_html(save_path)
    print(f"Attention map saved to {save_path}")

    return fig

class LLM_att():
    def __init__(self, generator_config, prompt):
        generator_config['init_args']['attn_implementation'] = 'sdpa'
        self.llm = instantiate(generator_config['init_args'], prompt=prompt)
        #breakpoint()
        #self.llm = Generate(**generator_config, prompt=prompt, flash_att=False) if generator_config != None else None   

    def collate_fn(self, sample):
        # detect if the sample is a needle in haystack test instance, match the phrase "(The magic number is xx)"
        nih = False
        nih_res = re.search(r" \(The magic number is (\d+)\)", sample['instruction'])
        if nih_res:
            # extract whole phrase from text and tokenize it
            magic_phrase = nih_res.group(0)
            # tokenize it to know exactly the corresponding sequence of tokens
            magic_tokenized = self.llm.tokenizer([magic_phrase], is_split_into_words=True, add_special_tokens=False, return_tensors="pt")
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
            # find start end positions of magic_tokenized in the prompt
            magic_start = -1
            magic_end = -1
            for i in range(prompt_len):
                if torch.all(magic_tokenized['input_ids'] == prompt_tokenized['input_ids'][:, i:i+magic_tokenized.input_ids.size(1)]):
                    magic_start = i
                    magic_end = i+magic_tokenized.input_ids.size(1)
                    print("found magic tokenized start/end", magic_start, magic_end)
                    break
            # find start position of query
            # tokenize sample['question']
            question_start = -1
            question_end = -1
            question_tokenized = self.llm.tokenizer([sample['question']], is_split_into_words=True, add_special_tokens=False, return_tensors="pt")
            for i in range(prompt_len):
                if torch.all(question_tokenized['input_ids'] == prompt_tokenized['input_ids'][:, i:i+question_tokenized.input_ids.size(1)]):
                    question_start = i
                    question_end = i+question_tokenized.input_ids.size(1)
                    print("found question tokenized start/end", question_start, question_end)
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
        #i=0
        #batch_input_ids = instr_tokenized['input_ids'][i:i+self.llm.batch_size].to('cuda')
        #batch_attention_masks = instr_tokenized['attention_mask'][i:i+self.llm.batch_size].to('cuda')
        #instr_batch = instrs[i:i+self.llm.batch_size]
        #gen = self.llm.model.model.generate(input_ids=batch_input_ids, attention_mask=batch_attention_masks, do_sample=False, output_attentions=True, return_dict_in_generate=True)

        """
        def compute_att_variability(attentions):
            x = torch.div(torch.transpose(atts, 1,0), torch.sum(atts, axis=1))            
            ent 
        #output_ids = self.model.generate(**instr_tokenized.to('cuda'), max_new_tokens=self.max_new_tokens, max_length = self.max_length, do_sample=False)
        output = self.model.generate(**instr_tokenized.to('cuda'), max_new_tokens=self.max_new_tokens, max_length = self.max_length, do_sample=False,output_attentions=True, return_dict_in_generate=True)
        output_ids = output['sequences']
        full_attentions = output['attentions']
        prompt_toks = [self.tokenizer.decode(i) for  i in instr_tokenized['input_ids'][0]]        
        prompt_len = instr_tokenized.input_ids.size(1)     
        generated_ids = output_ids[:, prompt_len:]
        gen_toks  = [self.tokenizer.decode(i) for  i in generated_ids[0]]
        
        layer=-1
        nb_layers = len(full_attentions[0])
        #avg across heads 
        step = 3
        fig, axs = plt.subplots(1, int(nb_layers/step)+1, figsize=(50,50))            
        i = 0

        for layer in range(0, nb_layers, step):
            attentions = [torch.mean(att[layer], axis=1).squeeze(axis=1) for att in full_attentions]     
            #TODO : renormalize attention on prompt (will not sum to 1 for more than prompt_len tokens)

            prompt_to_gen_att = [attentions[j+1][0, :prompt_len] for j in range(generated_ids.shape[-1]-1)]
            atts = torch.stack(prompt_to_gen_att).float().cpu().numpy()
            df = pd.DataFrame(atts.transpose(), columns=gen_toks[1:], index=prompt_toks)
            sns.heatmap(df, ax=axs[i], cmap="crest")
            axs[i].set_title(f"Layer {layer}")
            i = i+1
        
        plt.savefig(f'plot_{self.model_name.split("/")[-1].replace("-","_")}.png')
        plt.clf()
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        """
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
            # print(examples[j])
            # print("batch_inputs", batch_inputs)
            # print("prompt_len", prompt_len)
            # print("input_types", input_types)
            #breakpoint()
            batch_input_ids = batch_inputs['input_ids'].to('cuda')
            batch_attention_masks = batch_inputs['attention_mask'].to('cuda')
            #breakpoint()
            output = self.llm.model.generate(input_ids=batch_input_ids, attention_mask=batch_attention_masks, max_new_tokens=1, do_sample=False, output_attentions=True, output_hidden_states=True, return_dict_in_generate=True)
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
                hidden_states = output['hidden_states'][0][layer-1][0]
                if layer == -1:
                    layer = "last"
                hidden_states_norm = torch.norm(hidden_states, dim=1)
                prompt_hidden_states = hidden_states_norm[:prompt_len]
                #gen_hidden_states = hidden_states_norm[prompt_len:]
                #prompt_to_gen_att_mh = full_attentions[:, prompt_len:, :prompt_len]
                # check the attentions matrix is symmetric
                # print("attentions is symmetric? ", torch.allclose(attentions, attentions.T, atol=1e-8))
                # print("attentions is upper triangular? ", torch.allclose(attentions, torch.triu(attentions), atol=1e-8))
                # print("attentions is lower triangular? ", torch.allclose(attentions, torch.tril(attentions), atol=1e-8))
                prompt_to_gen_att = attentions[prompt_len:, :prompt_len]
                # print("prompt_to_gen_att.sum(axis=1)", prompt_to_gen_att.sum(axis=1))
                # print("prompt_to_gen_att.shape", prompt_to_gen_att.shape)
                # compute attention from start_nih to end_nih tokens
                # print("start_nih", start_nih)
                # print("end_nih", end_nih)
                # print("query_start", query_start)
                # print("query_end", query_end)
                # print("prompt_len", prompt_len)
                # compute attention to bos token
                # prompt_to_gen_att_bos = attentions[prompt_len:, :1]
                # # compute attention to instruction
                # prompt_to_gen_instr = torch.cat([attentions[prompt_len:, 1:start_nih], attentions[prompt_len:, end_nih:query_start]], dim=1)
                # # compute attention on magic phrase
                prompt_to_gen_magic = attentions[prompt_len:, start_nih:end_nih]
                att_by_cat["att_last_magic"] = torch.mean(torch.sum(prompt_to_gen_magic, axis=1)).float().to('cpu').numpy()
                # # compute attention to query
                # prompt_to_gen_query = attentions[prompt_len:, query_start:query_end]
                # print shapes 
                # print("prompt_to_gen_att_bos.shape", prompt_to_gen_att_bos.shape)
                # print("prompt_to_gen_instr.shape", prompt_to_gen_instr.shape)
                # print("prompt_to_gen_query.shape", prompt_to_gen_query.shape)
                # print("attention on bos =", torch.mean(torch.sum(prompt_to_gen_att_bos, axis=1)).float().to('cpu').numpy())
                # print("attention on instruction =", torch.mean(torch.sum(prompt_to_gen_instr, axis=1)).float().to('cpu').numpy())
                # print("attention on magic phrase =", torch.mean(torch.sum(prompt_to_gen_magic, axis=1)).float().to('cpu').numpy())
                # print("attention on query =", torch.mean(torch.sum(prompt_to_gen_query, axis=1)).float().to('cpu').numpy())
                # check that the sum of attentions is 1
                # print("sum of attention on bos =", torch.sum(prompt_to_gen_att_bos).float().to('cpu').numpy())
                # print("sum of attention for each token generated after prompt on the prompt", torch.sum(prompt_to_gen_att, axis=1).float().to('cpu').numpy())
                # print("sum of attention for each token generated after prompt", torch.sum(attentions[prompt_len:], axis=1).float().to('cpu').numpy())
                assert batch_input_ids.shape[0] == 1
                plot_attention_map_with_bars(attentions, batch_input_ids.squeeze(), prompt_len, self.llm.tokenizer, save_path=f"figs/attention_map_{self.llm.model_name.replace('/', '_')}.{j}.html")
            elif not nih_output:
                print("NIH output is None")

            for layer in layers:
                attentions = output['attentions'][0][layer][0]                                        
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
        # print("d", d)
        # print("scores", scores)
        return d, scores

