# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
#device_ids = [0, 1]
import random
rng = random.Random()
import sys
import logging
import math
import numpy as np
import pickle
import time
import sched
import json
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from torch.utils.data import ConcatDataset
from torch.utils.data.dataset import Dataset
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

from transformers.modeling_electra import ElectraForPreTrainingOutput
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from transformers.modeling_bert import BertPooler

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    #DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    # LineByLineTextDataset,
    PreTrainedTokenizer,
    # TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)
schedule = sched.scheduler(time.time,time.sleep)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)

        directory, filename = os.path.split(file_path)

        cached_features_file = os.path.join(
            directory,
            "cached_feature_{}_{}_{}_line_by_line".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        cached_start_ent_file = os.path.join(
            directory,
            "cached_start_ent_{}_{}_{}_line_by_line".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        cached_end_ent_file = os.path.join(
            directory,
            "cached_end_ent_{}_{}_{}_line_by_line".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        cached_start_ngram_file = os.path.join(
            directory,
            "cached_start_ngram_{}_{}_{}_line_by_line".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        cached_end_ngram_file = os.path.join(
            directory,
            "cached_end_ngram_{}_{}_{}_line_by_line".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )
        if os.path.exists(cached_features_file):
            logger.info("Loading CWS features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            
            with open(cached_start_ent_file, "rb") as handle:
                self.start_ent = pickle.load(handle)

            with open(cached_end_ent_file, "rb") as handle:
                self.end_ent = pickle.load(handle)

            with open(cached_start_ngram_file, "rb") as handle:
                self.start_ngram = pickle.load(handle)

            with open(cached_end_ngram_file, "rb") as handle:
                self.end_ngram = pickle.load(handle)
            
        else:
            logger.info("Creating features from dataset file at %s", file_path)
            with open(file_path, encoding="utf-8") as f:
                lines = [line.strip() for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

            self.start_ent = []
            self.end_ent = []
            self.start_ngram = []
            self.end_ngram = []
            
            for line in lines:
                #line = list(line)
                line = tokenizer.tokenize(line)
                self.start_ent.append([])
                self.end_ent.append([])
                self.start_ngram.append([])
                self.end_ngram.append([])
                pos = 0
                for _ in range(len(line)):
                    if line[_] == '[':
                        self.start_ent[-1].append(pos + 1)
                        self.start_ngram[-1].append(pos + 1)
                    elif line[_] == ']':
                        self.end_ent[-1].append(pos + 1)
                        self.end_ngram[-1].append(pos + 1)
                    elif line[_] == '(':
                        self.start_ngram[-1].append(pos + 1)
                    elif line[_] == ')':
                        self.end_ngram[-1].append(pos + 1)
                    else:
                        pos += 1
                assert len(self.start_ent[-1]) == len(self.end_ent[-1])
                assert len(self.start_ngram[-1]) == len(self.end_ngram[-1])

           
            lines = [line.replace('[', '').replace(']', '').replace('(', '').replace(')', '').split() for line in lines]
            batch_encoding = tokenizer(lines, is_split_into_words=True, add_special_tokens=True, truncation=True, max_length=block_size)

            self.examples = batch_encoding["input_ids"]
           
            self.examples = [item + [0]*(block_size - len(item)) for item in self.examples]

            logger.info("Saving features into cached file %s", cached_features_file)

            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(cached_start_ent_file, "wb") as handle:
                pickle.dump(self.start_ent, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(cached_end_ent_file, "wb") as handle:
                pickle.dump(self.end_ent, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(cached_start_ngram_file, "wb") as handle:
                pickle.dump(self.start_ngram, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(cached_end_ngram_file, "wb") as handle:
                pickle.dump(self.end_ngram, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long), self.start_ent[i], self.end_ent[i], self.start_ngram[i], self.end_ngram[i]


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    sop_probability: float = 0

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]

        batch = [e[0] for e in examples]
        start_ent = [e[1] for e in examples]
        end_ent = [e[2] for e in examples]
        start_ngram= [e[3] for e in examples]
        end_ngram= [e[4] for e in examples]

        batch = self._tensorize_batch(batch)

        if self.mlm:
            # inputs, labels = self.mask_tokens(batch_start_end)
            return {"input_ids": batch, "labels": None, "start_ent": start_ent, "end_ent": end_ent, "start_ngram": start_ngram, "end_ngram": end_ngram}
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)


class ElectraCombineModel(nn.Module):
    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    sop_probability: float = 0

    def __init__(self, discriminator, loss_weights=50.0, tokenizer=None, mlm_probability=0.15,
                 sop_probability=0):
        super().__init__()


        self.loss_weights = loss_weights
        self.discriminator = discriminator
        self.vocab_size = discriminator.config.vocab_size
        self.mlm_probability = mlm_probability
        self.sop_probability = sop_probability
        self.tokenizer = tokenizer
        self.mse_fnc = torch.nn.MSELoss(reduce=True, size_average=True)
        self.discriminator.bert.pooler = BertPooler(discriminator.config)
        #self.mse_fnc = torch.nn.KLDivLoss(reduction='mean')
        print("### hello world the Electra model!!!")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,  # - 1 for tokens that are **not masked**,
                                  # - 0 for tokens that are **maked**.
            token_type_ids=None,
            labels=None,
            start_ent=None,
            end_ent=None,
            start_ngram=None,
            end_ngram=None,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``:

            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # copy the variables for use with discriminator.

        # run masked LM.

        #d_inputs_1, d_labels_1, d_inputs_2, d_labels_2, d_inputs_3 = self.mask_tokens((input_ids, start_ent, end_ent, start_ngram, end_ngram))
        d_inputs_1, d_labels_1, d_inputs_2, d_labels_2, d_inputs_3 = self.mask_tokens((input_ids, start_ent, end_ent, start_ent, end_ent))
        d_attention_mask_1 = (d_inputs_1 != 0).long()
        d_attention_mask_2 = (d_inputs_2 != 0).long()
        d_attention_mask_3 = (d_inputs_3 != 0).long()

        d_out_1 = self.discriminator(d_inputs_1,
                                     labels=d_labels_1,
                                     attention_mask=d_attention_mask_1,
                                     token_type_ids=token_type_ids,
                                     output_hidden_states=True,
                                     return_dict=True, )

        d_out_2 = self.discriminator(d_inputs_2,
                                     labels=d_labels_2,
                                     attention_mask=d_attention_mask_2,
                                     token_type_ids=token_type_ids,
                                     output_hidden_states=True,
                                     return_dict=True, )

        d_out_3 = self.discriminator(d_inputs_3,
                                     labels=None,
                                     attention_mask=d_attention_mask_3,
                                     token_type_ids=token_type_ids,
                                     output_hidden_states=True,
                                     return_dict=True, )
        hid_states_1 = d_out_1.hidden_states
        hid_states_2 = d_out_2.hidden_states
        hid_states_3 = d_out_3.hidden_states
        
        embedding_1 = self.discriminator.bert.pooler(d_out_1.hidden_states[-1])
        embedding_2 = self.discriminator.bert.pooler(d_out_2.hidden_states[-1])
        embedding_3 = self.discriminator.bert.pooler(d_out_3.hidden_states[-1])
       
        embedding_1 = embedding_1 / torch.sqrt(torch.sum(embedding_1 * embedding_1, 1).float()).reshape(-1, 1)
        embedding_2 = embedding_2 / torch.sqrt(torch.sum(embedding_2 * embedding_2, 1).float()).reshape(-1, 1)
        embedding_3 = embedding_3 / torch.sqrt(torch.sum(embedding_3 * embedding_3, 1).float()).reshape(-1, 1)
        embedding_4 = embedding_2 + embedding_3
        embedding_4 = embedding_4 / torch.sqrt(torch.sum(embedding_4 * embedding_4, 1).float()).reshape(-1, 1)
        #embedding_1 = self.discriminator.bert.pooler.activation(self.discriminator.bert.pooler.dense(torch.mean(hid_states_1[-1], -2)))
        #embedding_2 = self.discriminator.bert.pooler.activation(self.discriminator.bert.pooler.dense(torch.mean(hid_states_2[-1], -2)))
        #embedding_3 = self.discriminator.bert.pooler.activation(self.discriminator.bert.pooler.dense(torch.mean(hid_states_3[-1], -2)))
        
        #embedding_1 = torch.mean(hid_states_1[-1], -2)
        #embedding_2 = torch.mean(hid_states_2[-1], -2)
        #embedding_3 = torch.mean(hid_states_3[-1], -2)

        mse_loss = self.mse_fnc(embedding_4, embedding_1)

        mlm_loss = d_out_1.loss + d_out_2.loss


        #print('mse_loss: ', mse_loss)
        total_loss =  mlm_loss + mse_loss 

        return ElectraForPreTrainingOutput(
            loss=total_loss,
            logits=d_out_1.logits,
            hidden_states=None,
            attentions=None,
        )
    
    def mlm_span_mask(self, inputs, ngram_mask=None):
        labels = inputs.clone()
        mlm_masks = torch.full(inputs.shape, 0, dtype=int)
        mlm_length_distribution = np.array([0.2 * (1 - 0.2) ** (i - 1) for i in range(1, 11-5)])
        mlm_length_distribution /= mlm_length_distribution.sum()
        for i in range(inputs.size(0)):
            seq_len = torch.sum(inputs[i] != 0).cpu().numpy()
            max_mlm_tokens = seq_len * self.mlm_probability
            num_mlm_tokens = 0
            while num_mlm_tokens < max_mlm_tokens:
                sample_len = np.random.choice(range(1, 11-5), p=mlm_length_distribution)
                sample_pos = np.random.choice(seq_len)
                if ngram_mask is not None:
                    if ngram_mask[i][sample_pos: sample_pos + sample_len].sum() > 0:
                        continue
                if sample_pos == 0:
                    continue
                if sample_pos + sample_len >= seq_len:
                    continue
                if mlm_masks[i][sample_pos: sample_pos + sample_len].sum() > 0:
                    continue
                num_mlm_tokens += sample_len
                mlm_masks[i][sample_pos: sample_pos + sample_len] = 1
                if rng.random() < 0.8:
                    inputs[i][sample_pos: sample_pos + sample_len] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.mask_token)
                else:
                    if rng.random() < 0.5:
                        inputs[i][sample_pos: sample_pos + sample_len] = torch.randint(len(self.tokenizer),(sample_len,), dtype=torch.long)
        
        mlm_masks = mlm_masks.bool()
        labels[~mlm_masks] = -100
        return inputs, labels

    def mlm_token_mask(self, inputs, ngram_mask=None):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability).cuda()
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).cuda(), value=0.0)
        probability_matrix.masked_fill_((labels == 0).cuda(), value=0.0)
        if ngram_mask is not None:
            probability_matrix.masked_fill_(torch.tensor(ngram_mask.bool(), dtype=torch.bool).cuda(), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8).cuda()).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5).cuda()).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long).cuda()
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    def mlm_ngram_mask(self, inputs, start_pos, end_pos):
        labels = inputs.clone()
        mlm_masks = torch.full(inputs.shape, 0, dtype=int)

        for i in range(inputs.size(0)):
            num_mlm_tokens = 0
            seq_len = sum(inputs[i] != 0) - 2
            max_mlm_tokens = seq_len * self.mlm_probability

            while num_mlm_tokens < max_mlm_tokens and len(start_pos[i]) > 0:

                idx = np.random.choice(range(len(start_pos[i])))

                ngram_start = start_pos[i][idx]
                ngram_end = end_pos[i][idx]
                if ngram_start <=0  or ngram_start >= ngram_end or ngram_end > inputs.size(1) or inputs[i][ngram_end-1] == 0:
                    del start_pos[i][idx]
                    del end_pos[i][idx]
                    continue
                if rng.random() < 0.8:
                    inputs[i][ngram_start:ngram_end] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                else:
                    if rng.random() < 0.5:
                        '''
                        print(inputs[i])
                        print(ngram_start, ngram_end, inputs.size(1))
                        print(torch.randint(len(self.tokenizer), (ngram_end - ngram_start,), dtype=torch.long))
                        print(inputs[i][ngram_start:ngram_end])
                        '''
                        inputs[i][ngram_start:ngram_end] = torch.randint(len(self.tokenizer), (ngram_end - ngram_start,), dtype=torch.long)
                num_mlm_tokens += ngram_end - ngram_start
                mlm_masks[i][ngram_start:ngram_end] = 1

                del start_pos[i][idx]
                del end_pos[i][idx]

        mlm_masks = mlm_masks.bool()
        labels[~mlm_masks] = -100

        return inputs, labels

    def mask_tokens(self, batch: Tuple[torch.Tensor, List, List, List, List]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        inputs, start_ent, end_ent, start_ngram, end_ngram = batch

        inputs_1 = inputs.clone()
        inputs_2 = inputs.clone()
        inputs_3 = inputs.clone()

        ngram_mask = torch.full(inputs.shape, 0, dtype=int)

        start_ngram_1 = []
        end_ngram_1 = []
        start_ngram_2 = []
        end_ngram_2 = []
        
        for i in range(inputs.size(0)):

            start_ngram_1.append([])
            end_ngram_1.append([])
            start_ngram_2.append([])
            end_ngram_2.append([])
        
            if len(start_ent[i]) > 0:
              # TODO
              idx = np.random.choice(range(len(start_ent[i])))           
              ngram_start = start_ent[i][idx]
              ngram_end = end_ent[i][idx]
            else:
              mlm_length_distribution = np.array([0.2 * (1 - 0.2) ** (i - 1) for i in range(1, 11-5)])
              mlm_length_distribution /= mlm_length_distribution.sum()
              sample_idx = np.random.choice(range(torch.sum(inputs[i] != 0).cpu().numpy()))
              sample_len = np.random.choice(range(1, 11-5), p=mlm_length_distribution)
              ngram_start = sample_idx
              ngram_end = sample_idx + sample_len
            for j in range(0, len(start_ngram[i])):
                if start_ngram[i][j] == ngram_start:
                  continue
                elif start_ngram[i][j] < ngram_start:
                  start_ngram_1[-1].append(start_ngram[i][j])
                  end_ngram_1[-1].append(end_ngram[i][j])
                  start_ngram_2[-1].append(start_ngram[i][j])
                  end_ngram_2[-1].append(end_ngram[i][j])
                else:
                  start_ngram_1[-1].append(start_ngram[i][j])
                  end_ngram_1[-1].append(end_ngram[i][j])
                  start_ngram_2[-1].append(start_ngram[i][j] - (ngram_end - ngram_start))
                  end_ngram_2[-1].append(end_ngram[i][j] - (ngram_end - ngram_start))
            
            ngram_mask[i][ngram_start: ngram_end] = 1

            inputs_2[i][ngram_start: inputs.size(1) - ngram_end + ngram_start] = inputs[i][ngram_end:]
            inputs_2[i][inputs.size(1) - ngram_end + ngram_start:] = 0

            inputs_3[i][1: ngram_end - ngram_start + 1] = inputs[i][ngram_start: ngram_end]
            inputs_3[i][ngram_end - ngram_start + 1] = 102
            inputs_3[i][ngram_end - ngram_start + 2:] = 0

        inputs_1, labels_1 = self.mlm_token_mask(inputs_1, ngram_mask=ngram_mask)
        inputs_2, labels_2 = self.mlm_token_mask(inputs_2)
        #inputs_1, labels_1 = self.mlm_span_mask(inputs_1, ngram_mask=ngram_mask)
        #inputs_2, labels_2 = self.mlm_span_mask(inputs_2)
        #inputs_1, labels_1 = self.mlm_ngram_mask(inputs_1, start_ngram_1, end_ngram_1)
        #inputs_2, labels_2 = self.mlm_ngram_mask(inputs_2, start_ngram_2, end_ngram_2)
        
        return inputs_1, labels_1, inputs_2, labels_2, inputs_3

    def save_pretrained(self, directory):
        # print("### I want to save something!!!!")
        # if self.config.xla_device:
        #    self.discriminator.config.xla_device = True
        #    self.generator.config.xla_device = True
        # else:
        self.discriminator.config.xla_device = False

        discriminator_path = os.path.join(directory, "discriminator")

        if not os.path.exists(discriminator_path):
            os.makedirs(discriminator_path)

        self.discriminator.save_pretrained(discriminator_path)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
            "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=True,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    sop_probability: float = field(
        default=0, metadata={"help": "Ratio of sop samples"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
):
    def _dataset(file_path):
        if args.line_by_line:
            return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
        else:
            return TextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=args.block_size,
                overwrite_cache=args.overwrite_cache,
                cache_dir=cache_dir,
                ngram_file=args.ngram_file,
                N=args.N,
            )

    if evaluate:
        return _dataset(args.eval_data_file)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = BertConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)


    model_bert = BertForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
    )

    model_bert.resize_token_embeddings(len(tokenizer))

    model = ElectraCombineModel(model_bert, tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
        tokenizer=tokenizer,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
    #schedule.enter(7*3600, 0, main, ())  # 3个小时后运行main()
    #schedule.run()
