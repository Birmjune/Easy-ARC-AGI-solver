# Copyright 2024 Daniel Franzen and Jan Disselhoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments
from datasets import Dataset

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import load_unsloth_4bit, keep_single_char_tokens, save_model_and_tokenizer

# input paths
base_model = 'meta-llama/Llama-3.1-8B-Instruct'

# output paths
save_model_path = os.path.join('./artifacts/pretrained_models', "Llama-3.1-8B")

# load base model & reduce embedding size
model = tokenizer = None  # free memory
model, tokenizer = load_unsloth_4bit(base_model, True)
keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

# set formatting options
fmt_opts = dict(
preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
    query_beg='I',
    reply_beg='\n+/-=O',
    reply_end='\n' + tokenizer.eos_token,
    lines_sep='\n',
    max_tokens=2048,
)

# create lora model
lora_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head']
model = FastLanguageModel.get_peft_model(
    model=model,
    target_modules=lora_layers,
    r=256,
    lora_alpha=24,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
    use_rslora=True,
    loftq_config=None,
)
train_dataset = ArcDataset.load_from_dataset("/home/student/workspace/dataset", n=200, sizes=[6], seed=42)

# augment data set and transform to list (eventually removing examples to stay below the max. token count)
train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
train_dataset_augment = train_dataset.augment(**train_aug_opts)
train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

# run training
FastLanguageModel.for_training(model)
tokenizer.padding_side = 'right'
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=Dataset.from_list(train_dataset_as_list),
    dataset_text_field="text",
    max_seq_length=fmt_opts['max_tokens'],
    packing=False,
    data_collator=InputMaskingDataCollator(
        instruction_template=fmt_opts['query_beg'],
        response_template=fmt_opts['reply_beg'],
        mlm=False,
        tokenizer=tokenizer,
        mask_first_n_examples=1,
    ),
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.25,
        num_train_epochs=2,
        max_steps=3500,
        learning_rate=1e-4,
        embedding_learning_rate=1e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.00,
        lr_scheduler_type='cosine',
        save_strategy="no",
        output_dir="./training_code",
        seed=42,
        report_to='none',
    ),
)
trainer_stats = unsloth_train(trainer)
save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)

import gc
import torch

del model
del tokenizer
del trainer_stats
del trainer

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

import torch
from peft import PeftModel
from model_tools import keep_single_char_tokens

base_model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
lora_adapter_path = f'{save_model_path}-lora'
merged_model_save_path = f'{save_model_path}-merged-1'

base_model, tokenizer = FastLanguageModel.from_pretrained(base_model_name_or_path, dtype=None, load_in_4bit=False, device_map="cpu")
keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
keep_single_char_tokens(base_model, tokenizer, keep=keep_tok, remove_unk=True)

model = PeftModel.from_pretrained(base_model, lora_adapter_path, is_trainable=False)
model = model.merge_and_unload()
model = model.half()
model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path)

import gc
import torch

del base_model
del model
del tokenizer

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(merged_model_save_path, load_in_4bit=True, device_map="cuda")
save_model_and_tokenizer(f'{save_model_path}-merged-2', model, tokenizer)