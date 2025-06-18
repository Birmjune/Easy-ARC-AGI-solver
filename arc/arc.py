import multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)   # CUDA 프로세스 안전
from multiprocessing import Process, Queue, Event

import unsloth
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

import sys

import queue
import os
import json
import copy
import torch
import numpy as np
from typing import List
from datasets import Dataset
import time
import uuid
from peft import LoraConfig, get_peft_model
from collections import defaultdict
import math
import torch
import numpy as np
import threading

from .training_code.arc_loader import ArcDataset
from .training_code.model_tools import InputMaskingDataCollator
from .training_code.model_tools import load_unsloth_4bit
from .training_code.inference_tools import infer_single
from transformers import GenerationConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from .CPU_util import *

save_model_path = os.path.abspath("artifacts/pretrained_models/Llama-3.1-8B-merged-2")
# save_model_path = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" 

def _cpu_worker(task_data, q: Queue):
    """가벼운 CPU solver 실행 → (성공, output) 전송"""
    res = CPU_solver(task_data)
    solved = (
        isinstance(res, list) and len(res) > 0 or
        isinstance(res, np.ndarray) and res.size > 0
    )
    # ndarray → list 변환은 메인 프로세스에서 통일해서 처리
    q.put(("cpu", solved, res))

class ARCSolver:
    """
    ARCSolver integrated with SingleTaskTTTPredictor for test-time training
    """

    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        self.token = token
        # Model configuration
        self.model_path = save_model_path

        # Formatting options
        self.fmt_opts = dict(
            preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
            query_beg='I',
            reply_beg='\n+/-=O',
            reply_end='\n</s>',
            lines_sep='\n',
            max_tokens=2048,
        )
        
        self.unsloth_loaded_base_model = None
        self.model = None 
        self.tokenizer = None
        self.eval_start_time = 0
        self.model_loaded = False
        self.ttt_adapter_name = "ttt_adapter" 
        self.lora_kwargs = None
        self.solved_questions = 0
        
    def load_base_model(self):
        if self.model_loaded:
            return

        self.unsloth_loaded_base_model, self.tokenizer = load_unsloth_4bit(self.model_path, True)

        # LoRA hyper-parameters you want to use everywhere
        self.lora_kwargs = dict(
            r=64,
            lora_alpha=16,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=0.0,
            bias="none",
            use_rslora=True,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.unsloth_loaded_base_model,
            **self.lora_kwargs,        # ← only LoRA params here
        )

        self.ttt_adapter_name = "ttt_adapter"
        self.model.add_adapter(self.ttt_adapter_name,
                            LoraConfig(**self.lora_kwargs))

        self.fmt_opts['reply_end'] = '\n' + self.tokenizer.eos_token
        self.model_loaded = True

    def prepare_single_task_dataset(self, task_data):
        """Prepare dataset from a single task"""
        # Create a single task dataset
        single_task_challenge = {
            'task001': {
                'train': task_data['train'],
                'test': task_data['test']
            }
        }
        single_task_solutions = {
            'task001': [None]  # We don't have the solution
        }
        
        task_dataset = ArcDataset(
            challenge=single_task_challenge,
            solutions=single_task_solutions,
            keys=['task001_0'],
            is_orig=True,
        )
        
        return task_dataset
    
    def prepare_leave_one_out(self, task_data, prefix: str = "single"): # TTT용. 그러니, test data는 빼고 만든다.
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)

        Args:
            examples (List[dict]): List of training examples,
                each list element is a dictionary that contains "input" and "output"
            questions_input (List[List[int]]): A 2d grid,
                which is a input for a given question
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        
        task_data = {
            'train': examples,
            'test': [{'input': questions_input}]
        }
        """
        # ────────────────── 1. 준비 ──────────────────
        train_examples = task_data["train"]
        T = len(train_examples)
        assert T >= 1, "train examples must be ≥1"

        challenge = {}
        solutions = {}
        keys = []

        # ────────────────── 2. Leave-One-Out 생성 ──────────────────
        for i in range(T):
            loo_train = train_examples[:i] + train_examples[i+1:]
            pseudo_test_full = copy.deepcopy(train_examples[i])   # {'input', 'output'}

            base_key = f"{prefix}-loo-{i:02d}"
            # ※ ArcDataset 포맷: test 요소에 'output' X
            challenge[base_key] = {
                "train": loo_train,
                "test":  [ {"input": pseudo_test_full["input"]} ]
            }
            solutions[base_key] = [ pseudo_test_full["output"] ]
            keys.append(f"{base_key}_0")   # 단일 test → reply_num = 0

        # ────────────────── 3. ArcDataset 인스턴스 반환 ──────────────────
        return ArcDataset(
            challenge=challenge,
            solutions=solutions,
            keys=keys,
            is_orig=True
        )
    
    def prepare_leave_one_out_inference(self, task_data, prefix: str = "single"):
        train_examples = task_data["train"]
        real_test_inp = task_data["test"][0]["input"] # 우리가 진짜 풀어야 할 문제의 입력

        T = len(train_examples)
        challenge, solutions, keys = {}, {}, []

        # 각 LOO 설정에 대해, "실제 테스트 입력"을 테스트 문제로 사용
        for i in range(T):
            # i번째 학습 예제를 제외한 나머지 학습 예제들을 few-shot 프롬프트에 사용
            loo_train_for_prompt = train_examples[:i] + train_examples[i+1:]
            base_key = f"{prefix}-loo-context-{i:02d}" 
            challenge[base_key] = {
                "train": loo_train_for_prompt,  # 프롬프트에 들어갈 few-shot 예제들
                "test":  [{"input": real_test_inp}] # 실제 풀어야 할 문제를 테스트 입력으로 사용
            }
            solutions[base_key] = [None] 
            keys.append(f"{base_key}_0") # 각 base_key 당 하나의 테스트 입력

        return ArcDataset(challenge, solutions, keys, is_orig=True)
    
    def test_time_train(self, task_data):
        """Perform test time training on a single task"""        

        if self.model is not None:
            try:
                delattr(self.model, "_flag_for_generation")
            except AttributeError:
                pass  # Attribute was not present on self.model
            # Attempt to delete _flag_for_generation from the underlying base model
            if hasattr(self.model, 'model') and self.model.model is not None:
                try:
                    delattr(self.model.model, "_flag_for_generation")
                except AttributeError:
                    pass  # Attribute was not present on self.model.model
        
        if self.ttt_adapter_name in self.model.peft_config:
            self.model.delete_adapter(self.ttt_adapter_name)

        lora_config = LoraConfig(**self.lora_kwargs)
        self.model.add_adapter(self.ttt_adapter_name, lora_config)
        self.model.set_adapter(self.ttt_adapter_name)
        
        self.model.train()
        
        # Prepare dataset for this single task
        # Using the method LOO
        task_dataset = self.prepare_leave_one_out(task_data)
        
        # Augment dataset
        train_dataset_base = task_dataset.remove_test_data() 
        
        # input 크기 (행, 열)
        input_dims  = [(len(ex["input"]),  len(ex["input"][0]))  for ex in task_data["train"]]
        # output 크기 (행, 열)
        output_dims = [(len(ex["output"]), len(ex["output"][0])) for ex in task_data["train"]]

        # 둘을 합쳐서, r*c의 최댓값을 구함
        all_dims = input_dims + output_dims
        max_size = max(r * c for r, c in all_dims)

        # different augmentation # for each size
        # 난이도에 따른 다른 augmentation 횟수
        if max_size >= 72:
            num_augmentation = 1
        elif 49 <= max_size < 72:
            num_augmentation = 1
        elif 17 <= max_size < 49:
            num_augmentation = 1
        else:
            num_augmentation = 1

        train_dataset_as_list = []
        for i in range(num_augmentation):
            current_aug = copy.deepcopy(train_dataset_base)
            augmented_dataset_single_pass = current_aug.augment(tp=True, rt=True, perm=True, shfl_ex=True, seed = i)
            train_dataset_as_list.extend(augmented_dataset_single_pass.as_list(len_name='text', **self.fmt_opts))
            
        # train_dataset_augment = train_dataset_augment.augment(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        # train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **self.fmt_opts)
        
        FastLanguageModel.for_training(self.model)
        
        # Run test-time training
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=Dataset.from_list(train_dataset_as_list),
            dataset_text_field="text",
            max_seq_length=self.fmt_opts['max_tokens'],
            data_collator=InputMaskingDataCollator(
                instruction_template=self.fmt_opts['query_beg'],
                response_template=self.fmt_opts['reply_beg'],
                mlm=False, 
                tokenizer=self.tokenizer,
                mask_first_n_examples=0,      
            ),
            args=TrainingArguments(
                per_device_train_batch_size=4,          
                gradient_accumulation_steps=2,           
                num_train_epochs=2,
                learning_rate=5e-5,                      
                warmup_ratio=0.15,
                lr_scheduler_type='cosine',                  
                optim="adamw_8bit",
                weight_decay=0.00,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                output_dir='tmp_output',
                logging_steps=3,
                save_strategy='no',
                report_to='none',
                seed=42,
            ),
        )
        
        trainer_stats = unsloth_train(trainer)
        
        return self.model

    def predict_output_majority_vote(self, model, task_data, stop_evt = None):
        """Predict output for the given task, using majority vote"""
        # Prepare dataset for inference
        task_dataset = self.prepare_leave_one_out_inference(task_data)
        
        # Set model to inference mode
        FastLanguageModel.for_inference(model)
        
        # Apply augmentation for inference
        # infer_aug_opts = dict(tp=True, rt=True, perm =False, shfl_ex=True, seed=10000)
        # infer_dataset = task_dataset.augment(**infer_aug_opts)
        # # infer_dataset = task_dataset
        
        dims = [(len(ex["input"]), len(ex["input"][0])) for ex in task_data["train"]]
        max_size = max(r*c for r, c in dims)
        
        print(f"\n Max size: {max_size}")
        print(f"# of train data: {len(dims)}")

        # different augmentation for each grid size
        # max_dim, min_dim으로 할지, grid의 총 size로 할지..
        
        base_keys = task_dataset.keys          
        merged_keys = list(base_keys)
        extra_frac = 0            
        
        inference_bound = 48
        passed_min = (time.time() - self.eval_start_time) / 60
        mean_time = (99 - passed_min) / (100 - self.solved_questions)
        print(f"\nSolved questions: {self.solved_questions} (Before Inference) \nMean time left: {mean_time:.3f}")
        
        flag = (max_size >= inference_bound and mean_time >= 1) # inference 더 할지 확인하는 flag
        
        if flag: 
            rng = np.random.default_rng(seed=10000)                          
            n_extra = max(1, int(len(base_keys) * extra_frac)) # 1개씩 추가되는 거임. 지금 input 3개를 보니. 
            extra_src_keys = rng.choice(base_keys, n_extra, replace=False).tolist()
            aug_ds = (task_dataset.change_keys(extra_src_keys).augment(seed=1, tp=True, rt=True, perm=False, shfl_ex=True))
            merged_keys.extend(aug_ds.keys)
        else:
            aug_ds = task_dataset.augment(seed=1, tp=True, rt=True, perm=False, shfl_ex=True)
            merged_keys = aug_ds.keys                

        # ArcDataset 객체들을 병합
        infer_dataset = task_dataset.change_keys(merged_keys)
        
        # Run inference + majority vote check ? << inference 과정을 아예 바꿔야 하나 흠..
        pick_min = 1 # LOO 하나당 뽑는 후보의 수 
        print(f"Majority vote, pick min = {pick_min}")
        
        vote_box = defaultdict(lambda: {"count": 0, "best_score": float("inf"), "output": None})
        
        half = 0
        if (flag):
            # 지금 extra frac을 0으로 해놔서 +1을 뒤에 추가해 둠. 시간 남으면 수정? 
            half = int(len(infer_dataset.keys) * pick_min * (1 + extra_frac) + 1) // 2 
        else:
            half = int(len(infer_dataset.keys) * pick_min) // 2
        
        for key in infer_dataset.keys:
            if stop_evt is not None and stop_evt.is_set():
                return [[0]]      # 조기 탈출
            task_fmt = infer_dataset.get_task(key, len_name="input", **self.fmt_opts)[1]
            
            inference_results = infer_single(
                prompt=task_fmt['input'],
                model_tok=(model, self.tokenizer),
                min_prob=0.05, # 0.1,
                max_new_tokens=150,
            )
            
            # for debugging
            print(f"Infernce #: {self.solved_questions}")
            print(len(inference_results))

            # 프롬프트당 가장 점수가 낮은 1개 후보만 사용
            if not inference_results:       
                continue
            
            # 점수 오름차순 정렬 후 하위 k개 선택
            inference_results.sort(key=lambda x: x[1])
            k = min(pick_min, len(inference_results))
            
            for seq, score in inference_results[:k]:
                output, _, _ = infer_dataset.decode(seq, self.fmt_opts["lines_sep"], key)
                if output is None:
                    continue
                
                print(output)
                print(score)
                
                out_key = tuple(map(tuple, output))
                v = vote_box[out_key]
                v["count"] += 1
                if score < v["best_score"]:
                    v["best_score"] = score
                    v["output"] = output

                # 과반수 즉시 반환  
                if v["count"] > half:
                    return output
        
        #  4. 최종 다수결
        if vote_box:
            max_votes = max(v["count"] for v in vote_box.values())
            winners = [v for v in vote_box.values() if v["count"] == max_votes]
            return min(winners, key=lambda x: x["best_score"])["output"]
        else:
            return [[0]]
      
    def predict(self, examples, questions_input):
        # ───── 0. 타임아웃 확인 ─────
        if (time.time() - self.eval_start_time) / 60 >= 99:
            return questions_input

        task_data = {"train": examples,
                    "test":  [{"input": questions_input}]}

        # ───── 1. IPC 구성 ─────
        q          = mp.Queue()
        stop_evt   = mp.Event()

        # ───── 2. CPU 프로세스 시작 ─────
        cpu_proc = mp.Process(
            target=_cpu_worker,
            args=(task_data, q)
        )
        cpu_proc.start()

        # ───── 3. GPU 경로는 메인에서 바로 실행 ─────
        # (모델 이미 로드되어 있으므로 추가 지연 없음)
        gpu_out = None
        gpu_success = False
        def gpu_path():
            nonlocal gpu_out, gpu_success

            # --- 1) CPU 쪽이 이미 풀었으면 즉시 종료 ---
            if stop_evt.is_set():
                return

            trained = self.test_time_train(task_data)

            # --- 2) 학습 직후 다시 한 번 체크 ---
            if stop_evt.is_set():
                return

            gpu_out = self.predict_output_majority_vote(
                trained, task_data, stop_evt)

            # --- 3) 추론 직후에도 체크 ---
            if stop_evt.is_set():
                return

            gpu_success = True
            q.put(("gpu", True, gpu_out))

            del gpu_out ; torch.cuda.empty_cache()


        gpu_path_thread = threading.Thread(target=gpu_path, daemon=True)
        gpu_path_thread.start()
        # ───── 4. 결과 판정 ─────
        # Queue 안에는 CPU (먼저) + GPU(늦게 혹은 X) 결과가 차례로 들어온다
        first_src, success, out = q.get()   # 첫 도착 결과
        if first_src == "cpu" and success:
            self.solved_questions += 1
            print(f"\nElapsed time: {(time.time() - self.eval_start_time) / 60}, Solved questions: {self.solved_questions}", flush = True)
            # CPU 성공 → GPU 중단
            stop_evt.set()
            cpu_proc.join()
            gpu_path_thread.join(timeout=3)
            if gpu_path_thread.is_alive():
                print("[warn] GPU thread didn’t terminate within 3 s", flush=True)

            q.close(); q.cancel_join_thread()     # 메모리 누수 방지
            return out if isinstance(out, list) else out.tolist()
        elif first_src == "cpu" and not success:
            # CPU 실패 → GPU 결과 기다림
            try:
                src2, ok2, out2 = q.get(timeout = 600)       # 반드시 GPU
            except queue.Empty:
                gpu_path_thread.join()
                if gpu_path_thread.is_alive():
                    print("[warn] GPU thread still alive after join", flush=True)

                q.close(); q.cancel_join_thread()
                self.solved_questions += 1
                print(f"\nElapsed time: {(time.time() - self.eval_start_time) / 60}, Solved questions: {self.solved_questions}", flush = True)
                return [[0]]

            gpu_path_thread.join()
            self.solved_questions += 1
            print(f"\nElapsed time: {(time.time() - self.eval_start_time) / 60}, Solved questions: {self.solved_questions}", flush = True)
            return out2 if isinstance(out2, list) else out2.tolist()
        else:
            # GPU 가 먼저 끝난 희귀 케이스
            stop_evt.set()                  # CPU solver 이미 끝났지만 혹시 모르니
            cpu_proc.join(timeout=1)
            q.close(); q.cancel_join_thread()
            self.solved_questions += 1
            print(f"\nElapsed time: {(time.time() - self.eval_start_time) / 60}, Solved questions: {self.solved_questions}", flush = True)
            return out if isinstance(out, list) else out.tolist()

    def prepare_evaluation(self):
        self.eval_start_time = time.time()
        self.load_base_model()