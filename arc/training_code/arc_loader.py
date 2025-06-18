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
import re
import itertools
import json
import hashlib
import numpy as np
from numpy.random import randint
from glob import glob
from tqdm import tqdm
from collections import OrderedDict


class ArcDataset(object):
    def __init__(self, challenge, solutions={}, keys=None, is_fake=False, is_orig=False):
        if keys is None:
            self.keys = []
            for k, v in challenge.items():
                reply_num = len(v['test'])
                self.keys.extend([f'{k}_{i}' for i in range(reply_num)] if reply_num else [k])
            self.keys = sorted(self.keys)
        else:
            self.keys = [k for k in keys]
        base_keys = set(map(self.get_base_key, self.keys))
        self.challenge = {k: challenge[k] for k in base_keys}
        self.solutions = {k: solutions[k] for k in base_keys if k in solutions}
        self.is_orig = is_fake
        self.is_orig = is_orig

    @classmethod
    def load_from_json(cls, challenges_file):  # for loading challenges in kaggle json arc dataset format
        with open(challenges_file) as f:
            challenge = f.read()
        return cls(
            challenge=json.loads(challenge),
            is_fake=hashlib.md5(challenge.encode('utf-8')).hexdigest().lower() == 'a6b7dac3cab03abf2eb333e16610d6dc',
            is_orig=True,
        )

    def load_solutions(self, solutions_file):  # for loading solutions in kaggle json arc dataset format
        with open(solutions_file) as f: solutions = f.read()
        data = json.loads(solutions)
        solutions = {k: data[k] for k in self.challenge}
        return self.__class__(keys=self.keys, challenge=self.challenge, solutions=solutions, is_orig=self.is_orig)

    # loader for Michael Hodel's ReArc https://github.com/neoneye/arc-dataset-collection
    @classmethod
    def load_from_dataset(cls, path, n, sizes, seed, dataset_prefix="customds", mixdatasets={}, shuffle=True):
        """
        지정된 폴더에서 여러 JSON 파일을 로드하여 ARC 형식의 데이터셋을 생성합니다.
        각 JSON 파일은 다수의 {"input": ..., "output": ...} 쌍을 포함하는 리스트여야 합니다.

        Args:
            cls: 클래스 자신.
            path (str): JSON 파일들이 포함된 폴더 경로.
            n (int): 생성할 에포크 수.
            sizes (iterable): 각 문제 인스턴스에 사용할 학습 예제 개수 옵션.
            seed (int): 난수 생성 시드.
            dataset_prefix (str, optional): 생성될 키에 사용될 접두사. Defaults to "customds".
            mixdatasets (dict, optional): 혼합할 추가 데이터셋. Defaults to {}.
            shuffle (bool, optional): 에포크 내 문제 순서 셔플 여부. Defaults to True.

        Returns:
            ARCFormatDataset: 로드된 데이터셋 객체.
        """
        np.random.seed(seed)
        keys_per_epoch = [[] for _ in range(n)]
        challenge = {}
        solutions = {}
        train_sizes_options = list(sizes)

        json_filepaths = []
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)): # 파일 이름 순으로 정렬하여 일관성 유지
                if fname.endswith('.json'):
                    json_filepaths.append(os.path.join(path, fname))

        # 2. 각 JSON 파일(원본 문제 유형)을 순회하며 처리합니다.
        for filepath in tqdm(json_filepaths, desc=f"Load dataset '{dataset_prefix}'"):
            filename_with_ext = os.path.basename(filepath)
            # 파일명에서 확장자를 제거하여 이 파일의 기본 키로 사용합니다.
            file_base_key = os.path.splitext(filename_with_ext)[0]

            with open(filepath, 'r') as f:
                try:
                    # 파일 내 모든 예제 ({"input": ..., "output": ...} 쌍의 리스트)를 로드합니다.
                    all_examples_in_file = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from file {filename_with_ext}. Skipping.")
                    continue
                
                if not isinstance(all_examples_in_file, list):
                    print(f"Warning: Content of {filename_with_ext} is not a list. Skipping.")
                    continue
                if not all_examples_in_file:
                    print(f"Warning: File {filename_with_ext} is empty. Skipping.")
                    continue
            
            if len(all_examples_in_file) < 700: 
                # 700개로 맞추기 위해 예제를 반복
                while len(all_examples_in_file) < 700:
                    # 기존 예제들을 복사하여 추가
                    additional_examples = all_examples_in_file.copy()
                    # 남은 공간만큼만 추가 
                    space_left = 700 - len(all_examples_in_file)
                    all_examples_in_file.extend(additional_examples[:space_left])

            # 이 파일의 예제들을 무작위로 섞습니다. 이 리스트는 아래 에포크 루프에서 소진됩니다.
            # ReArc 로더의 `tasks` 변수와 유사하게 동작합니다.
            current_file_tasks = np.random.permutation(all_examples_in_file).tolist()

            # 각 파일(문제 유형)에 대해, 사용할 학습 예제 크기들을 관리합니다.
            # ReArc 로더의 `next_sizes`와 유사합니다.
            current_file_train_size_choices = []

            for epoch_idx in range(n): # 각 에포크에 대해
                if not current_file_train_size_choices:
                    # 사용할 학습 크기 옵션이 소진되면, 전체 옵션을 다시 섞어서 채웁니다.
                    current_file_train_size_choices = np.random.permutation(train_sizes_options).tolist()
                
                # 이번 에포크에서 사용할 학습 예제 수를 하나 선택합니다.
                num_train_samples = current_file_train_size_choices.pop()
                # 필요한 총 예제 수 (학습 예제 + 테스트 예제 1개)
                num_total_samples_to_take = num_train_samples + 1

                if len(current_file_tasks) < num_total_samples_to_take:
                    break

                # 이 문제 인스턴스의 고유 키를 생성합니다.
                # 형식: "데이터셋접두사-파일기본키-에포크번호(2자리16진수)"
                instance_base_key = f'{dataset_prefix}-{file_base_key}-{epoch_idx:02x}'
                
                challenge[instance_base_key] = {'train': [], 'test': []}
                # solutions는 각 테스트 케이스의 output을 담는 리스트여야 함.
                solutions[instance_base_key] = [] 

                # 1. 필요한 총 예제 수만큼 `current_file_tasks`에서 예제를 가져와 임시로 'train'에 모두 넣습니다.
                #    `pop()`은 리스트의 마지막 요소를 가져오므로, 섞인 리스트에서 무작위로 선택하는 효과가 있습니다.
                for _ in range(num_total_samples_to_take):
                    example = current_file_tasks.pop()
                    challenge[instance_base_key]['train'].append(example)
                
                # 2. 'train' 리스트에서 마지막으로 추가된 예제를 테스트 예제로 사용합니다.
                test_example_full = challenge[instance_base_key]['train'].pop()

                # 3. 테스트 예제의 'input' 부분만 `challenge`의 'test' 필드에 저장합니다.
                #    각 예제는 {"input": ..., "output": ...} 형태라고 가정합니다.
                if not all(k in test_example_full for k in ('input', 'output')):
                    raise ValueError(
                        f"Example in {filename_with_ext} is missing 'input' or 'output' field: {test_example_full}"
                    )
                challenge[instance_base_key]['test'].append({'input': test_example_full['input']})
                
                # 4. 테스트 예제의 'output' 부분을 `solutions`에 저장합니다.
                solutions[instance_base_key].append(test_example_full['output'])
                
                # 이 문제 인스턴스의 전체 키를 해당 에포크의 키 리스트에 추가합니다.
                # ReArc 로더는 `_0` 접미사를 사용했는데, 단일 테스트 세트를 의미할 수 있습니다. 일관성을 위해 유지.
                keys_per_epoch[epoch_idx].append(f'{instance_base_key}_0')

        # 4. 최종 키 리스트 생성 및 셔플링
        final_keys_list = []
        for epoch_idx in range(len(keys_per_epoch)):
            if shuffle:
                keys_per_epoch[epoch_idx] = np.random.permutation(keys_per_epoch[epoch_idx]).tolist()
            final_keys_list.extend(keys_per_epoch[epoch_idx])
            
        return cls(keys=final_keys_list, challenge=challenge, solutions=solutions, is_orig=True)


    # loader for neoneye's format, as used in https://github.com/neoneye/arc-dataset-collection
    @classmethod
    def load_from_neoneye(cls, path):
        pattern = os.path.join(path, 'data', '*', '*.json')
        files = set(glob(pattern))
        for i in itertools.count():
            updated = [fn for fn in files if fn.endswith(f'_v{i + 1}.json')]
            if not updated: break
            for fn in updated:
                files.remove(fn.replace(f'_v{i + 1}.json', ('.json' if i == 1 else f'_v{i}.json')))
        assert len(files), f"No files found for pattern '{pattern}'."
        challenge = {}
        solutions = {}
        assert len(files), 'no files found'
        for fn in tqdm(files, desc=f"load dataset '{os.path.split(path)[-1]}'"):
            with open(fn) as f:
                key = cls.base_key_replace_invalid_chars(os.path.split(fn)[-1].replace('.json', ''))
                challenge[key] = json.load(f)
                solutions[key] = [test_case.pop('output') for test_case in challenge[key]['test']]
        return cls(challenge=challenge, solutions=solutions, is_orig=True)

    def change_keys(self, keys):
        return self.__class__(challenge=self.challenge, solutions=self.solutions, keys=keys)

    def split(self, n, split_seed, **kwargs):
        assert self.is_orig, 'Must be run on original dataset.'
        keys = sorted(self.challenge.keys())
        if split_seed == 'len':
            keys = self.sort_keys_by_len(keys=keys, **kwargs)
        else:
            assert isinstance(split_seed, int)
            assert not kwargs
            np.random.seed(split_seed)
            keys = np.random.permutation(keys)
        split_datasets = []
        for new_keys in np.array_split(keys, n):
            new_challenge = {k: self.challenge[k] for k in new_keys}
            split_datasets.append(self.__class__(challenge=new_challenge, solutions=self.solutions, is_orig=True))
        return split_datasets

    def remove_test_data(self):
        assert self.is_orig, 'Must be run on original dataset.'
        new_challenge = {k: {'train': v['train'], 'test': []} for k, v in self.challenge.items()}
        return self.__class__(challenge=new_challenge)

    @staticmethod
    def base_key_replace_invalid_chars(base_key):
        return base_key.replace('_', '-').replace('.', '-')

    @staticmethod
    def get_base_key_and_reply_num(key):
        key_num = key.split('.', 1)[0]
        base_key, reply_num = key_num.split('_') if '_' in key_num else (key_num, -1)
        return base_key, int(reply_num)

    @classmethod
    def get_base_key(cls, key):
        return cls.get_base_key_and_reply_num(key)[0]

    def grouped_keys(self):
        grouped_keys = OrderedDict()
        for key in self.keys:
            base_key, reply_num = self.get_base_key_and_reply_num(key)
            if base_key not in grouped_keys:
                grouped_keys[base_key] = []
            while len(grouped_keys[base_key])<=reply_num:
                grouped_keys[base_key].append([])
            grouped_keys[base_key][reply_num].append(key)
        return grouped_keys

    def move_test_to_train(self):
        assert self.is_orig, 'Must be run on original dataset.'
        new_challenge = {}
        for k, v in self.challenge.items():
            new_challenge[k] = {
                'train': v['train'] + [{**t, 'output': self.solutions[k][i]} for i, t in enumerate(v['test'])],
                'test': []
            }
        return self.__class__(challenge=new_challenge, is_orig=self.is_orig)

    @staticmethod
    def permute_array(a, descriptor, invert=False):
        permutation = [int(i) for i in descriptor if str(i).isdigit()]
        assert sorted(permutation) == list(range(10))
        a = np.asarray(a)
        assert a.ndim == 2
        if invert: permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
        return a

    @classmethod
    def transform_array(cls, array, transforms, apply_perm=True, invert=False):
        if array is None: return None
        array = np.asarray(array)
        if invert: transforms = transforms[::-1]
        for tf in transforms:
            if tf == 'tp':
                array = np.swapaxes(array, 0, 1)
            if tf == 'rt':
                array = np.rot90(np.rot90(np.rot90(array)) if invert else array)
            if apply_perm and tf.startswith('perm'):
                array = cls.permute_array(array, tf, invert=invert)
        return array

    @classmethod
    def fmt_array(cls, array, lines_sep, tf=None):
        if tf is not None:
            array = cls.transform_array(array, tf)
        return lines_sep.join(''.join(map(str, row)) for row in array)

    @classmethod
    def fmt_input(cls, array, query_beg, reply_beg, **kwargs):
        return query_beg + cls.fmt_array(array, **kwargs) + reply_beg

    @classmethod
    def fmt_output(cls, array, reply_end, **kwargs):
        return cls.fmt_array(array, **kwargs) + reply_end

    @classmethod
    def fmt_train(cls, train_ex, preprompt, query_beg, reply_beg, reply_end, **kwargs):
        examples = [cls.fmt_input(x['input'], query_beg, reply_beg, **kwargs) +
                    cls.fmt_output(x['output'], reply_end, **kwargs) for x in train_ex]
        return preprompt + ''.join(examples)

    def fmt_task(self, key, preprompt, query_beg, reply_beg, reply_end, reply=True, **kwargs):
        key_num, *tf = key.split('.')
        base_key, reply_num = self.get_base_key_and_reply_num(key_num)
        data_train = self.challenge[base_key]['train']
        data_query = self.challenge[base_key]['test']
        if reply is True:
            reply = self.solutions[base_key][reply_num] if base_key in self.solutions and reply_num >= 0 else None
        elif reply is not None:
            assert reply_num >= 0
        for t in tf:
            if t.startswith('ex'):
                data_train = [data_train[int(i)] for i in t[2:].split('-')]
        ret = dict(key=key)
        ret['train'] = self.fmt_train(data_train, preprompt, query_beg, reply_beg, reply_end, tf=tf, **kwargs)
        ret['query'] = self.fmt_input(data_query[reply_num]['input'], query_beg, reply_beg, tf=tf, **kwargs) if reply_num >= 0 else ''
        ret['input'] = ret['train'] + ret['query'] if reply_num >= 0 else ''
        if reply is not None:
            ret['reply'] = self.fmt_output(reply, reply_end, tf=tf, **kwargs)
        ret['text'] = ret['train'] + (ret['query'] + ret['reply'] if reply is not None else '')
        return ret

    def get_task(self, key, max_tokens=None, len_name=None, **kwargs):
        while True:
            fmt = self.fmt_task(key, **kwargs)
            if max_tokens is None or self.count_tokens(fmt[len_name]) <= max_tokens:
                break
            if not key.split('.')[-1].startswith('ex'):
                base_key = self.get_base_key(key)
                key = f"{key}.ex{'-'.join(map(str, range(len(self.challenge[base_key]['train']))))}"
            key_split = key.split('.')
            key_split[-1] = '-'.join(key_split[-1].split('-')[:-1])
            assert len(key_split[-1]) > 2 and key_split[-1].startswith('ex')
            key = '.'.join(key_split)
        return key, fmt

    def repeat(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        new_keys = []
        for i in range(n):
            new_keys.extend(self.keys if seed is None else np.random.permutation(self.keys))
        return self.change_keys(new_keys)

    @staticmethod
    def count_tokens(data, replace_special=re.compile('<[^<]*>')):
        replaced = replace_special.sub('x', data)  # replace '<...>' by a single char to count special tokens only once
        return len(replaced)

    @classmethod
    def max_new_tokens(cls, reply_end, lines_sep, max_size=30, safety_margin=1, **_):
        max_sized_reply = np.zeros([max_size, max_size], dtype=int)
        fmt = cls.fmt_output(max_sized_reply, reply_end=reply_end, lines_sep=lines_sep)
        return cls.count_tokens(fmt) + safety_margin

    def get_length(self, key, len_name, max_of_transposed=False, max_tokens=None, **fmt_opts):
        if not fmt_opts:
            fmt_opts = dict(preprompt='', query_beg='', reply_beg='', reply_end='', lines_sep='')
            length = self.count_tokens(self.fmt_task(key, **fmt_opts)[len_name])
        else:
            length = self.count_tokens(self.fmt_task(key, **fmt_opts)[len_name])
            if max_of_transposed:
                length = max(length, self.count_tokens(self.fmt_task(f'{key}.tp', fmt_opts)[len_name]))
            length += 1  # for bos token
        return length

    def sort_keys_by_len(self, keys, reverse=False, **kwargs):
        lengths = [(key, self.get_length(key, **kwargs)) for key in keys]
        return [x[0] for x in sorted(lengths, reverse=reverse, key=lambda x: x[1])]

    def sorted_by_len(self,**kwargs):
        return self.change_keys(self.sort_keys_by_len(self.keys, **kwargs))

    def convert_with_token_limit(self, **kwargs):
        out_list = []
        new_keys = []
        for key in tqdm(self.keys, desc='convert dataset'):
            key, fmt = self.get_task(key, **kwargs)
            new_keys.append(key)
            out_list.append(fmt)
        return out_list, self.change_keys(new_keys)

    def as_list(self, **kwargs):
        return self.convert_with_token_limit(**kwargs)[0]

    @staticmethod
    def rand_perm(n, sep=None, keep_zero=False):
        permutation = np.random.permutation(n).tolist()
        if keep_zero:
            permutation = [0] + [x for x in permutation if x != 0]
        return permutation if sep is None else sep.join(map(str, permutation))

    def augment_keys(self, keys, tp=False, rt=False, n=1, perm=False, keep_background=False, shfl_ex=False):
        keys = [k + n * '.tp' for n in range(2) for k in keys] if tp == 'all' else keys
        keys = [k + n * '.rt' for n in range(4) for k in keys] if rt == 'all' else keys
        keys = [k + bool(tp) * randint(0, 2) * '.tp' for k in keys] if tp != 'all' else keys
        keys = [k + bool(rt) * randint(0, 4) * '.rt' for k in keys] if rt != 'all' else keys
        keys = keys * n  # repeat n times
        keys = [k + bool(perm) * ('.perm' + self.rand_perm(10, '', keep_background)) for k in keys]
        n_ex = lambda k: len(self.challenge[self.get_base_key(k)]['train'])
        keys = [k + bool(shfl_ex) * ('.ex' + self.rand_perm(n_ex(k), '-')) for k in keys]
        return keys

    def augment(self, seed, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return self.change_keys([k for key in self.keys for k in self.augment_keys([key], **kwargs)])

    def decode(self, text, lines_sep, key=None):
        correct, info = None, 'unknown'
        try:
            data = [[int(x) for x in row if x.isdigit()] for row in text.split(lines_sep)]
            data = [row for row in data if len(row)]
            data = np.array(data, dtype=int)
            assert data.ndim == 2 and all(0 < x <= 30 for x in data.shape)
        except:
            data = None
            correct, info = False, 'cant_decode'
        if key is not None and data is not None:
            key_num, *transforms = key.split('.')
            base_key, reply_num = self.get_base_key_and_reply_num(key_num)
            data = self.transform_array(data, transforms, invert=True)
            correct_solution = self.solutions.get(base_key)
            if correct_solution is None:
                info = 'sol_unknown'
            else:
                correct_solution = np.asarray(correct_solution[reply_num])
                if np.array_equal(correct_solution, data):
                    correct, info = True, 'ALL_CORRECT'
                else:
                    correct, info = False, ('bad_content' if correct_solution.shape == data.shape else 'bad_xy_size')
        return data, correct, info

    def get_submission(self, results=None):
        assert self.is_orig, 'Must be run on original dataset.'
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(v['test']))] for k, v in self.challenge.items()}
        if results is not None:
            self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        for base_key, data in results.items():
            for reply_num, guesses in enumerate(data):
                target_dict = submission[base_key][reply_num]
                for i, g in enumerate(guesses[:len(target_dict)]):
                    target_dict[f'attempt_{i + 1}'] = g['output'].tolist()

    def validate_submission(self, submission):
        assert self.is_orig, 'Must be run on original dataset.'
        assert self.solutions, 'Solutions must be loaded for submission verification.'
        score = 0
        for k, v in self.solutions.items():
            for i, r in enumerate(v):
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score
