import os
import importlib.util
import numpy as np
import sys
import time
import multiprocessing as mp

def run_with_timeout(func, args, timeout=0.3):
    """func(arg)를 timeout 초 안에 실행. 실패하면 None 반환"""
    def target(q):
        try:
            result = func(*args)
            q.put(result)
        except Exception as e:
            q.put(e)

    q = mp.Queue()
    p = mp.Process(target=target, args=(q,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None  # 타임아웃으로 실패

    if not q.empty():
        result = q.get()
        if isinstance(result, Exception):
            return None
        return result
    return None

# Predict + when confident, using common functions
def common_predict_confident(task_data, test_input, task_id):
    """
    - task_data: {
        "train": [ {"input": 2D-list, "output": 2D-list}, … ],
        "test":  [ {"input": 2D-list}, … ]
      }
    - test_input:  np.ndarray (또는 2D-list)
    - task_id:    int (디버깅용 ID)

    반환값: [np.ndarray] 또는 빈 리스트([])
    """
    debug = False
    def dbg_print(string_input):
        if (debug):
            print(string_input, flush = True)

    # 1) other_functions 폴더 경로
    base_dir = os.path.dirname(__file__)
    solvers_dir = os.path.join(base_dir, "gencode_merged")

    # 2) common.py를 “common” 모듈로 미리 로드해두기
    #    (solver 모듈 내의 `from common import *`를 가능하게 함)
    common_path = os.path.join(solvers_dir, "common.py")
    if os.path.isfile(common_path):
        spec_common = importlib.util.spec_from_file_location("common", common_path)
        if spec_common and spec_common.loader:
            common_mod = importlib.util.module_from_spec(spec_common)
            spec_common.loader.exec_module(common_mod)
            sys.modules["common"] = common_mod

    # 3) gencode 폴더 내부의 .py 파일 순회
    for fname in os.listdir(solvers_dir):
        dbg_print(f"[{task_id}] Loading {fname}")
        if not fname.endswith(".py"):
            continue
        if fname == "common.py":
            continue

        module_path = os.path.join(solvers_dir, fname)
        module_name = fname[:-3]  # 확장자(.py) 제거

        # 4) 동적으로 모듈 로드
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # 5) main 함수 가져오기
        solver_fn = getattr(mod, "main", None)
        if not callable(solver_fn):
            continue
        
        dbg_print(f"[{task_id}] Got main functions for {fname}")

        # 6) 해당 solver_fn이 모든 train을 맞추는지 확인
        solved_all = True
        for train_idx, train_task in enumerate(task_data["train"]):
            x_tr = np.asarray(train_task["input"])
            y_tr = np.asarray(train_task["output"])
            dbg_print(f"[{task_id}] x_tr & y_tr generated")

            try:
                pred = run_with_timeout(solver_fn, (x_tr, task_data), timeout=0.3)
                if pred is None:
                    solved_all = False
                    break
                dbg_print(f"[{task_id}] prediction generated")
            except Exception:
                solved_all = False
                break

            # 출력 형태와 값이 정확히 일치하는지 비교
            if not (isinstance(pred, np.ndarray)
                    and pred.shape == y_tr.shape
                    and np.array_equal(pred, y_tr)):
                solved_all = False
                break

        if solved_all:
            return run_with_timeout(solver_fn, (test_input, task_data), timeout=0.3)

    # 8) 어떤 모듈도 모든 train을 맞히지 못했을 때
    return []

# 최종적인 run 함수
# 최종적인 run 함수
def CPU_solver(task):
    for i in range(len(task['test'])): # 이건 사실상 그냥 1임
        # print(len(task['test']))
        start_time = time.time()
        test_input = np.array(task['test'][i]['input']) # 들어가는 input, 2D list

        result = common_predict_confident(task, test_input, i)
        print(f"CPU result: {result}")
        print(f"CPU Generated time: {round(time.time() - start_time, 2)}")

        return result