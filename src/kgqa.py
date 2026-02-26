import sys
import os
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from config import LLM_BASE
import json
import random
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils.utils import run_llm, get_timestamp, readjson
from utils.freebase_func import *
import time
from tqdm import tqdm
from utils import *
from config import *
from kg_instantiation import *
import tiktoken

PROMPT_PATH = "prompt/kgqa"

def parse_args():
    parser = ArgumentParser("KGQA for cwq or WebQSP")
    parser.add_argument("--full", action="store_true", help="full dataset.")
    parser.add_argument("--verbose", action="store_true", help="verbose or not.", default=False)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_token", type=int, default=1024)
    parser.add_argument("--max_token_reasoning", type=int, default=2048)
    parser.add_argument("--max_que", type=int, default=200)
    parser.add_argument("--dataset", type=str, required=True, help="choose the dataset.choices={\"cwq\", \"WebQSP\"}")
    parser.add_argument("--llm", type=str, choices=LLM_BASE.keys(), default="gpt35", help="base LLM model.")
    parser.add_argument("--openai_api_keys", type=str, help="openai_api_keys", default="", required=True)
    parser.add_argument("--count_token_cost", type=bool, help="count_token_cost", default=False)
    parser.add_argument("--initial_path_eval", type=bool, help="evaluate initial reasoning path (ablation study)", default=False)
    # 新增：断点续跑相关参数
    parser.add_argument("--resume", action="store_true", help="Whether to resume running from the breakpoint")
    parser.add_argument("--resume_file", type=str, default="", help="Output file path for continued run (must be consistent with the previous output_file)")
    
    args = parser.parse_args()
    args.LLM_type = LLM_BASE[args.llm]
    return args


def question_process(fpath):
    if fpath.endswith('jsonl'):
        data = read_jsonl(fpath)
    else:
        data = readjson(fpath)

    return data


def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string.  For calculating token cost."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def LLM_edit(reasoning_path_LLM_init, entity_label, feedback, question, options, input_token_cnt=0, output_token_cnt=0, llm_call_count=0):
    """
    reasoning path editing

    Args:
        reasoning_path_LLM_init : previous reasoning path
        entity_label : topic entity
        feedback : prepared error message for editing
        question 
        options 
        input_token_cnt 
        output_token_cnt
        llm_call_count: 累计大模型调用次数

    Returns:
        reasoning_path_LLM_init : edited reasoning path
        thought : LLM CoT
        input_token_cnt, output_token_cnt : to calculate token cost
        llm_call_count: 累计大模型调用次数
    """
    init_path = reasoning_path_LLM_init[entity_label]
    err_msg, grounded_know_string, candidate_rel = feedback

    prompt = open(
        os.path.join(PROMPT_PATH, f"{options.dataset}_edit.md"),
        'r', encoding='utf-8'
    ).read()

    prompts = prompt + "Question: " + question + "\nInitial Path: " + str(init_path) + "\n>>>> Error Message\n" + err_msg + ">>>> Instantiation Context\nInstantiate Paths:" + grounded_know_string +"\nCandidate Relations:" + str(candidate_rel)  + "\n>>>> Corrected Path\nGoal: "

    for _ in range(MAX_LLM_RETRY_TIME):
        try:
            response = run_llm(prompts, temperature=options.temperature, max_tokens=options.max_token, openai_api_keys=options.openai_api_keys, engine=options.LLM_type)

            # 统计大模型调用次数
            llm_call_count += 1

            new_path = response.split("Final Path:")[-1].strip().strip("\"").strip()
            thought = response
            if entity_label not in new_path or "->" not in new_path:
                raise ValueError("entity_label or -> is not in path")
            
            if new_path == init_path:
                raise ValueError("no changing origin plan")

            if "->" not in new_path or entity_label not in new_path:
                raise ValueError("output empty plan or without the starting point")

            elements = new_path.split(" -> ")
            if len(list(set(elements))) < len(elements):
                raise ValueError("same relation in path")
            
            if len(elements) > 5:
                raise ValueError("path too long!!!!")

            reasoning_path_LLM_init[entity_label] = new_path
            break
        except Exception as e:
            if options.verbose:
                error_line = "*" * 40
                print(error_line)
                print(e)
                print("---------- new path -----------:", new_path)
                print(error_line)
                print()
            time.sleep(0.3)
            
    if options.count_token_cost:
        input_token_cnt += num_tokens_from_string(prompt)
        output_token_cnt += num_tokens_from_string(response)
    
    return reasoning_path_LLM_init, thought, input_token_cnt, output_token_cnt, llm_call_count


def get_init_reasoning_path(question, topic_ent, options, input_token_cnt=0, output_token_cnt=0, llm_call_count=0):
    """generate initial reasoning path
    核心优化：
    1. 替换eval()为json.loads()，安全解析JSON
    2. 严格提取Path:后的纯JSON内容，过滤自然语言
    3. 增加多层异常兜底，避免单条Question阻塞整体流程
    4. 适配Prompt的强制JSON格式规则

    Args:
        llm_call_count: 累计大模型调用次数
        question: 输入问题文本
        topic_ent : 问题中的主题实体列表
        options : 解析后的命令行参数
        input_token_cnt : 累计输入token数（用于计费）
        output_token_cnt : 累计输出token数（用于计费）

    Returns:
        init_reasoning_path: 初始推理路径字典
        input_token_cnt: 更新后的输入token数
        output_token_cnt: 更新后的输出token数
        llm_call_count: 累计大模型调用次数
    """
    # 读取Prompt模板（已包含强制JSON格式规则）
    prompt = open(
        os.path.join(PROMPT_PATH, f"{options.dataset}_init.md"),
        'r', encoding='utf-8'
    ).read()

    # 默认空路径（兜底：解析失败时使用）
    default_relation_path = {
        k: k  # 路径默认值为实体名本身
        for k in topic_ent
    }

    # 拼接完整Prompt（问题+主题实体）
    prompt += "Question: " + question + "\nTopic Entities:" + str(topic_ent) + "\nThought:"
    init_reasoning_path = default_relation_path  # 初始化路径为默认值

    for _ in range(MAX_LLM_RETRY_TIME):
        try:
            # 调用LLM生成推理路径
            response = run_llm(
                prompt,
                options.temperature,
                options.max_token_reasoning,
                options.openai_api_keys,
                options.LLM_type
            )

            # 统计大模型调用次数
            llm_call_count += 1

            # ========== 核心步骤1：严格提取Path:后的内容 ==========
            # 分割Path:，只取最后一部分（避免LLM返回多个Path:字段）
            path_raw = response.split("Path:")[-1].strip()
            # 过滤空内容/纯空格
            if not path_raw:
                raise ValueError("There is no valid content after 'Path:'")

            # ========== 核心步骤2：安全解析JSON（替代危险的eval()） ==========
            # 用json.loads解析，而非eval，避免代码执行风险和语法错误
            reponse_dict = json.loads(path_raw)

            # ========== 核心步骤3：校验并赋值推理路径 ==========
            # 遍历JSON字典，提取每个实体的第一条路径（保持原有逻辑）
            for k, v in reponse_dict.items():
                # 严格校验值类型：必须是列表且非空
                if isinstance(v, list) and len(v) > 0:
                    init_reasoning_path[k] = v[0]
                    # 最终校验：结果必须是字典
            assert isinstance(init_reasoning_path, dict), "The inference path must be of dictionary type "
            break  # 解析成功，退出重试循环

        # ========== 异常兜底：覆盖所有可能的解析失败场景 ==========
        except json.JSONDecodeError as e:
            # JSON格式错误（如括号不匹配、引号未转义）
            init_reasoning_path = default_relation_path
            print(f"JSON parsing failed: {e}")
            error_line = "*" * 40
            print(f"LLM returns the original content (first 500 characters): {response[:500]}")
            time.sleep(0.3)
            print(error_line)

        except AssertionError as e:
            # 类型校验失败（非字典）
            init_reasoning_path = default_relation_path
            print(f"Path type error: {e}")
            error_line = "*" * 40
            print(f"Path format error: {e}")
            time.sleep(0.3)
            print(error_line)

        except Exception as e:
            # 其他所有异常（如分割失败、键缺失等）
            init_reasoning_path = default_relation_path
            print(f"Unknown parsing error: {e}")
            error_line = "*" * 40
            print(f"LLM returns the original content: {response}")
            time.sleep(0.3)
            print(error_line)

    # ========== Token计费（保持原有逻辑） ==========
    if options.count_token_cost:
        input_token_cnt += num_tokens_from_string(prompt)
        output_token_cnt += num_tokens_from_string(response)

    return init_reasoning_path, input_token_cnt, output_token_cnt, llm_call_count


def llm_reasoning(reasoning_paths_instances, question, options, llm_call_count=0):
    """call llm for QA reasoning"""
    kg_instances_str = ""
    kg_triple_set = []
    response = ""
    
    prompt = open(
        os.path.join(PROMPT_PATH, f"kgqa_reasoning.md"),
        'r', encoding='utf-8'
    ).read()
    
    cot_prompt = open(
        os.path.join(PROMPT_PATH, "cot_reasoning.md"),
        'r', encoding='utf-8'
    ).read()
    
    for lines in reasoning_paths_instances:
        for l in lines:
            triple = [l[0] , l[1] , l[2]]
            if triple not in kg_triple_set:
                kg_triple_set.append(triple)

    for l in kg_triple_set:
        kg_instances_str += "(" + l[0] + ", " + l[1] + ", " + l[2] + " )\n"
    kg_instances_str = kg_instances_str.strip("\n")   
    
    if len(kg_instances_str) > 0:
        prompts = prompt + "Q: " + question + "\nKnowledge Triplets: " + kg_instances_str + "\nA: "
        for _ in range(MAX_LLM_RETRY_TIME):
            try:
                response = run_llm(prompts, options.temperature, options.max_token, options.openai_api_keys, options.LLM_type)
                # 统计大模型调用次数
                llm_call_count += 1
                if len(response) == 0:
                    print(f"\n{'*'*10} Empty Results {'*'*10}")
                    print("Q: " + question)
                    print("*"*30 + '\n')
                    continue

                if "{" not in response or "}" not in response:
                    print(f"\n{'*'*10} Invalid Results {'*'*10}")
                    print(response)
                    print()
                    continue
                else:
                    break
            except Exception as e:
                continue
    
    # use internal knowledge if failed too many times
    if "{" not in response or "}" not in response or len(kg_instances_str) == 0 or len(response) == 0:
        prompts = cot_prompt + "Q: " + question + "\nA: "
        response = run_llm(prompts, options.temperature, options.max_token, options.openai_api_keys, options.LLM_type)

    return response, llm_call_count

def check_string(string):
    return "{" in string

def clean_results(string):
    """
    Extract result from LLM output.

    Args:
        string : LLM output

    Returns:
        extracted result
    """
    if "{" in string:
        start = string.find("{") + 1
        end = string.find("}")
        content = string[start:end]
        return content
    else:
        return "NULL"
   
def hit1(response, answers):
    clean_result = response.strip().replace(" ","").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ","").lower()
        # the line below is used by ToG
        # if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
        if clean_result == clean_answer or clean_answer in clean_result:
            return True
    return False 

def evaluate(results, ground_truth):
    """return hit"""
    hit = 0
    if check_string(results):
        response = clean_results(results)
        if type(response) != str:
            response=""
        if response=="NULL":
            response = results
        else:
            if response != "" and hit1(response, ground_truth):
                hit=1
    else:
        response = results
        if type(response) != str:
            response = ""
        if response != "" and hit1(response, ground_truth):
            hit = 1
    return hit

def check_ending(result_paths, grounded_knowledge_current, ungrounded_neighbor_relation_dict, reasoning_path_LLM_init, entity_label, question, options):
    """
    Check if we need to edit the reasoning path.
    If so, prepare the feedback from instantiation information.

    Args:
        result_paths : KG instances
        grounded_knowledge_current : stores all instances during BFS (length starting from 0)
        ungrounded_neighbor_relation_dict : if instantiation fails, this store some relations as candidates for editing
        reasoning_path_LLM_init : previous reasoning path from each topic entity
        entity_label : topic entity
        question 
        options

    Returns:
        max_path_len : length for the longest instance
        End_loop_cur_path: whether we need to edit the reasoning path
        (err_msg, grounded_know_string, candidate_rel) : prepared feedback for editing
    """
    
    max_path_len = grounded_knowledge_current[-1][-1]
    init_path = reasoning_path_LLM_init[entity_label]
    grounded_know = []
    ungrounded_cand_rel = {}
    max_grounded_len = 0
    cvt_ending = False
    
    if options.verbose:
        print("max len of grounded knowledge current: ", max_path_len)
    End_loop_cur_path = True

    # check if anything goes wrong and get the reasoning
    if len(result_paths) > 0:
        if max_path_len == 0:
            End_loop_cur_path = False

        for path in result_paths:
            if len(path) < max_path_len:
                continue
            # check if cvt node ending for Freebase (m. g.)
            if path[-1][-1].startswith("m.") or path[-1][-1].startswith("g."):
                End_loop_cur_path=False
                break
    else:
        End_loop_cur_path = False

    if len(grounded_knowledge_current) > 0:
        max_grounded_len = grounded_knowledge_current[-1][-1]

    # process grounded knowledge (some relations might be instantiated successfully)  - code below can be optional if we do not need to edit the path
    for know in grounded_knowledge_current:
        node_label = utils.id2entity_name_or_type_en(know[0])
        
        if node_label.startswith("m.") == False and know[2] != 0 and know[2] == max_grounded_len:
            if node_label in ungrounded_neighbor_relation_dict.keys():
                ungrounded_cand_rel[node_label] = ungrounded_neighbor_relation_dict[node_label]
            grounded_know.append(know[1])
            
        if know[2] == max_grounded_len and node_label.startswith("m."):
            cvt_ending = True

    # 15 is a hyperparameter to prevent too much knowledge in LLM context
    grounded_know = grounded_know if len(grounded_know) <= 15 else random.sample(grounded_know, 15)

    # process all cvt nodes into <cvt></cvt>
    cvt_know = [(i[0], i[1]) for i in grounded_knowledge_current if utils.id2entity_name_or_type_en(i[0]).startswith("m.") and len(i[1])>0 and i[2]==max_grounded_len]
    # cvt_know = list(set(cvt_know))
    # 10 is a hyperparameter to prevent too much knowledge in LLM context
    cvt_know = cvt_know if len(cvt_know) <= 10 else random.sample(cvt_know, 10)

    for cvt in cvt_know:
        if cvt[0] in ungrounded_neighbor_relation_dict.keys():
            # the label of a cvt node is the original mid
            ungrounded_cand_rel[cvt[0]] = ungrounded_neighbor_relation_dict[cvt[0]]
        grounded_know.append(cvt[1])

    grounded_know = [" -> ".join([i if not i.startswith("m.") else "<cvt></cvt>" for i in utils.path_to_string(knowledge).split(" -> ")]) for knowledge in grounded_know]
    grounded_know = list(set(grounded_know))
    grounded_know_string = "\n".join(grounded_know)

    if len(grounded_know) == 0 and len(ungrounded_neighbor_relation_dict) > 0:
        ungrounded_cand_rel = ungrounded_neighbor_relation_dict

    # prepare candidate relations
    candidate_rel = []
    for k, v in ungrounded_cand_rel.items():
        candidate_rel.extend(v)
    candidate_rel = list(set(candidate_rel))

    # filter similar relations as candidates. 35 is a hyperparameter to prevent too much knowledge in LLM context
    candidate_rel = candidate_rel if len(candidate_rel) <= 35 else utils.similar_search_list(question, candidate_rel, options)[:35]
    candidate_rel.sort()

    # prepare error message for editing
    err_msg_list = []
    if cvt_ending:
        err_msg_list.append("<cvt></cvt> in the end.")
    if "->" not in init_path:
        err_msg_list.append("Empty Initial Path.")
    else:
        relation_elements = init_path.split(" -> ")[1:]
        if max_grounded_len < len(relation_elements):
            ungrounded_relation = relation_elements[max_grounded_len]
            err_msg_list.append(f"relation \"{ungrounded_relation}\" not instantiated.")
            
    err_msg = ""
    for index, msg in enumerate(err_msg_list):
        err_msg += str(index+1)+". "+ msg +"\n"

    return max_path_len, End_loop_cur_path, (err_msg, grounded_know_string, candidate_rel)


def merge_different_path(grounded_revised_knowledge, reasoning_paths, options):
    """
    Merge different paths instances from different topic entities.
    For instances from each topic entity, we first take all entities in these instances and calculate the intersection of these entities.
    If the intersection is not empty, we retain all instances containing these intersected entities for instances from each topic entities.
    For example, for the question "What country bordering France contains an airport that serves Nijmegen?", we have instances from "France" and "Nijmegen".
    We take all entities from "France" and "Nijmegen" and calculate that the intersection is "German".
    Then, we retain all path instances for "France" and "Nijmegen" containing "German".

    Moreover, if one path instances contains too much instances (more than 50), we remove some these instances, because they might not be useful for LLM's QA reasoning.

    Args:
        grounded_revised_knowledge : knowledge instances from each topic entity
        reasoning_paths : all knowledge instances (cumulated)
        options

    Returns:
        merged path instances
    """
    """
        合并来自不同主题实体的路径实例。
        对于每个主题实体对应的实例，我们首先提取这些实例中包含的所有实体，并计算这些实体集合的交集。
        若交集非空，则保留所有包含这些交集实体的、各主题实体对应的实例。
        示例：针对问题「Which country bordering France has an airport serving Nijmegen?」（与法国接壤且有机场服务奈梅亨的国家是哪个？），
        我们有分别来自「France（法国）」和「Nijmegen（奈梅亨）」的实例；
        提取这两个实体对应实例中的所有实体后，计算得出交集为「German（德国）」；
        随后保留所有包含「German」的、来自「France」和「Nijmegen」的路径实例。

        此外，若某一主题实体对应的路径实例数量过多（超过50个），我们会移除部分实例——因为这类冗余实例可能对大语言模型（LLM）的问答（QA）推理无实际帮助。

        参数:
            grounded_revised_knowledge : 字典，键为主题实体，值为该主题实体对应的知识实例集合
            reasoning_paths : 所有主题实体的知识实例汇总列表（累计值）
            options : 全局配置参数（包含verbose、阈值等配置）

        返回值:
            合并/过滤后的路径实例列表
        """
    # 若开启verbose模式，打印合并流程的分隔符（调试用，标识进入路径合并阶段）
    if options.verbose:
        print("**********************merge*****************************")

    # ========== 初始化统计变量 ==========
    # entity_sets: 字典，key=主题实体名称，value=该实体对应的KG实例中包含的所有实体集合（用于计算实体交集）
    entity_sets = {}
    # knowledeg_len_dict: 字典（注：变量名存在笔误，应为knowledge_len_dict），key=主题实体名称，value=该实体的KG实例总数量（用于过滤过多实例）
    knowledeg_len_dict = {}

    # ========== 遍历每个主题实体，统计关联实体和实例数量 ==========
    # 遍历grounded_revised_knowledge：key=主题实体，value=该实体的所有KG实例路径
    for topic_entity, grounded_knowledge in grounded_revised_knowledge.items():
        # 初始化当前主题实体的实体集合（首次遍历该实体时执行）
        if not topic_entity in entity_sets.keys():
            entity_sets[topic_entity] = set()

        # 初始化当前主题实体的KG实例计数
        knowledge_len = 0
        # 遍历当前主题实体的每一组KG实例路径
        for paths in grounded_knowledge:
            # 累计当前主题实体的实例总数（每组路径的长度=该组内实例数）
            knowledge_len += len(paths)
            # 遍历路径中的每个三元组 (头实体, 关系, 尾实体)
            for triples in paths:
                # 将三元组的头实体、尾实体加入当前主题实体的实体集合
                entity_sets[topic_entity].add(triples[0])
                entity_sets[topic_entity].add(triples[2])
        # 保存当前主题实体的实例总数到字典
        knowledeg_len_dict[topic_entity] = knowledge_len

    # ========== 计算多主题实体的实体交集，过滤路径 ==========
    # intersec_set: 存储所有主题实体的实体交集，初始化为空字符串（用于首次赋值）
    intersec_set = ""
    # 遍历每个主题实体对应的实体集合
    for topic_entity, entities_in_knowledge in entity_sets.items():
        # 首次遍历：将空字符串替换为第一个主题实体的实体集合（初始化交集）
        if type(intersec_set) == str:
            intersec_set = entities_in_knowledge
        # 非首次遍历：计算当前交集与当前主题实体集合的交集（保留共同实体）
        else:
            intersec_set = intersec_set.intersection(entities_in_knowledge)

            # ---------- 有实体交集：保留包含交集实体的路径 ----------
            if len(intersec_set) > 0:
                # 初始化新的路径列表（仅保留包含交集实体的路径）
                new_reasoning_paths = []
                lists_of_paths = []
                # 遍历所有累计的KG实例路径
                for path in reasoning_paths:
                    # 检查当前路径是否包含任意一个交集实体
                    for i in intersec_set:
                        if i in str(path):
                            # 包含交集实体则加入新路径列表
                            new_reasoning_paths.append(path)
                            # 记录路径的字符串形式（用于后续去重，此处未实际去重仅记录）
                            lists_of_paths.append(utils.path_to_string(path))
                # 更新推理路径为过滤后的新列表（仅保留含交集实体的路径）
                reasoning_paths = new_reasoning_paths

            # ---------- 无实体交集：过滤过多的路径实例 ----------
            else:
                # 仅当总路径数超过30时执行过滤（避免路径过多影响LLM推理）
                if len(reasoning_paths) > 30:
                    # 遍历每个主题实体的实例数量
                    for k, v in knowledeg_len_dict.items():
                        # 若该主题实体的实例数超过50（认为是冗余实例）
                        if v > 50:
                            # 初始化新的路径列表（过滤该主题实体开头的路径）
                            new_reasoning_paths = []
                            lists_of_paths = []
                            # 遍历所有累计的KG实例路径
                            for path in reasoning_paths:
                                # 将路径转为字符串并去除多余符号（[、(、'、"等）
                                string_path = str(path).strip("[").strip("(").strip("\'").strip("\"").strip()
                                # 过滤掉以当前主题实体开头的路径（减少冗余）
                                if not string_path.startswith(k):
                                    new_reasoning_paths.append(path)
                                    lists_of_paths.append(utils.path_to_string(path))
                            # 更新推理路径为过滤后的新列表
                            reasoning_paths = new_reasoning_paths

    # 返回合并/过滤后的最终KG实例路径列表
    return reasoning_paths


def main():
    # ========== 初始化基础配置与文件路径 ==========
    # 根据选择的LLM名称（如gpt35），从配置中获取对应的LLM类型（API调用标识）
    options.LLM_type = LLM_BASE[options.llm]
    # 根据数据集名称（cwq/WebQSP）获取对应的输入数据文件路径
    input_file = get_dataset_file(options.dataset)
    # 拼接输出文件路径：OUTPUT_FILE_PATH/KGQA/数据集_LLM模型_时间戳.jsonl（保证文件名唯一）
    output_file = os.path.join(OUTPUT_FILE_PATH, f"KGQA/{options.dataset}_{options.llm}_{get_timestamp()}.jsonl")
    # 获取数据集中存储"问题文本"的字段名（不同数据集字段名可能不同，如cwq是"question"，WebQSP是"utterance"）
    question_string = get_question_string(options.dataset)
    # 加载数据集：根据文件后缀（json/jsonl）调用对应读取函数
    dataset = question_process(input_file)

    # ========== 评估指标初始化 ==========
    # metrics字典存储核心评估指标：
    # hit: 每个样本的命中结果（1=命中，0=未命中）
    # token_cost: 每个样本的token费用（美元）
    # init_path_hit: 初始推理路径的命中结果（消融实验用）
    metrics = {
        'hit': [],
        'token_cost': [],
        'init_path_hit': []
    }

    total_token_cost = 0.0

    # 空判断（无实际功能，疑似遗留代码，保留原逻辑）
    if not options.full:
        dataset = dataset

    # ========== 断点续跑核心逻辑 ==========
    # 初始化续跑起始索引：默认从第0个样本开始
    start_index = 0
    # 如果开启续跑模式（--resume参数）
    if options.resume:
        # 校验：续跑必须指定上次的输出文件路径（--resume_file）
        if not options.resume_file:
            raise ValueError("Resuming mode requires specifying the last output file path via --resume_file")
        # 如果续跑文件存在，读取已处理的样本数（文件行数=已处理样本数）
        if os.path.exists(options.resume_file):
            with open(options.resume_file, 'r', encoding='utf-8') as f:
                # 统计文件行数，作为已处理样本数
                lines = f.readlines()
                start_index = len(lines)

            # 新增：累加已处理样本的hit, token_cost
            print(f"Starting to count tokens and costs for {start_index} processed samples...")
            for line in tqdm(lines, desc="Counting tokens and costs for processed samples"):
                try:
                    sample_data = json.loads(line.strip())
                    # 累加命中结果（hit）
                    metrics['hit'].append(sample_data.get("hit", 0))
                    # 累加token费用（token_cost）
                    metrics['token_cost'].append(sample_data.get("token_cost", 0))
                    # 累加初始路径命中结果（init_path_hit）
                    metrics['init_path_hit'].append(sample_data.get("init_path_hit", 0))
                    # 累加token费用（token_cost）
                    total_token_cost += sample_data.get("token_cost", 0)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {lines.index(line) + 1} data: {e}. Skipping this sample.")
                    continue

            # 新增：打印历史统计结果
            print(f"Processed sample statistics completed:")
            print(f"  - history avg init_path_hit: {np.mean(metrics['init_path_hit']):.4f}")
            print(f"  - history avg hit: {np.mean(metrics['hit']):.4f}")
            print(f"  - history total token cost: ${total_token_cost:.6f}")
            print(f"Detected resuming file {options.resume_file}, processed {start_index} samples, will start from sample {start_index}")
        else:
            # 续跑文件不存在，提示并从头开始
            print(f"Resuming file {options.resume_file} does not exist, will start from the beginning")

        # 续跑时强制将输出文件改为指定的resume_file（保证数据追加到同一文件）
        output_file = options.resume_file

    # 打印最终输出文件路径，确认存储位置
    print("save output file to: ", output_file)
    print('*' * 50)
    print('*' * 50)

    # 确定文件打开模式：续跑用a+（追加），新跑用w+（覆盖）
    open_mode = 'a+' if options.resume else 'w+'
    # 打开输出文件（指定编码为utf-8，避免中文/特殊字符乱码）
    f = open(output_file, open_mode, encoding='utf-8')

    # ========== 遍历数据集处理每个样本 ==========
    # tqdm包裹数据集，显示处理进度条；enumerate获取样本索引和内容
    for index, item in enumerate(tqdm(dataset)):
        # 续跑时跳过已处理的样本（索引小于start_index的样本）
        if index < start_index:
            continue

        # 提取当前样本的主题实体列表（如["France", "Nijmegen"]）
        topic_ent_list = get_topic_entity_list(item, input_file)
        # 提取当前样本的主题实体字典（key=实体ID，value=实体标签/名称）
        topic_ent_dict = get_topic_entity_dict(item, input_file)

        # 跳过无主题实体的样本（主题实体是KGQA的核心，无实体无法推理）
        if topic_ent_list == []:
            print(f"Sample {index}: topic entity is empty, sample: {item}")
            # 提示：无主题实体的样本需确认是否继续，此处直接跳过
            continue

        # 初始化当前样本的LLM调用计数器
        llm_call_count = 0

        # 初始化LLM输入/输出token计数（用于计算API费用）
        input_token_cnt = 0
        output_token_cnt = 0
        token_cost = 0

        # 格式化问题文本：确保问题以问号结尾（统一格式，避免LLM解析异常）
        question = item[question_string] if not item[question_string].endswith('?') else item[question_string] + '?'
        # 提取当前样本的真实答案（ground truth），用于后续评估
        ground_truth = get_ground_truth(item, options.dataset)

        # ========== 生成初始推理路径 ==========
        # 调用LLM生成基于主题实体的初始推理路径，同时更新token计数
        init_reasoning_path, input_token_cnt, output_token_cnt, llm_call_count = get_init_reasoning_path(question, topic_ent_list,
                                                                                         options, input_token_cnt,
                                                                                         output_token_cnt, llm_call_count)

        # 若开启verbose模式，打印当前问题和初始推理路径（调试用）
        if options.verbose:
            print(f"Question:{question}")
            print(f"init reasoning path: {init_reasoning_path}")

        # 初始化推理路径编辑次数（迭代优化的次数）
        refine = 0

        # ========== 初始化KG实例化相关变量 ==========
        # knowledge_instance_final: 存储每个主题实体最终的KG实例路径
        knowledge_instance_final = dict.fromkeys(topic_ent_list, [])
        # len_of_grounded_knowledge: 每个主题实体的路径实例化成功的长度
        len_of_grounded_knowledge = dict.fromkeys(topic_ent_list, [])
        # len_of_predict_knowledge: 每个主题实体的预测路径长度（推理路径的关系数）
        len_of_predict_knowledge = dict.fromkeys(topic_ent_list, [])
        # reasoning_paths: 存储当前样本所有主题实体的KG实例路径（合并前）
        reasoning_paths = []
        # thought: 存储LLM编辑推理路径时的思维链（CoT）
        thought = ""
        # lists_of_paths: 存储路径的字符串形式，用于去重
        lists_of_paths = []
        # predict_path: 存储每个主题实体的预测路径（编辑前后）
        predict_path = dict.fromkeys(topic_ent_list, [])

        # ========== 遍历每个主题实体，进行路径实例化与编辑 ==========
        # 遍历主题实体字典（entity_id=实体ID，entity_label=实体名称）
        for entity_id, entity_label in topic_ent_dict.items():
            # 为每个主题实体单独初始化编辑次数
            entity_refine = 0

            # 跳过不在初始推理路径中的实体（避免无效处理）
            if entity_label not in init_reasoning_path.keys():
                continue

            # 若开启verbose模式，打印当前处理的主题实体（调试用）
            if options.verbose:
                print("Topic entity: ", entity_label)

            # 迭代编辑推理路径（最多MAX_REFINE_TIME次，避免无限循环）
            while entity_refine < MAX_REFINE_TIME:
                # ========== 关系绑定：为推理路径中的关系匹配KG中的真实关系 ==========
                # topk=5：为每个关系取前5个候选绑定关系（提升实例化成功率）
                binded_relations = relation_binding(init_reasoning_path, topk=5)
                # 将当前实体的推理路径字符串转为数组（如"France -> border -> country" → ["France", "border", "country"]）
                relation_path_array = utils.string_to_path(init_reasoning_path[entity_label])
                # 为路径中的每个关系匹配候选绑定关系，生成有序候选列表
                sequential_relation_candidates = [binded_relations[relation] for relation in relation_path_array]

                # ========== BFS路径实例化：在KG中查找匹配的实体路径 ==========
                # entity_id: 起始实体ID；relation_path_array: 推理路径；sequential_relation_candidates: 候选关系
                # 返回值：
                # result_paths: 实例化成功的路径列表；grounded_knowledge_current: BFS过程中所有实例化知识；ungrounded_neighbor_relation_dict: 未实例化的邻居关系（用于编辑）
                result_paths, grounded_knowledge_current, ungrounded_neighbor_relation_dict = bfs_for_each_path(
                    entity_id, relation_path_array, sequential_relation_candidates, options, options.max_que)

                # ========== 检查是否需要编辑推理路径 ==========
                # 输入：实例化结果、当前推理路径、问题等；输出：最大路径长度、是否停止编辑、编辑反馈（错误信息+上下文+候选关系）
                max_path_len, end_refine, feedback = check_ending(result_paths, grounded_knowledge_current,
                                                                  ungrounded_neighbor_relation_dict,
                                                                  init_reasoning_path, entity_label, question, options)

                # ========== 记录路径长度信息（分析用） ==========
                # 记录预测路径的长度（关系数=路径元素数-1）
                len_of_predict_knowledge[entity_label].append(len(init_reasoning_path[entity_label].split("->")) - 1)
                # 记录实例化成功的最大路径长度
                len_of_grounded_knowledge[entity_label].append(max_path_len)
                # 记录当前的预测路径（用于后续分析编辑效果）
                predict_path[entity_label].append(init_reasoning_path[entity_label])

                # 若开启verbose模式，打印当前预测路径的长度（调试用）
                if options.verbose:
                    print("len of predict path", len(init_reasoning_path[entity_label].split("->")) - 1)

                    # ========== 评估初始推理路径（消融实验） ==========
                # 仅在第一次编辑前（refine=0）且开启initial_path_eval时执行
                if options.initial_path_eval and refine == 0:
                    # 初始化初始路径的实例列表
                    reasoning_paths_init = []
                    reasoning_paths_init.extend(result_paths)
                    # 调用LLM基于初始路径的KG实例进行QA推理
                    response_init, _ = llm_reasoning(reasoning_paths_init, question, options)
                    # 评估初始路径的推理结果是否命中真实答案
                    hit_init = evaluate(response_init, ground_truth)
                    # 记录初始路径的hit值
                    metrics['init_path_hit'].append(hit_init)
                    # 打印初始路径的累计hit均值（消融实验监控）
                    print(f"init_path_hit:{np.mean(metrics['init_path_hit']):.4f}")

                    # ========== 调用LLM编辑推理路径（若需要） ==========
                # end_refine=False表示需要编辑（路径实例化不完整/有错误）
                if not end_refine:
                    # 编辑次数+1
                    entity_refine += 1
                    refine = max(refine, entity_refine)
                    # 调用LLM_edit函数：基于反馈编辑推理路径，更新token计数和思维链
                    init_reasoning_path, thought, input_token_cnt, output_token_cnt, llm_call_count = LLM_edit(init_reasoning_path,
                                                                                               entity_label, feedback,
                                                                                               question, options,
                                                                                               input_token_cnt,
                                                                                               output_token_cnt, llm_call_count)
                    # 若开启verbose模式，打印编辑反馈和当前编辑次数（调试用）
                    if options.verbose:
                        print(f"{f'feedback: {feedback}' if not end_refine else ''}")
                        print(f"Refine time:{refine}")

                # ========== 停止编辑，处理最终路径 ==========
                # 条件：无需继续编辑（end_refine=True） 或 达到最大编辑次数
                if end_refine or refine >= MAX_REFINE_TIME - 1:
                    # 将当前实体的实例路径加入总路径列表
                    reasoning_paths.extend(result_paths)
                    # 存储当前实体的最终实例路径
                    knowledge_instance_final[entity_label] = result_paths
                    # 将路径转为字符串形式，用于后续去重
                    lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]

                    # 若实例化成功的路径长度>0，补充BFS过程中的有效路径
                    if max_path_len > 0:
                        for grounded_path in grounded_knowledge_current:
                            # 跳过长度小于最大路径长度的路径（不完整）
                            if grounded_path[-1] < max_path_len:
                                continue
                            # 将路径转为字符串形式
                            string_path = utils.path_to_string(grounded_path[1])
                            # 过滤空路径，且仅保留未重复的路径
                            if len(string_path) > 0:
                                if string_path not in lists_of_paths:
                                    lists_of_paths.append(string_path)
                                    # 初始化/追加路径到总列表
                                    if len(reasoning_paths) == 0:
                                        reasoning_paths = [grounded_path[1]]
                                    else:
                                        reasoning_paths.extend([grounded_path[1]])
                        # 路径字符串去重（避免重复实例）
                        lists_of_paths = list(set(lists_of_paths))
                    # 退出当前实体的编辑循环，处理下一个实体
                    break

        # ========== 多主题实体路径合并 ==========
        # 若样本有多个主题实体，合并所有实体的KG实例路径（保留交集实体相关路径，减少冗余）
        if len(topic_ent_list) > 1:
            reasoning_paths = merge_different_path(knowledge_instance_final, reasoning_paths, options)

        # ========== LLM推理与结果评估 ==========
        # 调用LLM基于合并后的KG实例路径进行QA推理，得到最终答案
        response, llm_call_count = llm_reasoning(reasoning_paths, question, options, llm_call_count)
        # 评估推理结果是否命中真实答案（hit=1/0）
        hit = evaluate(response, ground_truth)

        # 计算当前样本的token费用
        token_cost_now = (TOKEN_RATES[options.LLM_type]['input'] * input_token_cnt + TOKEN_RATES[options.LLM_type]['output'] * output_token_cnt) / 1000

        # 累加当前样本的token费用
        total_token_cost += token_cost_now

        # ========== 保存当前样本结果 ==========
        # 构造结果字典，包含核心信息（用于后续分析/续跑）
        info = {
            'question': question,  # 问题文本
            'init_path': init_reasoning_path,  # 最终推理路径
            'predict': response,  # LLM预测答案
            'ground_truth': ground_truth,  # 真实答案
            'kg_instances': reasoning_paths,  # KG实例路径
            'thought': thought,  # LLM编辑的思维链
            'init_path_hit': metrics['init_path_hit'][-1],  # 初始路径是否命中（1/0）
            'hit': hit,  # 命中结果（1/0）
            "Number of edits": refine,  # 编辑次数
            "input_token_cnt": input_token_cnt,  # 输入token数量
            "output_token_cnt": output_token_cnt,  # 输出token数量
            'token_cost': token_cost_now,  # 样本token费用
            "llm_call_count": llm_call_count,  # 累计大模型调用次数
        }
        # 将字典转为JSON字符串（保证可序列化）
        d = json.dumps(info)
        # 写入文件（每行一个JSON，jsonl格式）
        f.write(d + '\n')

        # 强制刷新文件缓冲区：确保数据实时写入硬盘，避免程序中断导致数据丢失
        f.flush()

        # ========== 累计评估指标 ==========
        # 累计当前样本的hit值和token费用
        metrics["hit"].append(hit)
        # 打印当前累计的hit均值（实时监控效果）
        print(f"hit:{np.mean(metrics['hit']):.4f}")

        # 可选：打印累计token费用均值（成本监控）
        print(f"total token cost:{total_token_cost:.6f}")
        print("-" * 50)


    # ========== 收尾工作 ==========
    # 关闭输出文件（释放资源）
    f.close()
    # 打印最终评估结果（分隔线区分，更醒目）
    print("\n================== Run Finished ==================")
    print(f"total input token: {input_token_cnt}")
    print(f"total output token: {output_token_cnt}")
    print(f"total token cost: ${token_cost:.6f}")
    if metrics['hit']:
        print(f"avg hit@1: {np.mean(metrics['hit']):.4f}")

if __name__ == '__main__':
    options = parse_args()
    main()
