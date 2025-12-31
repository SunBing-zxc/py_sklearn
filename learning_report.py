# learning_report.py
import json
import streamlit as st
from api_deepseek import ask_ai_assistant

# 基本概念模块
def load_module_records(module_type, raw_records):
    """
    解析不同模块的原始记录，返回统一格式的处理后数据
    module_type: 模块类型（如"linear_regression"、"logistic_regression"）
    raw_records: 从session_state获取的原始记录
    """
    if module_type == "linear_regression":
        return {
            "knowledge": {
                "quiz_score": raw_records.get("quiz", {}).get("score", 0),
                "incorrect_topics": [q["topic"] for q in raw_records.get("quiz", {}).get("incorrect_questions", [])],
                "ai_questions": [q["question"] for q in raw_records.get("ai_interactions", [])]
            },
            "skills": {
                "data_generation":{
                    "data_type":[q["data_type"] for q in raw_records.get("data_generation", [])]
                    },
                "manual_fitting": {
                    "final_mse": raw_records.get("manual_fitting", [{}])[-1].get("mse", float('inf')) if raw_records.get("manual_fitting") else None
                },
                "gradient_descent": {
                    "experiment_count": len(raw_records.get("gradient_descent", [])),
                    "convergence_rate": sum(1 for exp in raw_records.get("gradient_descent", []) if exp.get("convergence_status") == "模型收敛") 
                                      / (len(raw_records.get("gradient_descent", [])) + 1e-6)
                }
            },
            "habits": {
                "quiz_count": len(raw_records.get("quiz", [])),
                "module_sequence": raw_records.get("module_sequence", []),
                "module_duration": {
                    module: {  
                        "duration_min": round((data.get("exit_time", 0) - data.get("enter_time", 0)) / 60, 1)
                    } for module, data in raw_records.get("module_timestamps", {}).items()  # 正确位置：for循环是外部字典的推导式
                },
                "repeat_modules": len([m for i, m in enumerate(raw_records.get("module_sequence", [])) 
                                     if i > 0 and m == raw_records["module_sequence"][i-1]])
                }
        }
    
    elif module_type == "logistic_regression":
        # 逻辑回归特有的记录解析（根据实际记录结构调整）
        return {
            "knowledge": {
                "quiz_score": raw_records.get("logistic_quiz", {}).get("score", 0),
                "incorrect_topics": [q["topic"] for q in raw_records.get("logistic_quiz", {}).get("incorrect_questions", [])],
                "ai_questions": [q["question"] for q in raw_records.get("ai_interactions", [])]
            },
            "skills": {
                "data_generation":{
                    "data_type":[q["data_type"] for q in raw_records.get("data_generation", [])]
                    },
                "sig_function":{
                    "z_value":[q["z_value"] for q in raw_records.get("sig_function", [])]
                    },
                "para_tuning":{
                    "score_weight":[q["score_weight"] for q in raw_records.get("para_tuning", [])],
                    "absence_weight":[q["absence_weight"] for q in raw_records.get("para_tuning", [])],
                    "bias":[q["bias"] for q in raw_records.get("para_tuning", [])],
                    "threshold":[q["threshold"] for q in raw_records.get("para_tuning", [])]
                    },
                "gradient_descent":{
                    "learning_rate":[q["learning_rate"] for q in raw_records.get("gradient_descent", [])],
                    "n_iterations":[q["n_iterations"] for q in raw_records.get("gradient_descent", [])]
                    },                
                "model_evaluation":{
                    "scenario":[q["scenario"] for q in raw_records.get("model_evaluation", [])]
                    }                
            },
            "habits": {
                "quiz_count": len(raw_records.get("logistic_quiz", {})),
                "module_sequence": raw_records.get("module_sequence", []),
                "module_duration": {
                    module: {  
                        "duration_min": round((data.get("exit_time", 0) - data.get("enter_time", 0)) / 60, 1)
                    } for module, data in raw_records.get("module_timestamps", {}).items()  
                },
                "repeat_modules": len([m for i, m in enumerate(raw_records.get("module_sequence", [])) 
                                     if i > 0 and m == raw_records["module_sequence"][i-1]])
                }
            }
    elif module_type == "kmeans":
        # KMeans特有的记录解析（根据实际记录结构调整）
        return {
            "knowledge": {
                "quiz_score": raw_records.get("kmeans_quiz", {}).get("score", 0),
                "incorrect_topics": [q["topic"] for q in raw_records.get("kmeans_quiz", {}).get("incorrect_questions", [])],
                "ai_questions": [q["question"] for q in raw_records.get("ai_interactions", [])]
            },
            "skills": {
                "data_generation":{
                    "data_type":[q["data_type"] for q in raw_records.get("data_generation", [])]
                    },
                "kmeans_basics_section":{
                    "k_value":[q["k_value"] for q in raw_records.get("kmeans_basics_section", [])]
                    },
                "k_selection_section":{                   
                    "max_k_elbow":[q["max_k_elbow"] for q in raw_records.get("k_selection_section", [])],
                    "max_k_silhouette":[q["max_k_silhouette"] for q in raw_records.get("k_selection_section", [])]
                    },
                "evaluation_metrics_section":{
                    "k_value":[q["k_value"] for q in raw_records.get("evaluation_metrics_section", [])]
                    },                
                "real_world_example_section":{
                    "example":[q["example"] for q in raw_records.get("real_world_example_section", [])]
                    }                
            },
            "habits": {
                "quiz_count": len(raw_records.get("kmeans_quiz", {})),
                "module_sequence": raw_records.get("module_sequence", []),
                "module_duration": {
                    module: {  
                        "duration_min": round((data.get("exit_time", 0) - data.get("enter_time", 0)) / 60, 1)
                    } for module, data in raw_records.get("module_timestamps", {}).items()  
                },
                "repeat_modules": len([m for i, m in enumerate(raw_records.get("module_sequence", [])) 
                                     if i > 0 and m == raw_records["module_sequence"][i-1]])
                }
            }
    elif module_type == "Neural_Network":
        return {
            "knowledge": {
                "quiz_score": raw_records.get("ANN_quiz", {}).get("score", 0),
                "incorrect_topics": [q["topic"] for q in raw_records.get("ANN_quiz", {}).get("incorrect_questions", [])],
                "ai_questions": [q["question"] for q in raw_records.get("ai_interactions", [])]
            },
            "skills": {
                "multi_layer_nn_section":{
                    "hidden_units":[q["hidden_units"] for q in raw_records.get("multi_layer_nn_section", [])],
                    "activation":[q["activation"] for q in raw_records.get("multi_layer_nn_section", [])],
                    "max_iter":[q["max_iter"] for q in raw_records.get("multi_layer_nn_section", [])],
                    "learning_rate":[q["learning_rate"] for q in raw_records.get("multi_layer_nn_section", [])]
                    },
                "activation_functions_section":{
                    "activation1":[q["activation1"] for q in raw_records.get("activation_functions_section", [])],
                    "activation":[q["activation2"] for q in raw_records.get("activation_functions_section", [])],
                    },
                "nn_parameter_tuning_section":{
                    "hidden_layer_sizes":[q["hidden_layer_sizes"] for q in raw_records.get("nn_parameter_tuning_section", [])],
                    "neurons_per_layer":[q["neurons_per_layer"] for q in raw_records.get("nn_parameter_tuning_section", [])],
                    "regularization":[q["regularization"] for q in raw_records.get("nn_parameter_tuning_section", [])],
                    "learning_rate":[q["learning_rate"] for q in raw_records.get("nn_parameter_tuning_section", [])],
                    "solver":[q["solver"] for q in raw_records.get("nn_parameter_tuning_section", [])]
                    },
            },
            "habits": {
                "quiz_count": len(raw_records.get("ANN_quiz", {})),
                "module_sequence": raw_records.get("module_sequence", []),
                "module_duration": {
                    module: {  
                        "duration_min": round((data.get("exit_time", 0) - data.get("enter_time", 0)) / 60, 1)
                    } for module, data in raw_records.get("module_timestamps", {}).items()  
                },
                "repeat_modules": len([m for i, m in enumerate(raw_records.get("module_sequence", [])) 
                                     if i > 0 and m == raw_records["module_sequence"][i-1]])
                }
            }
    elif module_type == "text_analysis":
        return {
            "knowledge": {
                "quiz_score": raw_records.get("text_analysis_quiz", {}).get("score", 0),
                "incorrect_topics": [q["topic"] for q in raw_records.get("text_analysis_quiz", {}).get("incorrect_questions", [])],
                "ai_questions": [q["question"] for q in raw_records.get("ai_interactions", [])]
            },
            "skills": {
                "text_preprocessing_section":{
                    "lang":[q["lang"] for q in raw_records.get("text_preprocessing_section", [])],
                    },
                "native_bys_section":{
                    "user_guess":[q["user_guess"] for q in raw_records.get("native_bys_section", [])],
                    },
                "text_analysis_section":{
                    "selected_sample":[q["selected_sample"] for q in raw_records.get("text_analysis_section", [])],
                    "test_size":[q["test_size"] for q in raw_records.get("text_analysis_section", [])],
                    "model_name":[q["model_name"] for q in raw_records.get("text_analysis_section", [])],
                    },
                "sentiment_analysis_section":{
                    "test_size":[q["test_size"] for q in raw_records.get("sentiment_analysis_section", [])],
                    "model_name":[q["model_name"] for q in raw_records.get("sentiment_analysis_section", [])],
                    "user_comment":[q["user_comment"] for q in raw_records.get("sentiment_analysis_section", [])],
                    },
            },
            "habits": {
                "quiz_count": len(raw_records.get("text_analysis_quiz", {})),
                "module_sequence": raw_records.get("module_sequence", []),
                "module_duration": {
                    module: {  
                        "duration_min": round((data.get("exit_time", 0) - data.get("enter_time", 0)) / 60, 1)
                    } for module, data in raw_records.get("module_timestamps", {}).items()  
                },
                "repeat_modules": len([m for i, m in enumerate(raw_records.get("module_sequence", [])) 
                                     if i > 0 and m == raw_records["module_sequence"][i-1]])
                }
            }

        
# 基本概念模块
def build_evaluation_prompt(module_type, processed_data):
    """根据模块类型构建针对性的评价提示词"""
    base_prompt = f"""
请仔细分析以下JSON文件中记录的学生学习数据，客观评价该学生的学习情况。
需要重点关注文件中明确记录的测验分数，务必以文件内的实际数据为准，不得自行假设或更改分数。
请从测验表现、薄弱知识点、学习行为（如学习模块的参与情况、时长、是否向AI提问等）、技能掌握等方面进行分析，指出其优势与不足，并给出针对性的学习建议。

注意：不要出现具体的次数，用“少数，多数，较多”等词表示

学生数据：{json.dumps (processed_data, ensure_ascii=False, indent=2)}
输出要求：总体总结（1 句）+ 三维度分析（结合数据）+ 3 条提升建议，500 字以内。
    """

    return base_prompt.strip()

# 基本概念模块
def generate_evaluation(module_type, raw_records):
    """生成最终学习报告的入口函数"""
    try:
        processed_data = load_module_records(module_type, raw_records)
            
        prompt = build_evaluation_prompt(module_type, processed_data)
        with st.spinner(f"AI正在分析你的学习记录..."):
            return ask_ai_assistant(question=prompt, context=f"{module_type}学习评价")
    except Exception as e:
        return f"评价生成失败：{str(e)}"

# 分步编程模块专用记录处理函数
def load_records_step(raw_records,steps):
    """处理分步编程的原始学习记录，提取关键评估数据"""
    step_metrics = {}
    for step_num in range(steps+1):  
        step_key = f"step_{step_num}"
        step_data = raw_records['step_records'].get(step_key, {})
        
        error_topics = []
        for error in step_data.get('error_details', []):
            if 'Q' in error['question'] or 'T' in error['question']:
                topic = error['question'].split('：')[-1].strip()
                error_topics.append(topic)
        
        step_metrics[step_key] = {
            "completed": step_data.get('is_completed', False),
            "duration_min": round(step_data.get('duration', 0) / 60, 1),
            "error_count": step_data.get('error_count', 0),
            "error_topics": list(set(error_topics)),
            "reflection": raw_records.get('reflection', {}).get(step_key, "")
        }
        
    total_steps = len(step_metrics)
    completed_steps = sum(1 for step in step_metrics.values() if step["completed"])

    return {
        "total_progress": round(completed_steps / total_steps * 100, 1),
        "step_metrics": step_metrics,
        "total_errors": raw_records.get('total_errors', 0),
        "completion_rate": f"{completed_steps}/{total_steps}",

    }

# 提示词构建（分步编程）
def build_prompt_step(processed_data,steps):
    """构建针对分步编程的评价提示词"""
    return f"""
你是机器学习教育评估专家，需基于学生的分步编程的学习记录，重点从以下维度生成评价：
1. **总体评价**
- 对学生完成流程给予热情的表扬

2. **错误知识点解析**（40%）
- 总结学生答题的整体正确率
- 归纳总结学生在单选/填空答错的知识点
- 指明对应知识点的学习方向（如"建议复习模型参数调用规范"）

3. **反思内容回应**（30%）
- 归纳总结学生在各步骤提出的反思内容(步骤{steps}中反思先不要提及）
- 若学生没有反思(不包含步骤{steps}中反思），请给出委婉的建议

4. **模型分析评价**（20%）
- 重点引用学生在步骤{steps}中的反思内容，积极肯定合理观点，引用内容要突出显示
- 明确指出学生未考虑的关键内容

5. **耗时分析**（10%）
- 说明各步骤耗时情况，指出耗时过长的步骤名称，可能存在卡壳。

注意：
- 不涉及流程完成度和连贯性评价
- 避免具体数字，用"少数/多次错误"等表述
- 语言生动热情，适当使用Emoji
- 总字数500以内，合理安排好各部分的字数

学生数据：
{json.dumps(processed_data, ensure_ascii=False, indent=2)}

输出结构：
1. 总体评价（1句）
2. 错误知识点解析（含解释+学习方向）
3. 反思内容回应（逐条对应解释）
4. 模型分析评价（肯定+未考虑点）
5. 耗时分析（指出可能卡壳步骤,步骤名称显示具体名称，例如：'步骤[模型评估]'）

"""

# 报告生成入口（分步编程）
def generate_report_step(raw_records,steps):
    """生成分步编程模块学习报告的主函数"""
    try:
        processed_data = load_records_step(raw_records,steps)
        prompt = build_prompt_step(processed_data,steps)
        with st.spinner("AI正在生成你的分步编程学习记录..."):
            report_content = ask_ai_assistant(prompt, context="分步编程评价")
            return report_content
    except Exception as e:
        return {"error": True, "message": f"分步编程报告生成失败：{str(e)}"}


