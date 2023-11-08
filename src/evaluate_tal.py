from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
import os
import sys

# 指定本地模型的路径
model_path = "../meta-math/MetaMath-7B-V1.0"
tokenizer_path = "../meta-math/MetaMath-7B-V1.0"

val_dataset_path = "../data/AAAI/TAL-SAQ6K-EN.jsonl"
answers_path = "../data/answers.jsonl"

# 加载本地模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)

# 指定模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

model.config.use_offline = True
tokenizer.use_fast = False

# 通过pipeline加载模型
generator = pipeline("text-generation", model=model,
                     tokenizer=tokenizer, device=device,)

print(generator("what is 1+2?", num_return_sequences=1)[0]['generated_text'])

batch_size = 128

pro_id_list = []


with open(answers_path, 'r', encoding='utf-8-sig') as output_file:
    data = output_file.readlines()
    for line_number, line in enumerate(data, start=1):
        pro_id_list.append(json.loads(line)['queId'])

with open(answers_path, "a", encoding="utf-8") as answer_file:
    with open(val_dataset_path, 'r', encoding='utf-8-sig') as f:
        input_data = []
        que_ids = []

        for line in f:
            item = json.loads(line)
            que_id = item['queId']
            if que_id in pro_id_list:
                continue
            else:
                pro_id_list.append(que_id)

            question_text = item['problem']
            print('dealing with '+que_id)

            input_data.append(question_text)
            que_ids.append(que_id)

            # 达到批次大小后进行处理
            if len(que_ids) == batch_size:
                # 一次传递整个输入数据列表给生成器
                output = generator(input_data, num_return_sequences=1)

                for i, result in enumerate(output):
                    print(result)
                    answer = result[0]['generated_text']
                    output_data = {
                        "queId": que_ids[i],
                        "problem": input_data[i],
                        "answer": answer
                    }
                    answer_file.write(json.dumps(
                        output_data, ensure_ascii=False) + '\n')

                input_data = []
                que_ids = []

                torch.cuda.empty_cache()

        # 处理最后一个不足一批的样本
        if len(input_data) > 0:
            output = generator(input_data, num_return_sequences=1)

            for i, result in enumerate(output):
                answer = result[0]['generated_text']
                # 存储答案到文件
                output_data = {
                    "queId": que_ids[i],
                    "problem": input_data[i],
                    "answer": answer
                }
                answer_file.write(json.dumps(
                    output_data, ensure_ascii=False) + '\n')
        torch.cuda.empty_cache()
