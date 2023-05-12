import time
import openai
import json


openai.api_key = 'OPENAI_API_KEY'


def generate_qa_pairs(document_text):
    # 将文档数据拆分成段落
    document_paragraphs = document_text.split("\n")
    # 初始化空的问答对列表
    qa_pairs = []
    # 计算一下token的总量
    total_tokens = 0
    # 计算对于每段文章内容输入输出的token 例如：
    # {
    #     "completion_tokens": 239,
    #     "prompt_tokens": 224,
    #     "total_tokens": 463
    # }
    token_list = []

    # 对于每个段落，生成问题答案
    for paragraph in document_paragraphs:
        if not paragraph:
            continue
        # 构建一个 GPT 提示符，以段落作为上下文，生成一个相关问题
        prompt_text = f'仅针对以下内容生成几个问题并根据该内容进行作答' \
                      f'格式为：问：xx?' \
                      f'答：xx。' \
                      f'' \
                      f'内容为：" {paragraph}?"'

        # 使用 gpt-3.5-turbo 获取内容的问答对
        # https://platform.openai.com/docs/api-reference/chat/create
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_text}
            ]
        )

        mes = response.choices[0].message
        all_content = mes.content
        every_content = all_content.split("\n\n")
        try:
            for content in every_content:
                con = content.split('\n')
                qa_pair = {"prompt": con[0][2:], "completion": con[1][2:]}
                qa_pairs.append(qa_pair)
                print(qa_pair)
        except IndexError as e:

            print(all_content)
        print('\n-----------------\n')

        # 每次问答的token
        usage = response.usage
        token_list.append(usage)
        # 总token
        total_tokens += int(usage.total_tokens)
        # gpt-3.5-turbo 每分钟只能调用3次
        time.sleep(35)
    print(token_list)
    print(total_tokens)
    # 返回问答对列表
    return qa_pairs


# 获取文章内容
with open("./content.txt", "r") as f:
    document_text = f.read()

qa_pairs = generate_qa_pairs(document_text)
# 把问答对写入文件
with open("qa_pairs.json", "a") as f:
    json.dump(qa_pairs, f)

# 由于上传文件的格式要是JSONL的，使用官方 CLI 工具做一下转化，输入以下内容
# openai tools fine_tunes.prepare_data -f qa_pairs.json


# 上传文件
# https://platform.openai.com/docs/api-reference/files/upload
def create_file():
    res = openai.File.create(
      file=open("qa_pairs_prepared.jsonl", "rb"),
      purpose='fine-tune'
    )
    print(res)
    return res


# create_file()


# 根据上传的问答对文件，创建微调模型
# https://platform.openai.com/docs/api-reference/fine-tunes/create
def create_fine_tune():
    # file-xxx在上传文件后返回的结果中
    res = openai.FineTune.create(training_file="file-xxx")
    # res = openai.FineTune.retrieve(id="ft-AF1WoRqd3aJAHsqc9NY7iL8F")

    print(res)

# create_fine_tune()



