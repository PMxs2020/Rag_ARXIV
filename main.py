# 大模型配置
from langchain_core.messages import HumanMessage
from langchain_openai import *
import os
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
llm_completion = ChatOpenAI(model_name="Qwen2.5-14B")
llm_chat = ChatOpenAI(model_name="Qwen2.5-14B")
#嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings  # 替换 HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
#向量数据库
from langchain_community.vectorstores import Milvus
db = Milvus(embedding_function=embedding, collection_name="arXiv",connection_args={"host": "10.58.0.2", "port": "19530"})
from langchain.schema import HumanMessage

# 处理用户输入，执行检索并生成回答
def get_relevant_abstracts(question: str, top_k: int = 10):
    """
    通过Milvus向量数据库根据用户问题查询相关论文摘要
    """
    attempts = 0
    while attempts < 3:
        # 使用Milvus数据库进行检索
        search_results = db.similarity_search(question, k=top_k)
        # 检查是否找到了相关文献
        if len(search_results) > 0:
            # 返回查询到的文献信息
            return search_results

        # 如果没有找到，增加尝试次数
        attempts += 1

    return []

def generate_answer(question: str, abstracts: list):
    """
    使用LLM生成答案，并引用相关论文的摘要
    """
    # 创建Prompt模板
    prompt_template = """
    用户提问: {question}
    以下是与问题最相关的论文摘要和信息:
    {abstracts}
    其中access_id为文献的id,authors为文献的作者,abstract为文献的摘要,title为文献的标题。
    请结合上面的摘要回答用户的问题，并引用相关的文献。
    按照这种格式回复："(对问题中的概念做出解释),\n***认为(文献的观点）[1],***认为(文献的观点）[2]。依据的参考文献如下：[1](title),(authors),论文的详情页：https://arxiv.org/abs/(access_id) 论文的pdf地址:https://arxiv.org/pdf/(access_id)[2]....
    注意:()内是你要填充的内容，回答时要把()去掉
    注意:如果提供的文献中对提问的关键词有完全不同的含义解释，代表这是两个不同领域的概念，需要在回答中加以区分
    如果提供的论文摘要和信息与用户的问题无关，则按照你的知识理解来回答，可以参考类似格式：“我在arXiv文献库中没有找到相关文献，我将根据我的理解为您回答:....",作为完整回答，也就不用提及作者观点和参考文献的信息了
    下面是一个例子：“大语言模型（Large Language Models，LLMs）是指一类在大量文本数据上训练的深度学习模型，它们能够生成与训练数据相类似的文本，并且能够完成诸如语言翻译、文本生成、问答等多种任务。这些模型因其复杂性和潜在的风险，例如生成有害或误导性的内容，成为了当前研究中的一个重要课题。

Paul Rottger等认为大语言模型的安全性，包括防止生成偏见和有害内容，是当前研究中的一个重要方向[1]。Zishan Guo等认为大语言模型的评估应该包括知识和能力评估、对齐评估和安全评估三个方面，以确保模型的使用是安全和有益的[2]。

依据的参考文献如下：
[1] SafetyPrompts: a Systematic Review of Open Datasets for Evaluating and Improving Large Language Model Safety, Paul Rottger, Fabio Pernisi, Bertie Vidgen, Dirk Hovy,论文的详情页：https://arxiv.org/abs/2404.05399 论文的pdf地址:https://arxiv.org/pdf/2404.05399
[2] Evaluating Large Language Models: A Comprehensive Survey, Zishan Guo, Renren Jin, Chuang Liu, Yufei Huang, Dan Shi, Supryadi, Linhao Yu, Yan Liu, Jiaxuan Li, Bojian Xiong, Deyi Xiong,论文的详情页：https://arxiv.org/abs/2310.19736 论文的pdf地址:https://arxiv.org/pdf/2310.19736”
    """
    abstracts_info=[]
    for abstract in abstracts:
        access_id="access_id:"+abstract.metadata['access_id']
        authors="authors:"+abstract.metadata['authors']
        title = "title:" + abstract.metadata['title']
        text="abstract:"+abstract.page_content
        abstracts_info.append(access_id+","+authors+","+text+","+title)

    prompt = prompt_template.format(question=question, abstracts="\n\n".join(abstracts_info))

    # 使用LLM生成答案
    response = llm_chat.generate([prompt])
    return response.generations[0][0].text


def statement_optimize(question):
    question_milvus_template=("""
    用户提问{question}
    请将其进行关键词拆分并且优化问题语句并均以英文的方式回答我例如：什么是软件工程？，你的回答应该是 what is software engineering safe？ keywords include software engineering,safe.
    可以根据常用概念适度扩展1-2个关键词,比如big model一般被认为是big language model,关键词的顺序按照概念常用频率从高到低排序""")
    query=llm_completion.generate([question_milvus_template.format(question=question)])
    return query.generations[0][0].text


def answer_question(question: str):
    """
    完整的问答流程，检索摘要并生成答案
    """
    # 文献查询语句优化
    query=statement_optimize(question)
    # 获取与问题最相关的论文摘要
    abstracts = get_relevant_abstracts(query)
    if len(abstracts) > 0:
        return generate_answer(question, abstracts)
    else:
        return "没有搜索到相关文献，请重新提问"

# 主程序
if __name__ == "__main__":
    print("欢迎使用 arXiv 知识问答系统！")
    while True:
        question = input("请输入您的问题（或输入'退出'退出）：")
        if question.lower() == "退出":
            break
        answer = answer_question(question)
        print(f"回答: {answer}")