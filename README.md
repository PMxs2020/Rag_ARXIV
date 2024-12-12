# 如何运行
python版本3.9

```plain
pip install openai==0.28
pip install pymilvus==2.2.6
```

其他的按照需要安装最新版，main文件执行，在控制台问答

# 如何运行项目功能
构建一个针对 arXiv 的知识问答系统：</font>
给定一个入口，用户可以输入提问</font>
不要求要求构建 GUI 界面</font>
用户通过对话进行交互</font>
系统寻找与问题相关的论文 abstract：</font>
使用用户的请求对向量数据库进行请求</font>
寻找与问题最为相关的 abstract</font>
系统根据问题和论文 abstract 回答用户问题，并给出解答问题的信息来源</font>

本项目强化了知识问答系统的文献检索能力和知识回答能力

1.优化用户提问，支持对用户的提问概念分解并按关键词检索文献库，提升文献相关性

2.解答问题支持给出当前文献的对问题的观点，支持引用文献和提供文献访问的地址

3.对于文献库检索不到的文献，支持按大模型自己的理解回答

# 项目依赖资源
# 大模型（Qwen2.5-14B）
```python
from langchain_openai import *
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
llm_completion = ChatOpenAI(model_name="Qwen2.5-14B")
llm_chat = ChatOpenAI(model_name="Qwen2.5-14B")
```

# 嵌入模型（sentence-transformers/all-MiniLM-L12-v2）
```python
#嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings  # 替换 HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
```


# 向量数据
```plain
#向量数据库
from langchain_community.vectorstores import Milvus
db = Milvus(embedding_function=embedding, collection_name="arXiv",connection_args={"host": "10.58.0.2", "port": "19530"})
from langchain.schema import HumanMessage
```

文件解释：
questions.jsonl是示例的十个问题
answer.json是系统回答的结果

