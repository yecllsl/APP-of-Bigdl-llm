
from langchain import LLMChain, PromptTemplate
from bigdl.llm.langchain.llms import TransformersLLM
from langchain.memory import ConversationBufferWindowMemory

llm_model_path ="D:\\data\\chatglm2-6b" #"THUDM/chatglm2-6b" # the path to the huggingface llm model
CHATGLM_V2_LANGCHAIN_PROMPT_TEMPLATE = """{history}\n\n问：{human_input}\n\n答："""
prompt = PromptTemplate(input_variables=["history", "human_input"], template=CHATGLM_V2_LANGCHAIN_PROMPT_TEMPLATE)
max_new_tokens = 128

llm = TransformersLLM.from_model_id(
        model_id=llm_model_path,
        model_kwargs={"trust_remote_code": True},
)

# Following code are complete the same as the use-case
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    #llm_kwargs={"max_new_tokens":max_new_tokens},
    memory=ConversationBufferWindowMemory(k=2),
)

text = "红酒是什么？"
response_text = llm_chain.run(human_input=text,stop="\n\n")
print(response_text)