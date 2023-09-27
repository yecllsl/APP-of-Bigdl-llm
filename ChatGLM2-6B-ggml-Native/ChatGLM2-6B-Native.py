
from bigdl.llm.ggml.model.chatglm.chatglm import ChatGLM    
from bigdl.llm.transformers import BigdlNativeForCausalLM

model_name = "chatglm2-6b"
model_all_local_path = "D:\\yecll\\githubCode\\my-bigdl-llm-test\\checkpoint\\"

prompt = "制定一份健身计划"
if model_name == "chatglm2-6b":
    model = ChatGLM(model_all_local_path + "ggml-chatglm2-6b-q4_0.bin", n_threads=20,n_ctx=4096)
    response = ""
    for chunk in model(prompt, temperature=0.95,top_p=0.8,stream=True,max_tokens=512):
        response += chunk['choices'][0]['text'] 
    print(response)