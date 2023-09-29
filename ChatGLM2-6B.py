from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer
import time
import torch

model_path = "d:\\data\\chatglm2-6b" 
#model_path = "D:\\yecll\\githubCode\\my-bigdl-llm-test\\checkpoint\\ggml-chatglm2-6b-q4_0.bin"
"""    
    这段代码定义了一个名为from_pretrained的函数，用于从预训练模型库中加载模型。这个函数主要用于实现模型的加载和转换，以便在BigDL框架中使用。

    函数的主要功能包括：
    接收一个或多个参数，包括pretrained_model_name_or_path和*args、**kwargs。pretrained_model_name_or_path参数用于指定模型的名称或路径，*args和**kwargs用于传递其他参数。
    从PretrainedConfig.get_config_dict函数中获取模型的配置字典和预训练模型。
    检查模型是否为低位模型（即使用bigdl_transformers_low_bit键），如果是，则抛出一个错误，提示用户使用load_low_bit函数加载模型。
    检查用户是否提供了load_in_4bit和load_in_low_bit参数，如果有，则根据参数值进行相应的模型加载和转换。
    如果用户提供了optimize_model参数，则将其设置为默认值True。
    如果模型为低位模型，使用load_convert函数将其转换为指定低位格式的模型，并应用 relevant low bit optimizations。
    否则，使用HF_Model.from_pretrained方法加载原始格式的模型。 
"""   
model = AutoModel.from_pretrained(model_path,load_in_4bit = True,trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
CHATGLM_V2_PROMPT_TEMPLATE = "问：{prompt}\n\n答："

prompt = "红酒是什么？"
n_predict = 128

""" with torch.inference_mode():
    prompt = CHATGLM_V2_PROMPT_TEMPLATE.format(prompt=prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    st = time.time()
    output = model.generate(input_ids,
                            max_new_tokens=n_predict)
    end = time.time()
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f'Inference time: {end-st} s')
    print('-'*20, 'Output', '-'*20)
    print(output_str) """
    
with torch.inference_mode():
    question = "红酒是什么？"
    response_ = ""
    print('-'*20, 'Stream Chat Output', '-'*20)
    for response, history in model.stream_chat(tokenizer, question, history=[]):
        print(response.replace(response_, ""), end="")
        response_ = response