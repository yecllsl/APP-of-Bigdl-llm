import time
import torch
from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer
from bigdl.llm.transformers import AutoModelForCausalLM

# """ 这个转换的模型，并不能直接使用，AutoModel使用的还是fp16的模型，过程中进行转化 """

# model_name = "chatglm2-6b"
# model_all_local_path = "D:\\data\\"
# model_name_local = model_all_local_path + model_name

# if model_name == "chatglm2-6b":
#     tokenizer = AutoTokenizer.from_pretrained(model_name_local, trust_remote_code=True)
#     model = AutoModel.from_pretrained(model_name_local, trust_remote_code=True, load_in_4bit=True)
#     model.save_low_bit("D:\\data\\chatglm2-6b-int4\\")
#     tokenizer.save_pretrained("D:\\data\\chatglm2-6b-int4\\")
    
    
from bigdl.llm.transformers import AutoModelForCausalLM

model_path = "D:\\data\\chatglm2-6b"
save_directory = 'D:\\data\\chatglm2-6b-bigdl-llm-INT4'

######这部分是保存为low_bit模型
# model = AutoModel.from_pretrained(model_path,trust_remote_code=True,load_in_4bit=True)
# model.save_low_bit(save_directory)
# del(model)
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# tokenizer.save_pretrained(save_directory)

####读取low_bit模型，不需要每次都转化，注意还要读取tokenizer
model = AutoModelForCausalLM.load_low_bit(save_directory, 
                                          trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(save_directory,
                                          trust_remote_code=True)

CHATGLM_V2_PROMPT_TEMPLATE = "问：{prompt}\n\n答："
prompt = "AI是什么？"
n_predict = 128

with torch.inference_mode():
    prompt = CHATGLM_V2_PROMPT_TEMPLATE.format(prompt=prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    st = time.time()
    output = model.generate(input_ids,
                            max_new_tokens=n_predict)
    end = time.time()
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f'Inference time: {end-st} s')
    print('-'*20, 'Output', '-'*20)
    print(output_str)