from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer
from bigdl.llm.transformers import AutoModelForCausalLM
model_name = "chatglm2-6b"
model_all_local_path = "D:\\data\\"
model_name_local = model_all_local_path + model_name

if model_name == "chatglm2-6b":
    tokenizer = AutoTokenizer.from_pretrained(model_name_local, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_local, trust_remote_code=True, load_in_4bit=True)
    model.save_low_bit("D:\\data\\chatglm2-6b-int4\\")
    tokenizer.save_pretrained("D:\\data\\chatglm2-6b-int4\\")