## BigDL-LLM for native INT4 model
### 本文件夹下例子，针对Native  INT4模型，尤其是支持中文的ChatGLM2-6B模型，进行工作的展示。目标是在Intel的笔记本上，实现低延迟的推理。展示问答、聊天和基于文档的生成三种任务。
**[`bigdl-llm`](https://bigdl.readthedocs.io/en/latest/doc/LLM/index.html)** is a library for running **LLM** (large language model) on Intel **XPU** (from *Laptop* to *GPU* to *Cloud*) using **INT4** with very low latency[^1] (for any **PyTorch** model).

> *It is built on top of the excellent work of [llama.cpp](https://github.com/ggerganov/llama.cpp), [gptq](https://github.com/IST-DASLab/gptq), [ggml](https://github.com/ggerganov/ggml), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [qlora](https://github.com/artidoro/qlora), [gptq_for_llama](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [chatglm.cpp](https://github.com/li-plus/chatglm.cpp), [redpajama.cpp](https://github.com/togethercomputer/redpajama.cpp), [gptneox.cpp](https://github.com/byroneverson/gptneox.cpp), [bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp/), etc.*
    

### Working with `bigdl-llm`

<details><summary>Table of Contents</summary>

- [Install](#install)
- [Run Model](#run-model)
  - [Native INT4 Model](#2-native-int4-model)
  - [LangChain API for Native](#l3-angchain-api)
  - [CLI Tool for Native](#4-cli-tool)
- [`bigdl-llm` API Doc](#bigdl-llm-api-doc)

</details>

#### Install
##### CPU
You may install **`bigdl-llm`** on Intel CPU as follows:
```bash
pip install --pre --upgrade bigdl-llm[all]
```
> Note: `bigdl-llm` has been tested on Python 3.9

##### GPU
You may install **`bigdl-llm`** on Intel GPU as follows:
```bash
# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```
> Note: `bigdl-llm` has been tested on Python 3.9



##### Native INT4 model
 
You may also convert Hugging Face *Transformers* models into native INT4 model format for maximum performance as follows.

>**Notes**: Currently only llama/bloom/gptneox/starcoder/chatglm model families are supported; for other models, you may use the Hugging Face `transformers` model format as described above).
  
```python
#convert the model
from bigdl.llm import llm_convert
bigdl_llm_path = llm_convert(model='/path/to/model/',
        outfile='/path/to/output/', outtype='int4', model_family="llama")

#load the converted model
#switch to ChatGLMForCausalLM/GptneoxForCausalLM/BloomForCausalLM/StarcoderForCausalLM to load other models
from bigdl.llm.transformers import LlamaForCausalLM
llm = LlamaForCausalLM.from_pretrained("/path/to/output/model.bin", native=True, ...)
  
#run the converted model
input_ids = llm.tokenize(prompt)
output_ids = llm.generate(input_ids, ...)
output = llm.batch_decode(output_ids)
``` 

See the complete example [here](example/transformers/native_int4/native_int4_pipeline.py). 

##### LangChain API for native INT4 model
You may run the models using the LangChain API in `bigdl-llm`.


 
- **Using native INT4 model**

  You may also convert Hugging Face *Transformers* models into *native INT4* format, and then run the converted models using the LangChain API as follows.
  
  >**Notes**:* Currently only llama/bloom/gptneox/starcoder/chatglm model families are supported; for other models, you may use the Hugging Face `transformers` model format as described above).

  ```python
  from bigdl.llm.langchain.llms import LlamaLLM
  from bigdl.llm.langchain.embeddings import LlamaEmbeddings
  from langchain.chains.question_answering import load_qa_chain

  #switch to ChatGLMEmbeddings/GptneoxEmbeddings/BloomEmbeddings/StarcoderEmbeddings to load other models
  embeddings = LlamaEmbeddings(model_path='/path/to/converted/model.bin')
  #switch to ChatGLMLLM/GptneoxLLM/BloomLLM/StarcoderLLM to load other models
  bigdl_llm = LlamaLLM(model_path='/path/to/converted/model.bin')

  doc_chain = load_qa_chain(bigdl_llm, ...)
  doc_chain.run(...)
  ```

  See the examples [here](example/langchain/native_int4).

#####  CLI Tool
>**Note**: Currently `bigdl-llm` CLI supports *LLaMA* (e.g., *vicuna*), *GPT-NeoX* (e.g., *redpajama*), *BLOOM* (e.g., *pheonix*) and *GPT2* (e.g., *starcoder*) model architecture; for other models, you may use the Hugging Face `transformers` or LangChain APIs.

 - ##### Convert model
 
    You may convert the downloaded model into native INT4 format using `llm-convert`.
    
   ```bash
   #convert PyTorch (fp16 or fp32) model; 
   #llama/bloom/gptneox/starcoder model family is currently supported
   llm-convert "/path/to/model/" --model-format pth --model-family "bloom" --outfile "/path/to/output/"

   #convert GPTQ-4bit model
   #only llama model family is currently supported
   llm-convert "/path/to/model/" --model-format gptq --model-family "llama" --outfile "/path/to/output/"
   ```  
  
 - ##### Run model
   
   You may run the converted model using `llm-cli` or `llm-chat` (*built on top of `main.cpp` in [llama.cpp](https://github.com/ggerganov/llama.cpp)*)

   ```bash
   #help
   #llama/bloom/gptneox/starcoder model family is currently supported
   llm-cli -x gptneox -h

   #text completion
   #llama/bloom/gptneox/starcoder model family is currently supported
   llm-cli -t 16 -x gptneox -m "/path/to/output/model.bin" -p 'Once upon a time,'

   #chat mode
   #llama/gptneox model family is currently supported
   llm-chat -m "/path/to/output/model.bin" -x llama
   ```

### `bigdl-llm` API Doc
See the inital `bigdl-llm` API Doc [here](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/LLM/index.html).

[^1]: Performance varies by use, configuration and other factors. `bigdl-llm` may not optimize to the same degree for non-Intel products. Learn more at www.Intel.com/PerformanceIndex.


