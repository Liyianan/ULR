import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from run_mse import ElectraCombineModel
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

model_bert = AutoModelWithLMHead.from_pretrained('bert-base-chinese')
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = ElectraCombineModel(model_bert, tokenizer=tokenizer, mlm_probability=0.15)
path = 'output'
state_dict = torch.load(path + 'pytorch_model.bin')
model.load_state_dict(state_dict)
model.eval()
model.save_pretrained(path)
os.system('cp ' + path  + 'vocab.txt ' + path + 'discriminator/')
     
