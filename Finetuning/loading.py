import torch

from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from constants import *
from inference.utils import *
from prompting import *

def load_base_model(model_id):
    # model quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",  
        trust_remote_code=True,
    )
    return base_model


def load_model(model_id, weight_path=None, checkpoint=0, inference=True):
    model = load_base_model(model_id)
    if checkpoint > 0:
        checkpoint_weights = f"{weight_path}/checkpoint-{checkpoint}/"
        model = PeftModel.from_pretrained(
            model, 
            checkpoint_weights,
        ).merge_and_unload()    # TODO: does this do .to(device) automatically?
    
    if inference:
        model.config.use_cache = True
        model.eval()

    return model


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        padding_side="right",  
        add_eos_token=True,
        add_bos_token=True,
        trust_remote_code=True
    )
    return tokenizer