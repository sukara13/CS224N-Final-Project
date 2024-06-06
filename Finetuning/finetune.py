import torch, wandb
import argparse

from datasets import load_dataset
from trl import SFTTrainer  # TODO: Trainer or SFTTrainer?
from transformers import TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from loading import load_base_model, load_tokenizer
from inference.utils import *
from constants import *

api_key = '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'

# Constants for training arguments
WARMUP_STEPS = 500
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 2
OPTIMIZER = "adamw_torch"
LEARNING_RATE = 5e-5

# Logging and saving arguments
SAVE_STEPS = 10
LOGGING_STEPS = 10
LOG_DIR = "./logs"

# Sequence length
MAX_SEQUENCE_LENGTH = 1512

# PEFT settings
RANK = 6
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()
elif torch.backends.mps.is_available() and args.variant == 'vanilla':
    device = 'mps'

#hf_endpoint = "sukara13/finetuned-car-gemma-7b-it"
verbose = True

dataset_path = '/home/sukara/CS224/datasets'
save_weights_to = "./saved_weights"
model_id = 'google/gemma-7b-it'

class FineTune():

    def __init__(
        self,
        device,
        save_weights_to,
        #hf_endpoint,
        verbose=True,
    ):
        self.device = device
        self.save_weights_to = save_weights_to
        #self.hf_endpoint = hf_endpoint
        self.verbose = verbose

        # these two attributes get set with prepare_model method
        self.model = None
        self.tokenizer = None

    """
    Finetuning dataset: train: car_train.csv, test: car_test.csv
    Assumes the dataset is a .csv file with columns "Question" and "Answer".
    """
    def get_dataset(self, path):
        data_files = {"train": f"{path}/car_train.csv", "test": f"{path}/car_test.csv"}
        dataset = load_dataset("csv", data_files=data_files, delimiter='\t', column_names=["System", "User", "Recommendation"])
        train_dataset =  dataset["train"]
        if self.verbose: print("=> num datapoints in train: ", len(train_dataset))
        return train_dataset

    """
    This loads the model, tokenizer, and peft framework
    The exact framework is as follows: base -> quantize -> peft
    """
    def prepare_model(self, model_id):
        # loading model
        base_model = load_base_model(model_id)
        base_model.config.use_cache = False
        # base_model.gradient_checkpointing_enable()  # turn on if training gives OOM errors

        quantized_model = prepare_model_for_kbit_training(base_model)
        peft_config = LoraConfig(
            r=RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,   # query, key, value, output projections
        )
        model = get_peft_model(quantized_model, peft_config)
        self.model = model.to(device)

        # loading tokenizer
        tokenizer = load_tokenizer(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        return peft_config

    def formatting_func(example, tokenizer):
        """
        Apply the chat template to a single question-answer pair example.

        Args:
            example (dict): A dictionary with "Question" and "Answer".
            tokenizer (PreTrainedTokenizer): The tokenizer used for applying the chat template.

        Returns:
            str: The formatted conversation as a string.
        """
        chat = [
            {"role": "system", "content": example["System"]},
            {"role": "user", "content": example["User"]},
            {"role": "model", "content": example["Recommendation"]}
        ]

        # Apply the chat template to the conversation
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        return prompt

    """
    Creates the TrainingArguments and runs SFTTrainer.
    """
    def train(self, train_dataset, peft_config):
        training_args = TrainingArguments(
            output_dir=save_weights_to,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            gradient_checkpointing=True,
            optim=OPTIMIZER,
            learning_rate=LEARNING_RATE,

            # logging
            # NOTE: add eval if necessary..
            save_steps=SAVE_STEPS,
            save_strategy="steps",
            logging_dir=LOG_DIR,
            logging_steps=LOGGING_STEPS,
            #report_to="wandb",
        )
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            #eval_dataset=eval_dataset,
            peft_config=peft_config,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            tokenizer=self.tokenizer,
            args=training_args,
            formatting_func=lambda example: self.formatting_func(example, self.tokenizer),  # look up apply_chat_template
            packing=False,  # TODO: set to true?
        )

        # Train and save model locally
        if self.verbose: print("=> START TRAINING!")
        trainer.train()

        # change this path, otherwise will overwrite
        if self.verbose: print("=> SAVING WEIGHTS")
        trainer.model.save_pretrained(self.save_weights_to)
        #wandb.finish()

        self.model.config.use_cache = True
        self.model.eval()
        del self.model, trainer
        if self.device == "cuda":
            torch.cuda.empty_cache()


    """
    Pushes model weights to huggingface and returns link
    
    def push_to_hf(self):
        # push the weights in self.save_weights_to path to self.hf_endpoint. 
        # Then, return the link to the endpoint
        pass
    """


if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=> using device ", device)
    #wandb.login(key=WANDB_API_KEY)
    #run = wandb.init(project='Fine tuning mistral 7B', job_type="training", anonymous="allow")

    # Get configs
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--model', type=str, help='model', required=True)
    args = parser.parse_args()
    #dataset_path, save_weights_to, model_id, formatting_func = get_training_configs(args)
    print("model: ", (model_id))

    # Set up class and finetune
    FT = FineTune(
        device=device,
        save_weights_to=save_weights_to,
        #hf_endpoint=None,   # TODO: add support for this
        verbose=True,
    )
    train_dataset = FT.get_dataset(dataset_path)
    peft_configs = FT.prepare_model(model_id)
    FT.train(train_dataset, peft_configs)
