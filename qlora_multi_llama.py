from huggingface_hub import login
login(token='hf_asfd')


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_from_disk, load_dataset
from trl import SFTTrainer, setup_chat_format

from accelerate import PartialState
device_string = PartialState().process_index
device_map = {'':device_string}

from accelerate import init_empty_weights
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="DEBUG")

from accelerate import infer_auto_device_map
from accelerate import dispatch_model

import traceback

# Configuration
model_path = "meta-llama/Llama-3.1-8B-Instruct"
dataset_name = "jackhhao/jailbreak-classification"
new_model = "raft_llama-3dot1-8b"
torch_dtype = torch.bfloat16
save_path = "./finetuned_model/"


if __name__=="__main__":
    # Load the dataset and split into train and test sets
    data = load_dataset(dataset_name)

    # Load tokenizer and configure padding
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'right'

    # Configure BitsAndBytes for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                 quantization_config=bnb_config, 
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto"
                                                )
    
    # Infer device map for your model and available devices
    # device_map = infer_auto_device_map(model, max_memory={0: "24GiB", 1: "24GiB"})
    
    # Dispatch the model across the devices
    # dispatch_model(model, device_map=device_map)
    dispatch_model(model)#, device_map=device_map)    
    print("DEVICE MAP:", device_map)

    # Prepare model and tokenizer for chat-based input formatting    
    model, tokenizer = setup_chat_format(model, tokenizer)

    # Define LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj',]
    )
    model = get_peft_model(model, peft_config)

    # Setup training arguments
    training_arguments = TrainingArguments(
        output_dir=new_model,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=0.2,
        logging_steps=1,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        dataset_text_field="prompt",
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=4096,
        args=training_arguments,
    )

    # Train and save fine-tuned model
    try:
        logger.info("********** Start Training **********")
        trainer.train()
        logger.info("********** End Training **********")
    except Exception as e:
        logger.error(f"++++++++++ERROR++++++++++\n{e}\n+++++++++++++++++++++++++")
        logger.debug(traceback.format_exc())
        
    model.config.use_cache = True
    trainer.save_model(save_path)
