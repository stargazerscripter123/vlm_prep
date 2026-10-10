"""
Unsloth LoRA SFT of Qwen3-VL (launched by train.sh; hyper-parameters from pipelines/common/hparams.sh).

- Unsloth is imported first (it patches transformers / TRL).
- QUANT=bf16 (default) or 4bit. 4-bit is only used when the model does not fit one GPU (32B on 48 GB
  cards, reported as a separate reference row): NF4 with the vision tower, mergers, lm_head and
  embeddings kept in BF16 via llm_int8_skip_modules.
- LoRA on language-model attention + MLP only (finetune_vision_layers=False).
- UnslothVisionDataCollator(resize="max") keeps native resolution (the default "min" falls back to a
  512 px width cap for Qwen3-VL); images come from the top-level "images" column; loss on the
  assistant response only.
- Multi-GPU = DDP (torchrun); Unsloth pins each rank to its LOCAL_RANK GPU.
"""

import os  # noqa: I001  (unsloth must be imported before transformers / trl / peft)

import unsloth  # noqa: F401
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import json
from pathlib import Path

import torch
from trl import SFTConfig, SFTTrainer

ENV = os.environ
KEEP_BF16 = ["visual", "merger", "lm_head", "embed_tokens"]


def load_model():
    kwargs = {"max_seq_length": int(ENV["MAX_LENGTH"]), "dtype": torch.bfloat16,
              "use_gradient_checkpointing": "unsloth"}
    if ENV.get("QUANT", "bf16") == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs.update(load_in_4bit=False, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=KEEP_BF16))
    else:
        kwargs.update(load_in_4bit=False, load_in_16bit=True)
    model, processor = FastVisionModel.from_pretrained(ENV["MODEL"], **kwargs)
    image_processor = processor.image_processor
    image_processor.size = {"shortest_edge": int(ENV["MIN_PIXELS"]), "longest_edge": int(ENV["MAX_PIXELS"])}
    for attr, key in (("min_pixels", "MIN_PIXELS"), ("max_pixels", "MAX_PIXELS")):
        try:  # read-only property (derived from size) in some transformers versions
            setattr(image_processor, attr, int(ENV[key]))
        except AttributeError:
            pass
    return model, processor


def main():
    run = Path(ENV["RUN"])
    model, processor = load_model()
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=int(ENV["LORA_R"]), lora_alpha=int(ENV["LORA_ALPHA"]), lora_dropout=float(ENV["LORA_DROPOUT"]),
        bias="none", random_state=int(ENV["SEED"]), use_rslora=False, loftq_config=None)
    FastVisionModel.for_training(model)

    rows = [json.loads(line) for line in
            (Path(ENV["UNSLOTH_DATA"]) / f"{ENV['TRAIN_SPLIT']}.jsonl").read_text().splitlines() if line.strip()]
    collator = UnslothVisionDataCollator(
        model, processor, resize="max", train_on_responses_only=True,
        instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

    save_steps = int(ENV["SAVE_STEPS"])
    args = SFTConfig(
        output_dir=str(run / "out"),
        max_steps=int(ENV["TOTAL_STEPS"]),
        per_device_train_batch_size=int(ENV["MICRO_BS"]),
        gradient_accumulation_steps=int(ENV["GRAD_ACC"]),
        learning_rate=float(ENV["LR"]),
        lr_scheduler_type=ENV["LR_SCHEDULER"],
        warmup_steps=int(ENV["WARMUP_STEPS"]),
        weight_decay=float(ENV["WEIGHT_DECAY"]),
        adam_beta1=float(ENV["ADAM_BETA1"]),
        adam_beta2=float(ENV["ADAM_BETA2"]),
        adam_epsilon=float(ENV["ADAM_EPS"]),
        max_grad_norm=float(ENV["MAX_GRAD_NORM"]),
        optim="adamw_torch",
        bf16=True,
        logging_steps=int(ENV["LOGGING_STEPS"]),
        save_strategy="steps" if save_steps > 0 else "no",
        save_steps=save_steps if save_steps > 0 else 500,
        report_to="none",
        seed=int(ENV["SEED"]),
        dataloader_num_workers=int(ENV["DATALOADER_WORKERS"]),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=int(ENV["MAX_LENGTH"]),
    )
    trainer = SFTTrainer(model=model, processing_class=processor, data_collator=collator,
                         train_dataset=rows, args=args)
    trainer.train()
    if trainer.is_world_process_zero():
        model.save_pretrained(str(run / "adapter"))


if __name__ == "__main__":
    main()
