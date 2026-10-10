"""
TRL + PEFT LoRA SFT of Qwen3-VL (launched by train.sh; every hyper-parameter comes from the env
set by pipelines/common/hparams.sh).

Data: prompt/completion vision rows (prep_data.py) -> loss on the completion only.
SFTConfig is created BEFORE the model so accelerate/DeepSpeed ZeRO-3 can partition while loading;
in replicate mode every rank loads the model straight onto its own GPU (no 67 GB CPU copy per rank).
LoRA: language-model q/k/v/o/gate/up/down only (regex); the ViT, mergers and lm_head stay frozen.
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

ENV = os.environ
LORA_REGEX = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def build_args(run):
    save_steps = int(ENV["SAVE_STEPS"])
    return SFTConfig(
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(ENV["LOGGING_STEPS"]),
        save_strategy="steps" if save_steps > 0 else "no",
        save_steps=save_steps if save_steps > 0 else 500,
        report_to="none",
        seed=int(ENV["SEED"]),
        dataloader_num_workers=int(ENV["DATALOADER_WORKERS"]),
        ddp_find_unused_parameters=False,
        max_length=None,                 # never truncate (image tokens / answers)
        completion_only_loss=True,
    )


def load_processor(model_dir):
    processor = AutoProcessor.from_pretrained(model_dir)
    image_processor = processor.image_processor
    image_processor.size = {"shortest_edge": int(ENV["MIN_PIXELS"]), "longest_edge": int(ENV["MAX_PIXELS"])}
    for attr, key in (("min_pixels", "MIN_PIXELS"), ("max_pixels", "MAX_PIXELS")):
        try:  # read-only property (derived from size) in some transformers versions
            setattr(image_processor, attr, int(ENV[key]))
        except AttributeError:
            pass
    return processor


def main():
    run = Path(ENV["RUN"])
    args = build_args(run)                                  # before from_pretrained (ZeRO-3 init)
    model_kwargs = {"dtype": torch.bfloat16, "attn_implementation": ENV["ATTN_IMPL"]}
    if ENV["STRATEGY"] == "replicate":
        model_kwargs["device_map"] = {"": int(ENV.get("LOCAL_RANK", "0"))}
    model = AutoModelForImageTextToText.from_pretrained(ENV["MODEL"], **model_kwargs)
    processor = load_processor(ENV["MODEL"])

    data_file = Path(ENV["TRL_DATA"]) / f"{ENV['TRAIN_SPLIT']}.jsonl"
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    peft_config = LoraConfig(r=int(ENV["LORA_R"]), lora_alpha=int(ENV["LORA_ALPHA"]),
                             lora_dropout=float(ENV["LORA_DROPOUT"]), bias="none", task_type="CAUSAL_LM",
                             target_modules=LORA_REGEX)
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=processor,
                         peft_config=peft_config)
    # transformers marks Qwen3-VL `accepts_loss_kwargs = False`, so Trainer would not pass num_items_in_batch
    # and TRL would average the loss per micro-batch (= per SAMPLE at micro-batch 1). TRL computes the loss
    # itself from per-token log-probs and honors num_items_in_batch, so turn it on: the loss is then normalized
    # by the trained tokens of the whole global batch (token-weighted) -- the objective all 5 pipelines use.
    trainer.model_accepts_loss_kwargs = True
    trainer.train()
    trainer.save_model(str(run / "adapter"))               # adapter only (gathered under ZeRO-3)


if __name__ == "__main__":
    main()
