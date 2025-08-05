import logging
import os
import numpy as np


import transformers
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    HfArgumentParser, set_seed, )
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import is_main_process

from arguments import DataTrainingArguments, ModelArguments
from data import DatasetForPretraining, mlmCollator
from trainer import PreTrainer

logger = logging.getLogger(__name__)


class TrainerCallbackForSaving(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        control.should_save = True


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        ) 

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    training_args.remove_unused_columns = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        

    if training_args.local_rank in (0, -1):
        logger.info("Training/evaluation parameters %s", training_args)
        # logger.info("Model parameters %s", model_args)
        # logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    
    # from transformers import DataCollatorForLanguageModeling ,DataCollatorForWholeWordMask  #xxxxxxxxxx
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    if os.path.basename(model_args.model_name_or_path)=='encoder_model':
        model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path.replace("encoder_model", ""))  #参数文件保存位置有问题，重定向一下
    else:
        model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)     #从huggingface加载
    
    data_collator = mlmCollator(tokenizer)

    dataset = DatasetForPretraining(data_args.train_data)

    trainer = PreTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # eval_dataset=dataset,
    )
    trainer.add_callback(TrainerCallbackForSaving())
    
    if training_args.do_train:
        print("train mode")

        # trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload 
    else:
        # eval时将loss写入df
        print("eval mode")
        output=trainer.evaluate()
        print(output)
        
        import pandas as pd
        model_name_split = model_args.model_name_or_path.split('/')  # 分割路径并获取最后一个元素
        if training_args.local_rank in (0, -1):
            eval_loss = output['eval_loss']
            data_to_write = {model_name_split[-3]: [model_name_split[-2].split("-")[-1]], 'eval_loss': [np.round(eval_loss,3)]}
            df = pd.DataFrame(data_to_write)
            writepath='pretrain_eval_results/'+model_name_split[-3]+".csv"
            print(writepath)
            if not os.path.exists(writepath):
                df.to_csv(writepath, index=False)  # 第一次写入时创建文件并写入表头和数据
            else:
                df.to_csv(writepath, mode='a', header=False, index=False)  # 后续追加写入时仅写入数据

if __name__ == "__main__":
    main()
