from pathlib import Path
from transformers import HfArgumentParser, set_seed, AutoTokenizer

from ecai.ft_cls.arguments import DataArguments, ModelArguments, FineTuneTrainingArguments
from ecai.ft_cls.model import (
    MultiLevelClsModel,
)
from ecai.ft_cls.data import AddrClsDataset, AddrClsDataCollator
from ecai.ft_cls.trainer import AddrFineTuneTrainer
from ecai.utils.log import init_logger
from torch.utils.data import Subset, random_split
# from evaluate.evaluator.eval import vector_search, init_region_index_map


def main():
    argument_parser = HfArgumentParser((DataArguments, ModelArguments, FineTuneTrainingArguments))
    data_args, model_args, training_args = argument_parser.parse_args_into_dataclasses()

    logger = init_logger('addr_ft_cls')
    logger.info(training_args)

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    set_seed(training_args.seed)

    train_data, _ = AddrClsDataset.preprocess_train_dataset(data_args.train_data_path, split_ratio=0.5, one_resident_upper_num=500, absent_ratio=0.4)
    # few shot train data
    # train_data, _ = AddrClsDataset.preprocess_train_dataset(data_args.train_data_path, split_ratio=1, one_resident_upper_num=5, absent_ratio=0.4)
    train_dataset = AddrClsDataset(train_data, data_args.t_region_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    data_collator = AddrClsDataCollator(tokenizer)

    try:
        model = MultiLevelClsModel.from_pretrained(model_args.model_name_or_path)
    except:
        print('failed to load old model')
        model = MultiLevelClsModel(
            model_name_or_path=model_args.model_name_or_path,
            # bge-small hidden size=512
            # text2vec-base-chinese hidden size=768
            hidden_size=768,
            hidden_dropout=0.1,
            num_districts=train_dataset.num_districts,
            num_towns=train_dataset.num_towns,
            num_communities=train_dataset.num_communities,
            num_heads = model_args.num_heads,
            dis_hidden_size = 5,
            num_level = 3,
            scl_alpha = model_args.scl_alpha,
        )

    original_output_dir = training_args.output_dir
    for               name, fix_encoder, scl, num_epochs, lr in [
        (f'gte-base-1st-fix-encoder-2epoch-lr3-0.4noise-{model_args.scl_alpha}scl-0.5data-pos-gai',      True,    False,      2, 1e-3),
        (f'gte-base-2nd-fine-tune-3epoch-lr4-0.4noise-{model_args.scl_alpha}scl-0.5data-pos-gai',        False,   True ,      3, 1e-4),
        # (f'smallbge-2nd-fine-tune-50epoch-lr4-0.4noise-{model_args.scl_alpha}scl-0.5data-woscl-poi-fineture',       False,   False,       50, 1e-4),
    ]:
        print(f'[CONFIG] name: {name} fix_encoder: {fix_encoder} num_epochs: {num_epochs} lr: {lr}')
        for n, p in model.named_parameters():
            if n.startswith('encoder'):
                p.requires_grad = not fix_encoder
        training_args.output_dir = f'{original_output_dir}/{name}'
        training_args.num_train_epochs = num_epochs
        training_args.learning_rate = lr
        model.zero_grad()
        model.scl = scl
        model.train()
        trainer = AddrFineTuneTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.train()

        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)



if __name__ == '__main__':
    main()