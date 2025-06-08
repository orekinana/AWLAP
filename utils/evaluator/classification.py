import json
import os
import pandas as pd
from transformers import AutoTokenizer
from ssap.ft_seq2seq.model import SsapSeq2SeqMultiLevelModel
from ssap.ft_cls.model import SsapClassificationCrossMultiLevelModel, SsapClassificationMultiLevelDistributionConcatModel


class Seq2SeqEvaluator:


    def __init__(self, ModelClass, checkpoint_dir: str, testset_path: str, region_index_map_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=False)
        self.model = ModelClass.from_pretrained(checkpoint_dir).to('cuda:0')
        self.testset_path = testset_path

        with open(region_index_map_path, 'r') as f:
            self.region_index_map = json.load(f)
        self.region_text_map = {
            'district': { v: k for k, v in self.region_index_map['district'].items() },
            'town': { v: k for k, v in self.region_index_map['town'].items() },
            'community': { v: k for k, v in self.region_index_map['community'].items() }
        }


    def evaluate(self, batch_size: int = 32) -> float:
        test_df = pd.read_csv(self.testset_path, delimiter='\t')
        test_df = test_df[~test_df['locate_label'].str.contains('\|\|\|\|\|\|')]
        test_df = test_df[~test_df['locate_label'].str.contains('\|\|\|\|\|')]
        outputs, labels = [], test_df['locate_label'].replace(to_replace=r'^(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)$', value=r'\3|\4|\5', regex=True)
        for curr_idx in range(0, test_df.shape[0], batch_size):
            batch_df = test_df.iloc[curr_idx:curr_idx + batch_size]
            batch_queries = [ b for b in batch_df['address'] ]
            batch_tokens = self.tokenizer(batch_queries, padding=True, truncation=True, max_length=128, return_tensors='pt')
            batch_tokens = { k: v.to(self.model.encoder.device) for k, v in batch_tokens.items() }
            output_ids = self.model.predict(batch_tokens)
            batch_outputs = [ '|'.join(self.id2text(ids)) for ids in output_ids ]
            outputs += batch_outputs
            # print(f'[{curr_idx}/{test_df.shape[0]}] [{curr_idx/test_df.shape[0]:.2%}]')
        # print(outputs)
        return sum( int(o == l) for o, l in zip(outputs, labels) ) / len(outputs)


    def id2text(self, ids: tuple[int, int, int]) -> tuple[str, str, str]:
        return self.region_text_map['district'][max(3, ids[0].item())].split('|')[-1], \
            self.region_text_map['town'][max(3, ids[1].item())].split('|')[-1], \
            self.region_text_map['community'][max(3, ids[2].item())].split('|')[-1]


def main():
    data_dir = 'data/datasets'
    testset_path = f'{data_dir}/test_label.csv'
    region_index_map_path = f'{data_dir}/region_index_map.json'

    # exp_dir = '/home/huiling/code/address-mapping/data/models/bge-ft-cls/cross-multi-level'
    # for step in ('step2', 'step1'):
    #     step_dir = f'{exp_dir}/{step}'
    #     checkpoint_dirs = sorted([ d for d in os.listdir(step_dir) if d.startswith('checkpoint') ], key=lambda x: int(x.split('-')[-1]), reverse=True)
    #     for checkpoint_dir_name in checkpoint_dirs[::1]:
    #         checkpoint_dir = f'{step_dir}/{checkpoint_dir_name}'
    #         evaluator = Seq2SeqEvaluator(
    #             SsapClassificationCrossMultiLevelModel,
    #             checkpoint_dir,
    #             testset_path,
    #             region_index_map_path
    #         )
    #         res = evaluator.evaluate()
    #         print(f'[{step}] [{checkpoint_dir_name}] {res:.4%}')
    exp_dir = '/home/huiling/code/address-mapping/data/models/bge-ft-cls/cross-multi-level-distri-concat'
    for checkpoint_dir_name in ('fixencoderFalse-full-addr-cls', 'fixencoderFalse-full-addr-wo-community-cls', 'fixencoderFalse-poi-cls'):
        checkpoint_dir = f'{exp_dir}/{checkpoint_dir_name}'
        evaluator = Seq2SeqEvaluator(
            SsapClassificationMultiLevelDistributionConcatModel,
            checkpoint_dir,
            testset_path,
            region_index_map_path
        )
        res = evaluator.evaluate()
        print(f'[{checkpoint_dir_name}] {res:.4%}')


if __name__ == '__main__':
    main()
