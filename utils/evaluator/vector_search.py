import os
import json
import numpy as np
import pandas as pd
from text2vec import SentenceModel


class VectorSearchEvaluator:


    def __init__(self, model_name: str, model_dir: str, data_dir: str = 'data') -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None


    def evaluate(self) -> dict[str, float]:
        community_db = self.__init_community_db(use_cache=False, save_cache=False)
        query_db = self.__init_query_db(community_db['communities'], use_cache=False, save_cache=False)
        community_mat = np.array(community_db['vectors'])
        query_mat = np.array(query_db['vectors'])
        similarities = query_mat.dot(community_mat.T)
        sorted_similarities_indices = np.argsort(similarities, axis=1)

        # 准确率、MAP
        search_results = [
            [ community_db['communities'][i] for i in indices[::-1] ]
            for indices in sorted_similarities_indices
        ]
        p1 = self.__calc_precision(search_results, query_db['labels'], n=1)
        p5 = self.__calc_precision(search_results, query_db['labels'], n=5)
        p10 = self.__calc_precision(search_results, query_db['labels'], n=10)
        map = self.__calc_map(search_results, query_db['labels'])

        # Detail
        lines = []
        for query, label, simis, indices in zip(query_db['queries'], query_db['labels'], similarities, sorted_similarities_indices):
            line1 = [ community_db['communities'][i] for i in indices[-20:][::-1] ]
            line2 = [ f'{simis[i]:.4f}' for i in indices[-10:][::-1] ]
            if line1[0] != label:
                lines.append(f'{query},{label}')
                lines.append(f',{",".join(line1)}')
                lines.append(f',{",".join(line2)}')
        with open(f'{self.data_dir}/experiments/eval/detail_{self.model_name}.csv', 'w') as f:
            f.write('\n'.join(lines))

        return { 'Precision@1': p1, 'Precision@5': p5, 'Precision@10': p10, 'MAP': map }


    @classmethod
    def __calc_precision(cls, search_results: list[list[str]], labels: list[str], n: int = 1) -> float:
        # return sum( int(l in r[:n]) for r, l in zip(search_results, labels) ) / len(labels)
        return sum( int( any( s in l for s in r[:n] ) ) for r, l in zip(search_results, labels) ) / len(labels)
    


    @classmethod
    def __calc_map(cls, search_results: list[list[str]], labels: list[str]) -> float:
        def calc_one_map(search_result: list[str], label: str) -> float:
            return 1.0 / (search_result.index(label) + 1)
        return float(np.mean([ calc_one_map(r, l) for r, l in zip(search_results, labels) ]))


    def __init_community_db(self, use_cache: bool, save_cache: bool) -> dict[str, list]:
        output_path = f'{self.data_dir}/cache/vectors/community_db_{self.model_name}.json'
        if use_cache and os.path.exists(output_path):
            with open(output_path, 'r') as f:
                return json.load(f)
        with open(f'{self.data_dir}/datasets/full_community_list.txt', 'r') as input_f:
            output = { 'communities': [], 'vectors': [] }
            lines = input_f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip().replace('北京市|北京市|', '').replace('|', '').replace('$', '')
                if not line:
                    continue
                output['communities'].append(line)
            output['vectors'] = [ [ float(v) for v in vector ] for vector in self.__embedding(output['communities']) ]
            if save_cache:
                with open(output_path, 'w') as output_f:
                    json.dump(output, output_f, ensure_ascii=False)
            return output


    def __init_query_db(self, communities: list[str], use_cache: bool, save_cache: bool) -> dict[str, list]:
        output_path = f'{self.data_dir}/cache/vectors/query_db_{self.model_name}.json'
        if use_cache and os.path.exists(output_path):
            with open(output_path, 'r') as f:
                return json.load(f)
        communities = set(communities)
        output = { 'queries': [], 'vectors': [], 'labels': [] }
        # test_df = pd.read_csv(f'{self.data_dir}/datasets/test_label.csv', delimiter='\t')
        test_df = pd.read_csv(f'{self.data_dir}/datasets/test_label.csv', delimiter='\t')
        test_df = test_df[~test_df['locate_label'].str.contains('\|\|\|\|\|\|')]
        test_df = test_df[~test_df['locate_label'].str.contains('\|\|\|\|\|')]
        test_df['locate_label'] = test_df['locate_label'].replace(to_replace=r'^(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)$', value=r'\3\4\5\9', regex=True).replace(to_replace='$', value='')
        # import pdb; pdb.set_trace()
        for idx, (_, r) in enumerate(test_df.iterrows()):
            if r['locate_label'] not in communities:
                continue
            output['queries'].append(r['address'])
            output['labels'].append(r['locate_label'])
        output['vectors'] = [ [ float(v) for v in vector ] for vector in self.__embedding(output['queries']) ]
        if save_cache:
            with open(output_path, 'w') as output_f:
                json.dump(output, output_f, ensure_ascii=False)
        return output


    def __embedding(self, texts: list[str]) -> np.array:
        if not isinstance(self.model, SentenceModel):
            self.model = SentenceModel(self.model_dir)
        return self.model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)


def main():
    from datetime import datetime
    start_dt = datetime.now()
    DATA_DIR = 'data'
    models = [
        # ('bge-base-zh-v1.5', f'{DATA_DIR}/models/bge-base-zh-v1.5'),
        ('bge-small-zh-v1.5', f'{DATA_DIR}/models/bge-small-zh-v1.5'),
        # ('gte-chinese-base', f'{DATA_DIR}/models/nlp_gte_sentence-embedding_chinese-base'),
        # ('gte-chinese-small', f'{DATA_DIR}/models/nlp_gte_sentence-embedding_chinese-small'),
        # ('text2vec-base-chinese', f'{DATA_DIR}/models/text2vec-base-chinese'),
        # ('bge-small-pretrain-default', f'{DATA_DIR}/models/bge-small-pretrain/default'),
        # ('bge-small-ft-1', f'{DATA_DIR}/models/bge-ft/contrastive/level3'),
        # ('bge-small-ft-1', f'{DATA_DIR}/models/bge-ft/cosine'),
        # ('bge-small-ft-cls1-1000', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-concat-mixin/fixencoderFalse-full-addr-cls'),
        # ('bge-small-ft-cls2-1000', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-concat-mixin/fixencoderFalse-mixin-cls'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/first-train-cls-1100'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/first-train-dis-1001'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-1-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-1-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-2-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-2-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-3-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-3-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-4-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-4-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-5-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-5-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-6-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-6-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-7-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-7-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-8-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-8-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-9-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-9-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-10-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-10-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-11-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-11-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-12-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-12-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-13-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-13-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-14-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-14-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-15-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-15-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-16-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-16-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-17-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-17-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-18-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-18-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-19-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-19-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-20-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-20-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-21-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-21-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-22-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-22-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-23-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-23-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-24-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-24-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-25-0110'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-dis-25-0011'),
        # ('bge-small-ft-scl', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-scl/train-cls-only-0100'),

        # ('bge-small-ft-cls-5000', f'{DATA_DIR}/models/bge-ft/contrastive/level3-1'),
        # ('bge-small-ft-cls-10000', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-custom-kernel-pooling/step2/checkpoint-10000'),
        # ('bge-small-ft-cls-15000', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-custom-kernel-pooling/step2/checkpoint-15000'),
        # ('bge-small-ft-cls-20000', f'{DATA_DIR}/models/bge-ft-cls/cross-multi-level-distri-custom-kernel-pooling/step2/checkpoint-20000'),
    ]
    results = []
    for model_name, model_dir in models:
        evaluator = VectorSearchEvaluator(model_name, model_dir)
        res = evaluator.evaluate()
        results.append({ 'name': model_name, **res })
    results_df = pd.DataFrame(results)
    print(results_df)
    end_dt = datetime.now()
    print(end_dt - start_dt)
    results_df.to_csv(f'{DATA_DIR}/experiments/eval/results.csv', index=False)


if __name__ == '__main__':
    main()
