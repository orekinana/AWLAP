from typing import Any
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
import random
import hashlib


class AddrClsDataset(Dataset):


    def __init__(self, data: list, t_region_path: str) -> None:
        super().__init__()

        self.district_map, self.town_map, self.community_map = self.preprocess_t_region_data(t_region_path)
        self.num_districts = len(self.district_map)
        self.num_towns = len(self.town_map)
        self.num_communities = len(self.community_map)

        self.data = data
        for record in self.data:
            try:
                record['district_id'] = self.district_map[record['district']]
                record['town_id'] = self.town_map[(record['district'], record['town'])]
                record['community_id'] = self.community_map[(record['district'], record['town'], record['community'])]
            except:
                import traceback
                traceback.print_exc()
                import pdb; pdb.set_trace()


    def __getitem__(self, index: int) -> dict[str, str]:
        return self.data[index]


    def __len__(self) -> int:
        return len(self.data)


    @classmethod
    def preprocess_t_region_data(cls, t_region_path: str) -> tuple[dict, dict, dict]:
        df = pd.read_csv(t_region_path, delimiter='\t', low_memory=False)
        df = df[(df['del_stat'] < 1) & ((df['region_type'] >= 4) | (df['region_type'] <= 5))]
        district_map, town_map, community_map = {}, {}, {}
        for addr in df['full_address']:
            if not isinstance(addr, str):
                continue
            try:
                _, _, district, town, community, *_ = addr.split('|')
                if district not in district_map:
                    district_map[district] = len(district_map)
                if (district, '') not in town_map:
                    town_map[(district, '')] = len(town_map)
                if (district, '', '') not in community_map:
                    community_map[(district, '', '')] = len(community_map)

                if (district, town) not in town_map:
                    town_map[(district, town)] = len(town_map)
                if (district, town, '') not in community_map:
                    community_map[(district, town, '')] = len(community_map)

                if (district, town, community) not in community_map:
                    community_map[(district, town, community)] = len(community_map)
            except:
                print(addr)
                exit()
        return district_map, town_map, community_map


    @classmethod
    def preprocess_train_dataset(cls, train_data_path: str, split_ratio: float, one_resident_upper_num: int, absent_ratio: float) -> list[dict]:

        def calculate_md5(string):
            md5_hash = hashlib.md5()
            md5_hash.update(string.encode('utf-8'))
            md5_value = md5_hash.hexdigest()
            decimal_value = int(md5_value, 16)
            return decimal_value

        train, test, data = [], [], []
        with open(train_data_path, 'r') as f:
            for l in f:

                addr, label = l.strip().split('\t')
                _, _, district, town, community, resident, *_ = label.split('|')

                r = {
                    'query': addr,
                    'gt': label,
                    'district': district,
                    'town': town,
                    'community': community,
                    'resident': resident,
                }
                data.append(r)
        data = cls.balance_train_dataset(data, upper_num=one_resident_upper_num)
        duplicate = set()
        absent_ratio = 0.4
        for r in data:
            ratio = (calculate_md5(r['query']) % 10000) / 10000
            # print(ratio)
            if ratio < split_ratio:
                # add noise
                absent_resident = r['resident'].replace('小区', '')
                index = random.randint(0, len(absent_resident) - int(absent_ratio * len(absent_resident)))
                noise_resident = absent_resident[:index] + absent_resident[index + int(absent_ratio * len(absent_resident)):]
                new_r = dict(r)
                new_r['query'] = r['query'].replace(r['resident'], noise_resident) 
                train.append(new_r)
            else:
                if r['query'] in duplicate:
                    continue
                # add noise
                duplicate.add(r['query'])
                absent_resident = r['resident'].replace('小区', '')
                index = random.randint(0, len(absent_resident) - int(absent_ratio * len(absent_resident)))
                noise_resident = absent_resident[:index] + absent_resident[index + int(absent_ratio * len(absent_resident)):]
                new_r = dict(r)
                new_r['query'] = r['query'].replace(r['resident'], noise_resident)
                test.append(new_r)

        # split_size = int(len(data) * split_ratio)
        # random.shuffle(data)
        print('train size:', len(train), 'test size:', len(test))
        return train, test

    @classmethod
    def balance_train_dataset(cls, data: list[dict], upper_num: int) -> list[dict]:
        resident_num = {}
        balance_data = {}
        for record in data:
            if (record['resident'] in resident_num) and (resident_num[record['resident']] > upper_num):
                continue
            if record['resident'] not in balance_data:
                balance_data[record['resident']] = [record]
                resident_num[record['resident']] = 1
            else:
                balance_data[record['resident']].append(record)
                resident_num[record['resident']] += 1
        for resident, num in resident_num.items():
            if num < upper_num:
                copied_dict = []
                for _ in range(upper_num // len(balance_data[resident])):
                    copied_dict.extend(balance_data[resident].copy())
                balance_data[resident] = copied_dict
    
        list_of_dicts = []
        distribution = []
        for _, value in balance_data.items():
            list_of_dicts.extend(value)
            distribution.append(len(value))
        print('total residents:', len(distribution), 'max residents num:', max(distribution), 'min residents num:', min(distribution), 'avg residents num:', sum(distribution) / len(distribution))
        print('median residents num:', sorted(distribution)[len(distribution) // 2], '1/4 residents num:', sorted(distribution)[len(distribution) // 4], '3/4 residents num:', sorted(distribution)[3 * (len(distribution) // 4)])

        return list_of_dicts


class AddrClsDataCollator(DataCollatorWithPadding):


    def __call__(self, batch_data: list[dict[str, str]]) -> dict[str, Any]:
        queries = [ data['query'] for data in batch_data ]
        query_tokens = self.tokenizer(queries, padding=True, truncation=True, max_length=128, return_tensors='pt')

        district_ids = torch.LongTensor([ d['district_id'] for d in batch_data ])
        town_ids = torch.LongTensor([ d['town_id'] for d in batch_data ])
        community_ids = torch.LongTensor([ d['community_id'] for d in batch_data ])

        return { 'inputs': query_tokens, 'district_ids': district_ids, 'town_ids': town_ids, 'community_ids': community_ids, 'texts': queries }



if __name__ == '__main__':
    DATA_DIR              = 'data'
    dataset_config = {'jdl': {'testset_path' : f'{DATA_DIR}/datasets/train/jdl_cls_dataset.txt', 'split_ratio' : 0.99, 'one_resident_upper_num' : 500}, 
                        'poi' : {'testset_path' : f'{DATA_DIR}/datasets/test/poi.csv', 'split_ratio' : 0.1, 'one_resident_upper_num' : 5}}
    dataset_mode = 'poi'
    absent_ratio = 0.4
    _, test_data = AddrClsDataset.preprocess_train_dataset(dataset_config[dataset_mode]['testset_path'], split_ratio=dataset_config[dataset_mode]['split_ratio'], one_resident_upper_num=dataset_config[dataset_mode]['one_resident_upper_num'], absent_ratio=absent_ratio)
    # pd.DataFrame(test_data).to_csv(f'{DATA_DIR}/datasets/test/{dataset_mode}_absent{absent_ratio}.csv')