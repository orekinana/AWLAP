from dataclasses import dataclass
import json
from typing import Optional
import torch
import torch.nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.file_utils import ModelOutput


class MultiLevelClsModel(torch.nn.Module):


    @dataclass
    class Output(ModelOutput):
        similarities: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None


    def __init__(
        self,
        model_name_or_path: str,
        hidden_size: int,
        hidden_dropout: float,
        num_districts: int,
        num_towns: int,
        num_communities: int,
        num_heads: int,
        dis_hidden_size: int,
        num_level: int,
        scl_alpha: float,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.scl = False
        self.scl_alpha = scl_alpha
        self.hidden_size = hidden_size
        self.num_districts = num_districts
        self.num_towns = num_towns
        self.num_communities = num_communities
        self.hidden_dropout = hidden_dropout
        self.num_heads = num_heads
        self.dis_hidden_size = dis_hidden_size
        self.num_level = num_level

        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = torch.nn.Dropout(hidden_dropout)

        self.district_linear = torch.nn.Sequential(
                                    torch.nn.Linear(hidden_size, hidden_size // 2),
                                    torch.nn.ReLU(),  
                                    torch.nn.Linear(hidden_size // 2, hidden_size // 4),
                                    torch.nn.ReLU(),  
                                    torch.nn.Linear(hidden_size // 4, hidden_size // 8),
                                    torch.nn.ReLU(), 
                                    torch.nn.Linear(hidden_size // 8, num_districts)
                                )
        self.town_linear = torch.nn.Linear(hidden_size, num_towns)
        self.community_linear = torch.nn.Sequential(
                            torch.nn.Linear(hidden_size, hidden_size * 2),
                            torch.nn.ReLU(),  
                            torch.nn.Linear(hidden_size * 2, hidden_size * 4),
                            torch.nn.ReLU(),  
                            torch.nn.Linear(hidden_size * 4, num_communities)
                        )

        for i in range(self.num_heads):
            setattr(self, f'weight_attn_town_{i}', torch.nn.Linear(num_towns, num_towns))
            setattr(self, f'weight_attn_district_{i}', torch.nn.Linear(num_districts, num_districts))

        
        # self.weight_attn_town = [torch.nn.Linear(num_towns, num_towns).to(self.device) for i in range(self.num_heads)]
        # self.weight_attn_district = [torch.nn.Linear(num_districts, num_districts).to(self.device) for i in range(self.num_heads)]

        self.avr_heads_town = torch.nn.Linear(num_towns * self.num_heads, num_towns)
        self.avr_heads_district = torch.nn.Linear(num_districts * self.num_heads, num_districts)

        self.discriminator = Discriminator(dis_hidden_size, num_districts, num_towns, num_communities, num_level).to(self.device)

        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        self.dis_loss_func = torch.nn.BCELoss()
    
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor: # B x k.dim(T or D)
        scores = torch.matmul(q.unsqueeze(-1), k.unsqueeze(-1).transpose(-2, -1)) # B x C x 1 x 1 x T x B = B x C x T
        
        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores.transpose(-2, -1), v.unsqueeze(-1)).squeeze(-1) # B x T x C x B x C x 1 = B x T

        return output


    def forward(self, inputs: dict, district_ids: torch.Tensor, town_ids: torch.Tensor, community_ids: torch.Tensor, **kwargs) -> 'Output':
        '''
        @param input_seq: B x L1 x 1
        @param district_ids: B
        @param town_ids: B
        @param community_ids: B
        '''
        # print(kwargs['texts'])

        embeddings = self.encoder(**inputs, return_dict=True).last_hidden_state[:, 0] # B x D
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings = self.dropout(embeddings)
        district_embedding  = self.district_linear(embeddings)  # B x D
        town_embedding      = self.town_linear(embeddings)      # B x T
        community_embedding = self.community_linear(embeddings)  # B x C

        multi_heads_town, multi_heads_district = [], []
        for i in range(self.num_heads):
            weight_attn_town = getattr(self, f'weight_attn_town_{i}')
            weight_attn_district = getattr(self, f'weight_attn_district_{i}')
            multi_heads_town.append(self.attention(community_embedding, weight_attn_town(town_embedding), community_embedding))
            multi_heads_district.append(self.attention(town_embedding, weight_attn_district(district_embedding), town_embedding))
        
        attn_output_town = self.avr_heads_town(torch.cat(multi_heads_town, dim=1))
        attn_output_distict = self.avr_heads_district(torch.cat(multi_heads_district, dim=1))

        district_loss  = self.loss_func(attn_output_distict, district_ids)
        town_loss      = self.loss_func(attn_output_town, town_ids)
        community_loss = self.loss_func(community_embedding, community_ids)      
          
        district_prob, town_prob, community_prob = self.discriminator(torch.unsqueeze(district_loss, dim=1).detach(), torch.unsqueeze(town_loss, dim=1).detach(), torch.unsqueeze(community_loss, dim=1).detach(), attn_output_distict.detach(), attn_output_town.detach(), community_embedding.detach())
        dis_label_district = (torch.argmax(attn_output_distict, dim=1).view(-1, 1) == torch.unsqueeze(district_ids, dim=1)).int()
        dis_label_town = (torch.argmax(attn_output_town, dim=1).view(-1, 1) == torch.unsqueeze(town_ids, dim=1)).int()
        dis_label_community = (torch.argmax(community_embedding, dim=1).view(-1, 1) == torch.unsqueeze(community_ids, dim=1)).int()

        district_dis_loss = self.dis_loss_func(district_prob, dis_label_district.float())
        town_dis_loss = self.dis_loss_func(town_prob, dis_label_town.float())
        community_dis_loss = self.dis_loss_func(community_prob, dis_label_community.float())
        
        if not self.scl:
            loss = district_loss.mean() + town_loss.mean() + community_loss.mean() + district_dis_loss + town_dis_loss + community_dis_loss
            # print('cls loss: ', (district_loss.mean() + town_loss.mean() + community_loss.mean()).item(), 'dis loss: ', (district_dis_loss + town_dis_loss + community_dis_loss).item())
        else:
            # district_weight = torch.clamp(1 + self.scl_alpha / (F.softmax(attn_output_distict)[torch.arange(len(district_ids)), district_ids] + 1e-6), max=10).detach()
            district_weight = torch.clamp(1 + self.scl_alpha / (district_prob + 1e-6), max=10).detach()
            weight_district_loss = (district_weight * district_loss).mean()

            # town_weight = torch.clamp(1 + self.scl_alpha / (F.softmax(attn_output_town)[torch.arange(len(town_ids)), town_ids] + 1e-6), max=10).detach()
            town_weight = torch.clamp(1 + self.scl_alpha / (1 + town_prob + 1e-6), max=10).detach()
            weight_town_loss = (town_weight * town_loss).mean()

            # community_weight = torch.clamp(1 + self.scl_alpha / (F.softmax(community_embedding)[torch.arange(len(community_ids)), community_ids] + 1e-6), max=10).detach()
            community_weight = torch.clamp(1 + self.scl_alpha / (community_prob + 1e-6), max=10).detach()
            weight_community_loss = (community_weight * community_loss).mean()

            loss = weight_district_loss + weight_town_loss + weight_community_loss + district_dis_loss + town_dis_loss + community_dis_loss

        return self.Output(similarities=None, loss=loss)


    @torch.no_grad()
    def predict(self, inputs: dict, labels: dict) -> torch.Tensor: # B x 3
        from datetime import datetime
        start_time = datetime.now()
        embeddings = self.encoder(**inputs, return_dict=True).last_hidden_state[:, 0] # B x D
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings = self.dropout(embeddings)
        community_embedding = self.community_linear(embeddings)  # B x num_communities
        print('timing: ', (datetime.now() - start_time) / embeddings.shape[0])
        print('finish cls predict')
        district_embedding  = self.district_linear(embeddings)  # B x D
        town_embedding      = self.town_linear(embeddings)      # B x T
        multi_heads_town, multi_heads_district = [], []
        for i in range(self.num_heads):
            weight_attn_town = getattr(self, f'weight_attn_town_{i}')
            weight_attn_district = getattr(self, f'weight_attn_district_{i}')
            multi_heads_town.append(self.attention(community_embedding, weight_attn_town(town_embedding), community_embedding))
            multi_heads_district.append(self.attention(town_embedding, weight_attn_district(district_embedding), town_embedding))
        attn_output_town = self.avr_heads_town(torch.cat(multi_heads_town, dim=1))
        attn_output_distict = self.avr_heads_district(torch.cat(multi_heads_district, dim=1))

        district_loss  = self.loss_func(attn_output_distict, labels['district_ids'])
        town_loss      = self.loss_func(attn_output_town, labels['town_ids'])
        community_loss = self.loss_func(community_embedding, labels['community_ids'])
        _, _, community_prob = self.discriminator(torch.unsqueeze(district_loss, dim=1).detach(), torch.unsqueeze(town_loss, dim=1).detach(), torch.unsqueeze(community_loss, dim=1).detach(), attn_output_distict.detach(), attn_output_town.detach(), community_embedding.detach())
        
        print('finish dis predict')
        mask = (torch.argmax(community_embedding, dim=1).view(-1, 1) == torch.unsqueeze(labels['community_ids'], dim=1)).int()
        return community_embedding, community_prob, mask


    def save(self, output_dir: str) -> None:
        encoder_state_dict = self.encoder.state_dict()
        encoder_state_dict = type(encoder_state_dict)({
            k: v.clone().cpu()
            for k, v in encoder_state_dict.items()
        })
        self.encoder.save_pretrained(output_dir, state_dict=encoder_state_dict)

        rest_model_state_dict = {
            k: v.clone().cpu()
            for k, v in self.state_dict().items()
            if not k.startswith('encoder')
        }
        torch.save(rest_model_state_dict, f'{output_dir}/checkpoint_beyond_encoder.ckpt')

        with open(f'{output_dir}/model_config_beyond_encoder.json', 'w') as f:
            json.dump({
                'hidden_size': self.hidden_size,
                'hidden_dropout': self.hidden_dropout,
                'num_districts': self.num_districts,
                'num_towns': self.num_towns,
                'num_communities': self.num_communities,
                'num_heads': self.num_heads,
                'dis_hidden_size': self.dis_hidden_size,
                'num_level': self.num_level,
                'scl_alpha': self.scl_alpha
            }, f, indent=4, ensure_ascii=False)


    @classmethod
    def from_pretrained(cls, checkpoint_dir: str) -> 'MultiLevelClsModel':
        with open(f'{checkpoint_dir}/model_config_beyond_encoder.json', 'r') as f:
            config = json.load(f)
        model = cls(
            model_name_or_path=checkpoint_dir,
            hidden_size=config['hidden_size'],
            hidden_dropout=config['hidden_dropout'],
            num_districts=config['num_districts'],
            num_towns=config['num_towns'],
            num_communities=config['num_communities'],
            num_heads=config['num_heads'],
            dis_hidden_size=config['dis_hidden_size'],
            num_level=config['num_level'],
            scl_alpha=config['scl_alpha']
        )
        with open(f'{checkpoint_dir}/checkpoint_beyond_encoder.ckpt', 'rb') as f:
            model.load_state_dict(torch.load(f), strict=False)
        return model


class Discriminator(torch.nn.Module):
    @dataclass
    class Output(ModelOutput):
        loss: Optional[torch.Tensor] = None


    def __init__(
        self,
        hidden_size: int,
        num_districts: int,
        num_towns: int,
        num_communities: int,
        num_level: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_size = hidden_size
        self.num_districts = num_districts
        self.num_towns = num_towns
        self.num_communities = num_communities
        self.num_level = num_level

        self.loss_linear = torch.nn.Sequential(
                                    torch.nn.Linear(self.num_level, self.num_level * 2),
                                    torch.nn.ReLU(),  
                                    torch.nn.Linear(self.num_level * 2, self.num_level)
                                )
        self.district_linear = torch.nn.Linear(self.num_districts, self.hidden_size)
        self.town_linear = torch.nn.Sequential(
                            torch.nn.Linear(self.num_towns, self.num_towns // 2),
                            torch.nn.ReLU(),  
                            torch.nn.Linear(self.num_towns // 2, self.num_towns // 4),
                            torch.nn.ReLU(),  
                            torch.nn.Linear(self.num_towns // 4, self.num_towns // 8),
                            torch.nn.ReLU(),  
                            torch.nn.Linear(self.num_towns // 8, self.num_towns // 16),
                            torch.nn.ReLU(),  
                            torch.nn.Linear(self.num_towns // 16, self.hidden_size)
                        )
        self.community_linear = torch.nn.Sequential(
                            torch.nn.Linear(self.num_communities, self.num_communities // 2),
                            torch.nn.ReLU(),  
                            torch.nn.Linear(self.num_communities // 2, self.num_communities // 4),
                            torch.nn.ReLU(),  
                            torch.nn.Linear(self.num_communities // 4, self.num_communities // 8),
                            torch.nn.ReLU(),  
                            torch.nn.Linear(self.num_communities // 8, self.num_communities // 16),
                            torch.nn.ReLU(),  
                            torch.nn.Linear(self.num_communities // 16, self.hidden_size)
                        )

        self.output_district = torch.nn.Linear(self.hidden_size + self.num_level, 1)
        self.output_town = torch.nn.Linear(self.hidden_size + self.num_level, 1)
        self.output_community = torch.nn.Linear(self.hidden_size + self.num_level, 1)
        
    def forward(self, district_loss: torch.Tensor, town_loss: torch.Tensor, community_loss: torch.Tensor, district_distribution: torch.Tensor, town_distribution: torch.Tensor, community_distribution: torch.Tensor):
        
        loss_emb = self.loss_linear(torch.cat((district_loss, town_loss, community_loss), dim=1))
        district_embedding  = self.district_linear(district_distribution)  # B x D
        town_embedding      = self.town_linear(town_distribution)      # B x T
        community_embedding = self.community_linear(community_distribution)  # B x C

        district_prob = self.output_district(torch.cat((district_embedding, loss_emb), dim=1))
        town_prob = self.output_town(torch.cat((town_embedding, loss_emb), dim=1))
        community_prob = self.output_community(torch.cat((community_embedding, loss_emb), dim=1))
        return torch.sigmoid(district_prob), torch.sigmoid(town_prob), torch.sigmoid(community_prob)