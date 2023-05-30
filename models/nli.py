import torch

import models


class ClassifyWrapper():

    def __init__(self, model_name='microsoft/deberta-large-mnli', device='cuda:3') -> None:
        self.model_name = model_name
        self.model, self.tokenizer = models.load_model_and_tokenizer(model_name, device)

        pass

    @torch.no_grad()
    def _batch_pred(self, sen_1: list, sen_2: list, max_batch_size=128):
        inputs = [_[0] + ' [SEP] ' + _[1] for _ in zip(sen_1, sen_2)]
        inputs = self.tokenizer(inputs, padding=True, truncation=True)
        input_ids = torch.tensor(inputs['input_ids']).to(self.model.device)
        attention_mask = torch.tensor(inputs['attention_mask']).to(self.model.device)
        logits = []
        for st in range(0, len(input_ids), max_batch_size):
            ed = min(st + max_batch_size, len(input_ids))
            logits.append(self.model(input_ids=input_ids[st:ed],
                                attention_mask=attention_mask[st:ed])['logits'])
        return torch.cat(logits, dim=0)

    @torch.no_grad()
    def create_sim_mat_batched(self, question, answers):
        unique_ans = sorted(list(set(answers)))
        semantic_set_ids = {ans: i for i, ans in enumerate(unique_ans)}
        _rev_mapping = semantic_set_ids.copy()
        sim_mat_batch = torch.zeros((len(unique_ans), len(unique_ans),3))
        anss_1, anss_2, indices = [], [], []
        for i, ans_i in enumerate(unique_ans):
            for j, ans_j in enumerate(unique_ans):
                if i == j: continue
                anss_1.append(f"{question} {ans_i}")
                anss_2.append(f"{question} {ans_j}")
                indices.append((i,j))
        if len(indices) > 0:
            sim_mat_batch_flat = self._batch_pred(anss_1, anss_2)
            for _, (i,j) in enumerate(indices):
                sim_mat_batch[i,j] = sim_mat_batch_flat[_]
        return dict(
            mapping = [_rev_mapping[_] for _ in answers],
            sim_mat = sim_mat_batch
        )

    @torch.no_grad()
    def _pred(self, sen_1: str, sen_2: str):
        input = sen_1 + ' [SEP] ' + sen_2
        input_ids = self.tokenizer.encode(input, return_tensors='pt').to(self.model.device)

        logits = self.model(input_ids)['logits']
        # logits: [Contradiction, neutral, entailment]
        return logits

    @torch.no_grad()
    def pred_qa(self, question:str, ans_1:str, ans_2:str):
        return self._pred(f"{question} {ans_1}", f'{question} {ans_2}')

    @torch.no_grad()
    def _compare(self, question:str, ans_1:str, ans_2:str):
        pred_1 = self._pred(f"{question} {ans_1}", f'{question} {ans_2}')
        pred_2 = self._pred(f"{question} {ans_2}", f'{question} {ans_1}')
        preds = torch.concat([pred_1, pred_2], 0)

        deberta_prediction = 0 if preds.argmax(1).min() == 0 else 1
        return {'deberta_prediction': deberta_prediction,
                'prob': torch.softmax(preds,1).mean(0).cpu(),
                'pred': preds.cpu()
                }
