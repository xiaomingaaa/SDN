import pandas as pd

def process_entity_relation(dataset='WN18RR'):
    entity_ids = dict()
    id2entity = []
    relation_ids = dict()
    id2relation = []
    for t in ['train', 'valid', 'test']:
        with open(f'data/{dataset}/{t}.txt', 'r') as f:
            for line in f:
                e1, r, e2 = line.strip().split('\t')
                if e1 not in entity_ids:
                    entity_ids[e1] = len(entity_ids)
                    id2entity.append([entity_ids[e1], e1])
                if e2 not in entity_ids:
                    entity_ids[e2] = len(entity_ids)
                    id2entity.append([entity_ids[e2], e2])
                if r not in relation_ids:
                    relation_ids[r] = len(relation_ids)
                    # id2relation[relation_ids[r]] = r
                    id2relation.append([relation_ids[r], r])

    pd.DataFrame(id2entity).to_csv(f'data/{dataset}/entities.dict', '\t', index=False, header=False)
    pd.DataFrame(id2relation).to_csv(f'data/{dataset}/relations.dict', '\t', index=False, header=False)

process_entity_relation(dataset='WN18RR_v2')