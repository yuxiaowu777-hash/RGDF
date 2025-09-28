import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict


def compute_graph_structures(triples, n_ent):
    # Build an adjacency dictionary.
    adj_dict = {}
    for h, r, t in triples:
        if h not in adj_dict:
            adj_dict[h] = set()
        if t not in adj_dict:
            adj_dict[t] = set()
        adj_dict[h].add(t)
        adj_dict[t].add(h)
    
    degrees = np.zeros(n_ent)
    triangles = np.zeros(n_ent)
    cycles_4 = np.zeros(n_ent)
    
    for node in range(n_ent):
        if node not in adj_dict:
            continue
            
        neighbors = list(adj_dict[node])
        degrees[node] = len(neighbors)
        
        if len(neighbors) >= 2:
            triangle_count = 0
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj_dict.get(neighbors[i], set()):
                        triangle_count += 1

            triangles[node] = triangle_count / 3
        
        # 4-cycle is defined as：node -> neighbor1 -> node_2hop -> neighbor2 -> node
        cycle4_count = 0
        for i, neighbor1 in enumerate(neighbors):
            if neighbor1 in adj_dict:
                for node_2hop in adj_dict[neighbor1]:
                    if node_2hop != node and node_2hop not in neighbors:
                        if node_2hop in adj_dict:
                            for neighbor2 in adj_dict[node_2hop]:
                                if neighbor2 != neighbor1 and neighbor2 in neighbors:
                                    cycle4_count += 1
        
        cycles_4[node] = cycle4_count / 4
    
    return degrees, triangles, cycles_4


class DataLoader:
    def __init__(self, args):
        self.args = args
        self.task_dir = task_dir = args.data_path

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            n_ent = 0
            for line in f:
                entity = line.strip()
                self.entity2id[entity] = n_ent
                n_ent += 1

        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            n_rel = 0
            for line in f:
                relation = line.strip()
                self.relation2id[relation] = n_rel
                n_rel += 1

        self.n_ent = n_ent
        self.n_rel = n_rel

        # prepare triples
        self.filters = defaultdict(lambda: set())
        self.fact_triple = self.read_triples('facts.txt')
        self.train_triple = self.read_triples('train.txt')
        self.valid_triple = self.read_triples('valid.txt')
        self.test_triple = self.read_triples('test.txt')
        self.all_triple = np.concatenate([np.array(self.fact_triple), np.array(self.train_triple)], axis=0)
        self.tmp_all_triple = np.concatenate(
            [np.array(self.fact_triple), np.array(self.train_triple), np.array(self.valid_triple),
             np.array(self.test_triple)], axis=0)

        # add inverse
        self.fact_data = self.double_triple(self.fact_triple)
        self.train_data = np.array(self.double_triple(self.train_triple))
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data = self.double_triple(self.test_triple)

        self.shuffle_train()
        self.load_graph(self.fact_data)
        self.load_test_graph(self.double_triple(self.fact_triple) + self.double_triple(self.train_triple))
        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q, self.test_a = self.load_query(self.test_data)

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test = len(self.test_q)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h, r, t])
                self.filters[(h, r)].add(t)
                self.filters[(t, r + self.n_rel)].add(h)
        return triples

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r + self.n_rel, h])
        return triples + new_triples

    def load_graph(self, triples):
        self.train_degrees, self.train_triangles, self.train_cycles_4 = compute_graph_structures(triples, self.n_ent)
        
        # Add self-loops for entities: (e, 2*n_rel, e)
        # idd: A tensor of shape [self.n_ent, 3].
        entities = torch.arange(self.n_ent, dtype=torch.long, device='cuda')  # [self.n_ent]
        rela_self_loop = torch.full((self.n_ent,), 2 * self.n_rel, dtype=torch.long, device='cuda')  # [self.n_ent]
        idd = torch.stack([
            entities,           # head
            rela_self_loop,     # relation (2*n_rel)
            entities            # tail
        ], dim=1)  # [self.n_ent, 3]

        # self.KG: [self.n_fact, 3]
        triples_tensor = torch.tensor(triples, dtype=torch.long, device='cuda')
        self.KG = torch.cat([triples_tensor, idd], dim=0)
        self.n_fact = self.KG.size(0)

        # M_sub: A sparse tensor of shape [self.n_fact, self.n_ent] representing the projection from head entities to facts (triples).
        indices = torch.stack([
            torch.arange(self.n_fact, dtype=torch.long, device='cuda'),
            self.KG[:, 0]
        ], dim=0)  # [2, self.n_fact]
        values = torch.ones(self.n_fact, dtype=torch.float, device='cuda')
        self.M_sub = torch.sparse_coo_tensor(
            indices, values, (self.n_fact, self.n_ent)
        )  # Convert to a COO-format sparse tensor and merge (coalesce) duplicate entries.

    def load_test_graph(self, triples):
        self.test_degrees, self.test_triangles, self.test_cycles_4 = compute_graph_structures(triples, self.n_ent)
        
        
        entities = torch.arange(self.n_ent, dtype=torch.long, device='cuda')  # [self.n_ent]
        rela_self_loop = torch.full((self.n_ent,), 2 * self.n_rel, dtype=torch.long, device='cuda')  # [self.n_ent]
        idd = torch.stack([
            entities,           # head
            rela_self_loop,     # relation (2*n_rel)
            entities            # tail
        ], dim=1)  # [self.n_ent, 3]

        # self.tKG: [self.tn_fact, 3]
        triples_tensor = torch.tensor(triples, dtype=torch.long, device='cuda')
        self.tKG = torch.cat([triples_tensor, idd], dim=0)
        self.tn_fact = self.tKG.size(0)

        # tM_sub: [self.tn_fact, self.n_ent]
        indices = torch.stack([
            torch.arange(self.tn_fact, dtype=torch.long, device='cuda'),
            self.tKG[:, 0]
        ], dim=0)  # [2, self.tn_fact]
        values = torch.ones(self.tn_fact, dtype=torch.float, device='cuda')
        self.tM_sub = torch.sparse_coo_tensor(
            indices, values, (self.tn_fact, self.n_ent)
        )  # Convert to a COO-format sparse tensor and merge (coalesce) duplicate entries.

    def load_query(self, triples):
        trip_hr = defaultdict(lambda: list())
        for trip in triples:
            h, r, t = trip
            trip_hr[(h, r)].append(t)
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, batchsize, mode='train'):
        # Sample neighbors for the given nodes, returning the edges and the updated node set.
        if mode == 'train':
            KG = self.KG
            M_sub = self.M_sub
            degrees = self.train_degrees
            triangles = self.train_triangles
            cycles_4 = self.train_cycles_4
        else:
            KG = self.tKG
            M_sub = self.tM_sub
            degrees = self.test_degrees
            triangles = self.test_triangles
            cycles_4 = self.test_cycles_4
            
        # nodes: [N_ent_of_all_batch_last, 2] with (batch_idx, node_idx)
        node_indices = torch.stack([nodes[:, 1], nodes[:, 0]], dim=0)  # [2, N_ent_of_all_batch_last]
        node_values = torch.ones(nodes.size(0), dtype=torch.float, device=nodes.device)  # [N_ent_of_all_batch_last]
        node_1hot = torch.sparse_coo_tensor(
            node_indices, node_values, (self.n_ent, nodes.size(0))
        )  # Create a sparse tensor on the GPU.
        # M_sub: [N_fact, self.n_ent] * node_1hot: [self.n_ent, N_ent_of_all_batch_last]
        # -> edge_1hot: [N_fact, N_ent_of_all_batch_last]
        edge_1hot = torch.sparse.mm(M_sub, node_1hot)
        # [2, N_edge_of_all_batch] with (fact_idx, batch_idx)
        edges = edge_1hot.coalesce().indices()
        # sampled_edges: [N_edge_of_all_batch, 4] with (batch_idx, head, rela, tail)
        sampled_edges = torch.cat([
            edges[1].unsqueeze(1),  # batch_idx
            KG[edges[0]]  # head, rela, tail
        ], dim=1).to(torch.long)  # [N_edge_of_all_batch, 4]，已在 GPU 上
        # Get the unique head and tail nodes, and compute the relative indices.
        head_nodes, head_index = torch.unique(
            sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True
        )  # head_nodes: [N_ent_of_all_batch_last, 2], head_index: [N_edge_of_all_batch]
        tail_nodes, tail_index = torch.unique(
            sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True
        )  # tail_nodes: [N_ent_of_all_batch_this, 2], tail_index: [N_edge_of_all_batch]

        # sampled_edges: [N_edge_of_all_batch, 6] with (batch_idx, head, rela, tail, head_index, tail_index)
        sampled_edges = torch.cat([
            sampled_edges,
            head_index.unsqueeze(1),
            tail_index.unsqueeze(1)
        ], dim=1)

        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        old_nodes_new_idx = tail_index[mask].sort()[0]  # [N_ent_of_all_batch_last]
        
        node_degrees = torch.tensor(degrees[tail_nodes[:, 1].cpu().numpy()], dtype=torch.float, device=nodes.device)
        node_triangles = torch.tensor(triangles[tail_nodes[:, 1].cpu().numpy()], dtype=torch.float, device=nodes.device)
        node_cycles_4 = torch.tensor(cycles_4[tail_nodes[:, 1].cpu().numpy()], dtype=torch.float, device=nodes.device)
        
        return tail_nodes, sampled_edges, old_nodes_new_idx, node_degrees, node_triangles, node_cycles_4

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data == 'train':
            return np.array(self.train_data)[batch_idx]
        if data == 'valid':
            query, answer = np.array(self.valid_q), self.valid_a
        if data == 'test':
            query, answer = np.array(self.test_q), self.test_a

        subs = []
        rels = []
        objs = []
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), self.n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self):
        all_triple = self.all_triple
        n_all = len(all_triple)
        rand_idx = np.random.permutation(n_all)
        all_triple = all_triple[rand_idx]

        bar = int(n_all * self.args.fact_ratio)
        self.fact_data = np.array(self.double_triple(all_triple[:bar].tolist()))
        self.train_data = np.array(self.double_triple(all_triple[bar:].tolist()))

        if self.args.remove_1hop_edges:
            print('==> removing 1-hop links...')
            tmp_index = np.ones((self.n_ent, self.n_ent))
            tmp_index[self.train_data[:, 0], self.train_data[:, 2]] = 0
            save_facts = tmp_index[self.fact_data[:, 0], self.fact_data[:, 2]].astype(bool)
            self.fact_data = self.fact_data[save_facts]
            print('==> done')

        self.n_train = len(self.train_data)
        self.load_graph(self.fact_data)