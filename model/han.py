
import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.nn import HANConv
import config.includes as inc
from .dataset import BNNHDataSet

import torch.nn.functional as F
from torch_geometric.nn import Linear, HANConv
from sklearn.metrics import classification_report

from model.dataset import BNNHDataSet

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class BNNHAN(nn.Module):
    '''HAN Model for BNN EEG Graph HeteroData'''
    def __init__(self, dim_in, dim_out, dim_h=128, heads=8, metadata=[]):
        super().__init__()
        self.han = HANConv(dim_in, dim_h, heads=heads, dropout=0.6, metadata=metadata)
        self.linear = Linear(dim_h, dim_out)

    def forward(self, x_dict, edge_index_dict):
        out = self.han(x_dict, edge_index_dict)
        out = self.linear(out['SUBJECT'])
        return out

class BNNHANPOC(object):
    '''HAN Proof of Concept'''
    def __init__(self):
        self.path = inc.BNNHDSDIR

        metapaths = [[('SUBJECT', 'READ_LOC'), ('READ_LOC', 'WAVE_ABP')],
                    [('WAVE_ABP', 'READ_LOC'), ('READ_LOC', 'SUBJECT')]]

        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True, # type: ignore
                                drop_unconnected_node_types=True)

        dataset = BNNHDataSet(root=self.path, transform=transform)

        self.data= dataset[0]

    @torch.no_grad()
    def test(self,mask, model, data):
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)
        acc = (pred[mask] == data['SUBJECT'].y[mask]).sum() / mask.sum()
        return float(acc)

    def fit(self, model, optimizer, data, nepocs=200, patience=100):
        best_val_acc = 0
        start_patience = patience

        for epoch in range(nepocs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict)
            mask = data['SUBJECT'].train_mask
            loss = F.cross_entropy(out[mask], data['SUBJECT'].y[mask])
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                train_acc = self.test(data['SUBJECT'].train_mask, model, data)
                val_acc = self.test(data['SUBJECT'].val_mask, model, data)
                test_acc = self.test(data['SUBJECT'].test_mask, model, data)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, 'f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

                if best_val_acc <= val_acc:
                    patience = start_patience
                    best_val_acc = val_acc
                else:
                    patience -= 1

            if patience <= 0:
                print('Stopping training as validation accuracy did not improve '
                    f'for {start_patience} epochs')
                break


    @torch.no_grad()
    def performance_report(self,mask, model, data):
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)
        y_true = data['SUBJECT'].y[mask].cpu()
        y_pred = pred[mask].cpu()
        labels = ['Chronic-Pain', 'Control']
        print(classification_report(y_true, y_pred, target_names=labels, digits=4))
        return


    def run(self):
        model = BNNHAN(dim_in=-1, dim_out=2,metadata=self.data.metadata())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data, model = self.data.to(str(device)), model.to(str(device))

        #print("Loading BNNHDataSet", "-"*40, "\n", self.data)
        print("-"*40, "\n")
        self.fit(model, optimizer, data)
        test_acc = self.test(data['SUBJECT'].test_mask, model, data)
        print(f'Test accuracy: {test_acc*100:.2f}%')
        print("-"*40, "\n")
        self.performance_report(data['SUBJECT'].test_mask, model, data)
        print("-"*40, "\n")
        return


