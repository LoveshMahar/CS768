import argparse
import os
import pickle
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
from transformers import BertTokenizer, BertModel
from collections import defaultdict
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
class PaperDataset:
    def __init__(self, dataset_path="./dataset_papers", graph_path="citation_graph.gpickle"):
        self.dataset_path = dataset_path
        self.graph_path = graph_path
        
        # Load or build the citation graph
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                self.G = pickle.load(f)
        
        # Initialize BERT for text embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Generate node features and edge index
        self.node_features = self._generate_node_features()
        self.edge_index = self._get_edge_index()
        self.data = self.get_pyg_data()
        
    def _generate_node_features(self):
        """Generate BERT embeddings for paper titles and abstracts"""
        features = {}
        for node, data in self.G.nodes(data=True):
            # print(node)
            title = data['title']
            folder = data['folder']
            
            # Get abstract if available
            abstract_path = os.path.join(self.dataset_path, folder, "abstract.txt")
            text = title
            if os.path.exists(abstract_path):
                with open(abstract_path, 'r', encoding='utf-8', errors='ignore') as f:
                    abstract = f.read().strip()
                    text += " " + abstract
            
            # Get BERT embeddings
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.bert(**inputs)
            features[node] = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Convert to numpy array
        num_nodes = len(self.G.nodes())
        feature_size = next(iter(features.values())).shape[0]
        feature_matrix = np.zeros((num_nodes, feature_size))
        
        for node in self.G.nodes():
            feature_matrix[node] = features[node]
            
        return torch.FloatTensor(feature_matrix)
    
    def _get_edge_index(self):
        """Convert networkx edges to PyG edge index format"""
        edge_index = torch.tensor(list(self.G.edges())).t().contiguous()
        return edge_index
    
    def get_pyg_data(self):
        """Convert to PyTorch Geometric Data object"""
        return Data(x=self.node_features, edge_index=self.edge_index)
    
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class LinkPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels * 2, in_channels)
        self.lin2 = nn.Linear(in_channels, 1)
        
    def forward(self, z_src, z_dst):
        h = torch.cat([z_src, z_dst], dim=1)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.lin2(h)
        return torch.sigmoid(h)

def train(model, predictor, data, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        predictor.train()
        optimizer.zero_grad()
        
        # Get node embeddings
        z = model(data.x, data.edge_index)
        
        # Sample positive and negative edges
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )
        
        # Calculate loss
        pos_out = predictor(z[pos_edge_index[0]], z[pos_edge_index[1]])
        neg_out = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])
        
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
def predict_citations(dataset, title, abstract, top_k=10):
    """Predict top-k citations for a new paper"""
    # Get embedding for the new paper
    text = title + " " + abstract
    inputs = dataset.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = dataset.bert(**inputs)
    new_paper_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # Shape: [768]
    
    # Get embeddings for all papers in the dataset
    with torch.no_grad():
        z = dataset.model(dataset.data.x, dataset.data.edge_index)  # Shape: [num_nodes, out_channels]
    
    # Prepare inputs for predictor
    new_paper_repeated = new_paper_embedding.repeat(z.size(0), 1)  # Shape: [num_nodes, 768]
    
    # Ensure dimensions match predictor expectations
    if new_paper_repeated.size(1) != z.size(1):
        # Project BERT embeddings to match GraphSAGE output dimension
        projection = nn.Linear(new_paper_repeated.size(1), z.size(1)).to(z.device)
        new_paper_repeated = projection(new_paper_repeated)
    
    # Calculate scores
    scores = dataset.predictor(
        new_paper_repeated,
        z
    ).squeeze()
    
    # Sort by score and return top-k
    _, top_k_indices = torch.topk(scores, k=top_k)
    top_papers = [dataset.G.nodes[paper_id.item()]['folder'] for paper_id in top_k_indices]
    
    return top_papers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-paper-title", type=str, required=True)
    parser.add_argument("--test-paper-abstract", type=str, required=True)
    args = parser.parse_args()

    ################################################
    #               YOUR CODE START                #
    ################################################

    # Load dataset
    dataset = PaperDataset()
    
    # Initialize models
    model = GraphSAGE(
        in_channels=dataset.data.num_features,
        hidden_channels=256,
        out_channels=128
    )
    predictor = LinkPredictor(in_channels=128)

    # Train or load models
    # if not (os.path.exists("graphsage_model.pth") and os.path.exists("link_predictor.pth")):
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=0.01,
        weight_decay=5e-4
    )
    train(model, predictor, dataset.data, optimizer, epochs=100)
    torch.save(model.state_dict(), "graphsage_model.pth")
    torch.save(predictor.state_dict(), "link_predictor.pth")
    
    # Attach models to dataset for prediction
    dataset.model = model
    dataset.predictor = predictor
    
    # Get predictions for new paper
    result = predict_citations(
        dataset=dataset,
        title=args.test_paper_title,
        abstract=args.test_paper_abstract
    )

    ################################################
    #               YOUR CODE END                  #
    ################################################
    
    ################################################
    #               DO NOT CHANGE                  #
    ################################################
    print('\n'.join(result))

if __name__ == "__main__":
    main()