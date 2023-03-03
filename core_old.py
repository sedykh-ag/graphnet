import torch
from torch import nn
import torch.nn.functional as F
import collections
import copy

# dimensions
node_dim = 1
edge_dim = 1

# collections
EdgeSet = collections.namedtuple('EdgeSet', ['senders', 'receivers', 'features'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_set'])

device = torch.device('cpu')

def unsorted_segment_sum(data, index, n_segments):
    n_features = data.shape[-1]
    return torch.zeros(n_segments, n_features).scatter_add_(0,
                                                           index.view(-1, 1).expand(-1, n_features),
                                                           data)

class EncoderDecoder():
    def __init__(self):
        self.latent_dim = 128
        self.hidden_dim = 128
        
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, self.latent_dim))
                                          
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, self.latent_dim))
                                          
        self.node_decoder = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, node_dim))
        
        self.edge_decoder = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, edge_dim))
        
        self.node_encoder.to(device)
        self.edge_encoder.to(device)
        self.node_decoder.to(device)
        self.edge_decoder.to(device)
        
    def parameters(self):
        return list(self.node_encoder.parameters()) + \
               list(self.edge_encoder.parameters()) + \
               list(self.node_decoder.parameters()) + \
               list(self.edge_decoder.parameters())
    
    def encode(self, graph):
        # edge encoding
        edge_features_latent = self.edge_encoder(graph.edge_set.features)
        edge_set_latent = EdgeSet(graph.edge_set.N, graph.edge_set.adjacency_list, edge_features_latent)
        
        # node encoding
        node_features_latent = self.node_encoder(graph.node_set.features)
        node_set_latent = NodeSet(graph.node_set.N, graph.node_set.node_types, node_features_latent)
        
        return MultiGraph(node_set_latent, edge_set_latent)
        
    def decode(self, graph):
        # edge decoding
        edge_features = self.edge_decoder(graph.edge_set.features)
        edge_set = EdgeSet(graph.edge_set.N, graph.edge_set.adjacency_list, edge_features)
        
        # node decoding
        node_features = self.node_decoder(graph.node_set.features)
        node_set = NodeSet(graph.node_set.N, graph.node_set.node_types, node_features)
        
        return MultiGraph(node_set, edge_set)
    
    def forward(self, graph):
        graph = self.encode(graph)
        graph = self.decode(graph)
        
        return graph

class StaticProcessor():
    def __init__(self, message_passing_steps):
        self.message_passing_steps = message_passing_steps
        self.latent_dim = 128
        self.hidden_dim = 128
    
        self.node_net = nn.Sequential(nn.Linear(self.latent_dim * 2, self.hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_dim, self.hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_dim, self.latent_dim))
        
        self.edge_net = nn.Sequential(nn.Linear(self.latent_dim * 3, self.hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_dim, self.hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_dim, self.latent_dim))
                                      
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, self.latent_dim))
                                          
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, self.latent_dim))
                                          
        self.node_decoder = nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, self.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_dim, node_dim))
                                          
        self.node_encoder.to(device)
        self.edge_encoder.to(device)
        self.node_net.to(device)
        self.edge_net.to(device)
        self.node_decoder.to(device)

    def parameters(self):
        return list(self.node_encoder.parameters()) + \
               list(self.edge_encoder.parameters()) + \
               list(self.node_net.parameters()) + \
               list(self.edge_net.parameters()) + \
               list(self.node_decoder.parameters()) + \
    
    def encode(self, graph):
        """Encodes both node and edge features into latent features"""
        node_features_latent = self.node_encoder(graph.node_features)
        edge_features_latent = self.edge_encoder(graph.edge_set.features)
        edge_set_latent = graph.edge_set._replace(features=edge_features_latent)
        
        return MultiGraph(node_features_latent, edge_set_latent)
    
    def decode(self, graph):
        """Decodes node features from graph"""
        return self.node_decoder(graph.node_features)
    
    
    """
    def encode(self, graph):
        # edge encoding
        edge_features_latent = self.edge_encoder(graph.edge_set.features)
        edge_set_latent = EdgeSet(graph.edge_set.N, graph.edge_set.adjacency_list, edge_features_latent)
        
        # node encoding
        node_features_latent = self.node_encoder(graph.node_set.features)
        node_set_latent = NodeSet(graph.node_set.N, graph.node_set.node_types, node_features_latent)
        
        return MultiGraph(node_set_latent, edge_set_latent)
        
    def decode(self, graph):
        # edge decoding
        edge_features = self.edge_decoder(graph.edge_set.features)
        edge_set = EdgeSet(graph.edge_set.N, graph.edge_set.adjacency_list, edge_features)
        
        # node decoding
        node_features = self.node_decoder(graph.node_set.features)
        node_set = NodeSet(graph.node_set.N, graph.node_set.node_types, node_features)
        
        return MultiGraph(node_set, edge_set)
    """
    
    def update_edges(self, node_features, edge_set):
    
        sender_features = torch.gather(node_features, edge_set.senders)
        receiver_features = torch.gather(node_features, edge_set.receivers)
        features = [sender_features, receiver_features, edge_set.features]
        
        return self.edge_net(torch.cat(features, axis=-1)) # new edge features
    
    def update_nodes(self, node_features, edge_set):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0]
        segment_sum = unsorted_segment_sum(edge_set.features,
                                           edge_set.receivers,
                                           num_nodes)
                                                         
        return self.node_net(torch.cat([node_features, segment_sum], axis=-1))
    """
    def update_edges(self, graph):
        edge_set = graph.edge_set
        node_set = graph.node_set
        
        features_pred = torch.zeros_like(edge_set.features).to(device) # shape: [N_edges, latent_dim]
        
        for i in range(edge_set.N):
            node_1, node_2 = edge_set.adjacency_list[i]
            
            X = torch.cat([node_set.features[node_1], node_set.features[node_2], edge_set.features[i]])
            features_pred[i] = self.edge_net(X)
            
        features_pred += edge_set.features # residual connection
        
        edge_set_pred = EdgeSet(edge_set.N, edge_set.adjacency_list, features_pred)
        return MultiGraph(node_set, edge_set_pred)
    
    
    def update_nodes(self, graph):
        edge_set = graph.edge_set
        node_set = graph.node_set
        
        features_pred = torch.zeros_like(node_set.features).to(device)
        
        for i in range(node_set.N):
            
            # if (node_set.node_types[i] == 0):
            if True:
            
                agg_edges = torch.zeros(self.latent_dim).to(device) # shape: [latent_dim]
                
                cnt = 0
                for j in range(edge_set.N):
                    if i in edge_set.adjacency_list[j]:
                        agg_edges += edge_set.features[j]
                        cnt += 1
                
                # print(f"agg_edges = {agg_edges}")
                
                X = torch.cat([node_set.features[i], agg_edges])
                
                features_pred[i] = self.node_net(X) + edge_set.features[i] # residual connection
                # print(f"aggregated {cnt} edges for node number {i},\t agg_edges = {agg_edges.item()}")
                
            else:
                features_pred[i] = node_set.features[i]
        
        # print(features_pred.view(9, 9))
       
        node_set_pred = NodeSet(node_set.N, node_set.node_types, features_pred)
        return MultiGraph(node_set_pred, edge_set)
        """
        
    
    def step(self, graph):
        """Updates latent graph"""
        new_edge_features = self.update_edges(graph.node_features, graph.edge_set)
        new_edge_set = graph.edge_set._replace(features=new_edge_features)
        
        new_node_features = self.update_nodes(graph.node_features, new_edge_set)
        
        return MultiGraph(new_node_features, new_edge_set)
    
    def forward(self, graph):
        
        graph_latent = self.encode(graph)
        
        for _ in range(self.message_passing_steps):
            graph_latent = self.step(graph_latent)
        
        return self.decode(graph_latent) # returns decoded node features only !