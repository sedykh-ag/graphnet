import torch
from torch import nn
import torch.nn.functional as F
import collections

# dimensions
node_input_dim = 3 # 2 for one-hot encoded node-type, 1 for scalar field
node_output_dim = 1
edge_dim = 1

# collections
EdgeSet = collections.namedtuple('EdgeSet', ['senders', 'receivers', 'features'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_set'])

device = torch.device('cuda')

def unsorted_segment_sum(data, index, n_segments):
    n_features = data.shape[-1]
    return torch.zeros(n_segments, n_features).to(device).scatter_add_(0,
                                                                       index.view(-1, 1).expand(-1, n_features),
                                                                       data)

class GraphNet():
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
                                      
        self.node_encoder = nn.Sequential(nn.Linear(node_input_dim, self.hidden_dim),
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
                                          nn.Linear(self.hidden_dim, node_output_dim))
                                          
        self.node_encoder.to(device)
        self.edge_encoder.to(device)
        self.node_net.to(device)
        self.edge_net.to(device)
        self.node_decoder.to(device)

    def parameters(self):
        """Return all networks' parameters"""
        return list(self.node_encoder.parameters()) + \
               list(self.edge_encoder.parameters()) + \
               list(self.node_net.parameters()) + \
               list(self.edge_net.parameters()) + \
               list(self.node_decoder.parameters())
    
    def encode(self, graph):
        """Encodes both node and edge features into latent features"""
        node_features_latent = self.node_encoder(graph.node_features)
        edge_features_latent = self.edge_encoder(graph.edge_set.features)
        edge_set_latent = graph.edge_set._replace(features=edge_features_latent)
        
        return MultiGraph(node_features_latent, edge_set_latent)
    
    def decode(self, graph):
        """Decodes node features from graph"""
        return self.node_decoder(graph.node_features)
    
    def update_edges(self, node_features, edge_set):
    
        sender_features = node_features[edge_set.senders]
        receiver_features = node_features[edge_set.receivers]
        features = [sender_features, receiver_features, edge_set.features]
        
        return self.edge_net(torch.cat(features, axis=-1)) # new edge features
    
    def update_nodes(self, node_features, edge_set):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0]
        segment_sum = unsorted_segment_sum(edge_set.features,
                                           edge_set.receivers,
                                           num_nodes)
                                                         
        return self.node_net(torch.cat([node_features, segment_sum], axis=-1))
    
    def step(self, graph):
        """Updates latent graph"""
        new_edge_features = self.update_edges(graph.node_features, graph.edge_set)
        new_edge_set = graph.edge_set._replace(features=new_edge_features)
        
        new_node_features = self.update_nodes(graph.node_features, new_edge_set)
        new_node_features += graph.node_features # residual connection
        
        return MultiGraph(new_node_features, new_edge_set)
    
    def forward(self, graph):
        """Encodes and processes a multigraph, and returns node features."""
        graph_latent = self.encode(graph)
        
        for _ in range(self.message_passing_steps):
            graph_latent = self.step(graph_latent)
        
        return self.decode(graph_latent) # returns decoded node features only!