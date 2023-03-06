import torch

def topology_to_edges(topology):
    edges = torch.cat((topology[:, 0:2],
                       topology[:, 1:3],
                       torch.stack((topology[:, 2], topology[:, 0]), dim=1))).to(torch.int32)
    
    senders = torch.min(edges, dim=1).values
    receivers = torch.max(edges, dim=1).values

    packed_edges = torch.stack([senders, receivers], dim=1).view(torch.int64)
    unique_edges = torch.unique(packed_edges).view(torch.int32).view(-1, 2)
    senders, receivers = torch.unbind(unique_edges, dim=1)

    return (torch.cat([senders, receivers], dim=0),
            torch.cat([receivers, senders], dim=0))
    