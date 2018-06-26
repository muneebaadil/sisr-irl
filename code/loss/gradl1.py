import torch
import torch.nn as nn 

class GradL1(nn.Module): 
    def __init__(self, n_channel, cuda): 
        super(GradL1, self).__init__()
        self.hkernel = torch.zeros(size=(n_channel, n_channel, 3, 3),dtype=torch.float32)
        l = range(n_channel)
        self.hkernel[l,l] = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32)
        
        self.vkernel = torch.zeros(size=(n_channel, n_channel, 3, 3), dtype=torch.float32)
        l = range(n_channel)
        self.vkernel[l,l] = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32)

        if cuda: 
            self.vkernel = self.vkernel.cuda()
            self.hkernel = self.hkernel.cuda()
    
    def forward(self, input, target):
        v_edges_input = nn.functional.conv2d(input, self.vkernel)
        h_edges_input = nn.functional.conv2d(input, self.hkernel)
        
        v_edges_target = nn.functional.conv2d(target, self.vkernel)
        h_edges_target = nn.functional.conv2d(target, self.hkernel)
        
        edges_mag_input = torch.add(torch.pow(v_edges_input,2),
                                    torch.pow(h_edges_input,2))        
        edges_mag_target = torch.add(torch.pow(v_edges_target,2),
                                    torch.pow(h_edges_target,2))

        v_edges_input_norm = v_edges_input / edges_mag_input
        h_edges_input_norm = h_edges_input / edges_mag_input

        v_edges_target_norm = v_edges_target / edges_mag_target
        h_edges_target_norm = h_edges_target / edges_mag_target
        
        v_edges_loss = torch.mean(torch.abs(v_edges_target_norm - v_edges_input_norm))
        h_edges_loss = torch.mean(torch.abs(h_edges_target_norm - h_edges_input_norm))

        out = v_edges_loss + h_edges_loss
        
        return out 
    