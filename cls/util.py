import torch
import torch.nn as nn
import torch.nn.functional as F


def Alpha(tensor, delta):
    Alpha = []
    for i in range(tensor.size()[0]):
        count = 0
        abssum = 0
        absvalue = tensor[i].view(1,-1).abs()
        if isinstance(delta, int):
            truth_value = absvalue > delta
        else:
            truth_value = absvalue > delta[i]
            
        count = truth_value.sum()
        #print (count, truth_value.numel())
        abssum = torch.matmul(absvalue, truth_value.to(torch.float32).view(-1,1))
        Alpha.append(abssum/count)
    
    alpha = torch.cat(Alpha, dim=0)
    return alpha

def Delta(tensor):
    n = tensor[0].nelement()
    if(len(tensor.size()) == 4):     #convolution layer
        delta = 0.75 * torch.sum(tensor.abs(), dim=(1,2,3))/n
    elif(len(tensor.size()) == 2):   #fc layer
        delta = 0.75 * torch.sum(tensor.abs(), dim=(1,))/n
        print(1/0)
    return delta

def Binarize(tensor):
    output = torch.zeros(tensor.size(), device=tensor.device)
    delta = 0
    alpha = Alpha(tensor, delta)
    for i in range(tensor.size()[0]):
        pos_one = (tensor[i] > delta).to(torch.float32)
        neg_one = pos_one-1
        out = torch.add(pos_one, neg_one)
        output[i] = torch.add(output[i], torch.mul(out, alpha[i]))
        
    return output

def Ternarize(tensor):
    # output = torch.zeros(tensor.size(), device=tensor.device)
    # delta = Delta(tensor)
    # alpha = Alpha(tensor,delta)
    # for i in range(tensor.size()[0]):
    #     pos_one = (tensor[i] > delta[i]).to(torch.float32)
    #     neg_one = -1 * (tensor[i] < -delta[i]).to(torch.float32)
    #     out = torch.add(pos_one, neg_one)
    #     output[i] = torch.add(output[i],torch.mul(out, alpha[i]))

    delta2 = Delta(tensor)
    absw = torch.abs(tensor)
    Iw = absw > delta2[:, None, None, None]
    alpha2 = (1/torch.sum(Iw, dim=(1, 2, 3)))*(torch.sum(absw * Iw, dim=(1, 2, 3)))
    w_ = 1*(tensor > delta2[:, None, None, None]) + (-1)*(tensor < -delta2[:, None, None, None])
    output2 = alpha2[:, None, None, None] * w_
    # diff = torch.sum(torch.abs(output - output2))
    return output2
            
            
class Conv2DFunctionQUAN(torch.autograd.Function):
    def __init__(self):
        super(Conv2DFunctionQUAN, self).__init__()
        self.com_num = 0
        self.weight_fp32 = None
        
    @staticmethod
    def forward(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, quan_mode='TERANRY'):
        self.weight_fp32 = weight.data.clone().detach() #save a copy of fp32 precision weight
        if quan_mode == 'TERANRY':
            weight.data[:,:,:,:] = Ternarize(weight.data.clone().detach())[:,:,:,:] #do ternarization
        elif quan_mode == 'BINARY':
            weight.data[:,:,:,:] = Binarize(weight.data.clone().detach())[:,:,:,:] #do ternarization
        else:
            pass 
        
        self.save_for_backward(input, weight, bias)
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        output = torch.nn.functional.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
        return output

    @staticmethod
    def backward(self, grad_output):
    
        input, weight, bias = self.saved_tensors
        stride, padding, dilation, groups = self.stride, self.padding, self.dilation, self.groups
        grad_input = grad_weight = grad_bias = grad_stride = grad_padding = grad_dilation = grad_groups = grad_quan_mode = None

        if self.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if self.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if self.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3), dtype=None, keepdim=False).squeeze(0) 
            
        self.saved_tensors[1].data[:,:,:,:] = self.weight_fp32[:,:,:,:] # recover the fp32 precision weight for parameter update
        
        return grad_input, grad_weight, grad_bias, grad_stride, grad_padding, grad_dilation, grad_groups, None

    
class TernaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(TernaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)
        # _in_channels = 3
        # _padding = 0
        # _dilation = 1
        # _groups = 1
        # _bias = True
        self.lossn_track = []
        tensor = self.weight.clone().detach()
        if tensor.shape[2] != 1:
            self.alpha_delta_network = nn.Sequential(
                nn.Conv2d(in_channels, 4*in_channels, 2, stride=1, padding=0, dilation=dilation, groups=groups, bias=bias),
                nn.BatchNorm2d(num_features=4*in_channels),
                nn.LeakyReLU(),
                nn.Conv2d(4*in_channels, 4*in_channels, 2, stride=1, padding=0, dilation=dilation, groups=groups, bias=bias),
                nn.BatchNorm2d(num_features=4*in_channels),
                nn.LeakyReLU(),
                nn.Flatten(start_dim=1, end_dim=3),
                nn.Linear(4*in_channels, 2, bias),
                nn.BatchNorm1d(num_features=2),
                nn.LeakyReLU()
                )
        else:
            self.alpha_delta_network = nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=bias),
                nn.LeakyReLU(),
                nn.Linear(in_channels, in_channels, bias=bias),
                nn.LeakyReLU(),
                nn.Linear(in_channels, 2, bias=bias),
                nn.LeakyReLU()
            )
        self.alpha_delta_network.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.alpha_delta_network.parameters(), lr=0.01, weight_decay=0.0001)
        # input = torch.ones((16, in_channels, 3, 3))
        # if tensor.shape[2] != 1:
        #     out2 = self.alpha_delta_network(tensor)
        # else:
        #     out2 = self.alpha_delta_network(torch.flatten(tensor, start_dim=1, end_dim=3))
        # print("")

    def fw_(self, x, delta):
        epsilon = 0.1
        v1 = x * (epsilon / delta)
        v2 = (x + delta) * (epsilon / delta) + (-1 + epsilon)
        v3 = (x - delta) * (epsilon / delta) + (1 - epsilon)

        return torch.logical_and(x >= -delta, x <= delta) * v1 + (x < -delta) * v2 + (x > delta) * v3

        # if x >= -delta and x <= delta:
        #     return x * (epsilon / delta)
        # elif x < -delta:
        #     return (x + delta) * (epsilon / (delta)) + (-1 + epsilon)
        # elif x > delta:
        #     return (x - delta) * (epsilon / (delta)) + (1 - epsilon)

    def forward(self, x):
        tensor = self.weight.clone().detach()

        output = torch.zeros(tensor.size(), device=tensor.device)
        delta = Delta(tensor)
        alpha = Alpha(tensor,delta)
        for i in range(tensor.size()[0]):
            pos_one = (tensor[i] > delta[i]).to(torch.float32)
            neg_one = -1 * (tensor[i] < -delta[i]).to(torch.float32)
            out = torch.add(pos_one, neg_one)
            output[i] = torch.add(output[i],torch.mul(out, alpha[i]))

        lossf = torch.sqrt(torch.sum((tensor - output)**2))

        up = torch.amax(tensor, dim=(1, 2, 3))
        down = torch.amin(tensor, dim=(1, 2, 3))
        rangew = up - down
        self.alpha_delta_network.requires_grad_(True)
        for i in range(200000 if tensor.shape[2] != 1 else 20):
            self.optimizer.zero_grad()
            if tensor.shape[2] != 1:
                alpha_delta = self.alpha_delta_network(tensor)
            else:
                alpha_delta = self.alpha_delta_network(torch.flatten(tensor, start_dim=1, end_dim=3))
            alpha2, delta2 = alpha_delta[:, 0], torch.abs(alpha_delta[:, 1])
            # w_ = 1 * (tensor > delta2[:, None, None, None]) + (-1) * (tensor < -delta2[:, None, None, None])
            w_ = self.fw_(tensor, delta2[:, None, None, None])
            output2 = alpha2[:, None, None, None] * w_
            loss = torch.sqrt(torch.sum((tensor - output2)**2)) + torch.sum((torch.abs(output2) > 1)*torch.abs(output2)) + 10*torch.sum((torch.abs(alpha2) > 0.1)*torch.abs(alpha2))
            print(f"i: {i}, loss: {loss.item()}")
            loss.backward()
            self.optimizer.step()
        self.alpha_delta_network.requires_grad_(False)

        # range_cover_by_alpha = alpha2[0] * (0.1 / 2 + (up[0] - delta2[0]) * (0.1 / delta2[0])) + alpha2[0] * ( 0.1 / 2 + (delta2[0] - down[0]) * (0.1 / delta2[0]))

        output3 = torch.zeros(tensor.size(), device=tensor.device)
        for i in range(tensor.size()[0]):
            pos_one = (tensor[i] > delta2[i]).to(torch.float32)
            neg_one = -1 * (tensor[i] < -delta2[i]).to(torch.float32)
            out = torch.add(pos_one, neg_one)
            output3[i] = torch.add(output3[i],torch.mul(out, alpha2[i]))

        loss3 = torch.sqrt(torch.sum((tensor - output3)**2))

        self.lossn_track.append(loss.item())
        print(f"lossf: {lossf.item()}, lossn: {self.lossn_track}")

        return Conv2DFunctionQUAN.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, 'TERANRY')
    
class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return Conv2DFunctionQUAN.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, 'BINARY')


def save_model(model, acc, name_prefix='mnist'):
    print('Saving model ...')
    state = {
        'acc':acc,
        'state_dict':model.state_dict() 
    }
    torch.save(state, name_prefix+'-latest.pth')
    print('*** DONE! ***')
