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

class WeightNetwork(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.a1 = nn.Flatten(start_dim=0, end_dim=1)
        self.a1_1 = nn.Flatten(start_dim=1, end_dim=2)
        self.a2 = nn.Linear(3*3, out_features=4*3*3, bias=True)
        self.a3 = nn.BatchNorm1d(num_features=4*3*3)
        self.a4 = nn.LeakyReLU()
        self.a41 = nn.Linear(4*3*3, out_features=4*3*3, bias=True)
        self.a42 = nn.BatchNorm1d(num_features=4*3*3)
        self.a43 = nn.LeakyReLU()
        self.a5 = nn.Linear(4*3*3, out_features=3*3, bias=True)
        self.a6 = nn.BatchNorm1d(num_features=3*3)
        self.a7 = nn.LeakyReLU()

        # self.a1 = nn.Conv2d(in_channels, in_channels, 2, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.a2 = nn.BatchNorm2d(num_features=in_channels)
        # self.a3 = nn.LeakyReLU()
        # self.a4 = nn.Conv2d(in_channels, in_channels, 2, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.a5 = nn.BatchNorm2d(num_features=in_channels)
        # self.a6 = nn.LeakyReLU()
        # self.a7 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, padding=0, dilation=1, groups=1, bias=True)
        # self.a8 = nn.BatchNorm2d(num_features=in_channels)
        self.a9 = nn.Tanh()

        # self.a10 = nn.Flatten()
        self.a11 = nn.Linear(3*3, 1, bias=True)
        self.a13 = nn.Sigmoid()

    def forward(self, x):
        y = self.a1(x)
        y = self.a1_1(y)
        y = self.a2(y)
        y = self.a3(y)
        y = self.a4(y)
        y = self.a41(y)
        y = self.a42(y)
        y = self.a43(y)
        y = self.a5(y)
        y = self.a6(y)
        y = self.a7(y)
        # y = self.a8(y)
        o1 = self.a9(y)
        # y = self.a10(o1)
        y = self.a11(y)
        o2 = self.a13(y)
        return o1.reshape(x.shape[0], self.in_channels, 3, 3), torch.mean(o2.reshape(x.shape[0], -1), dim=1)

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
        self.lossf = None
        tensor = self.weight.clone().detach().cuda()
        if tensor.shape[2] != 1:
            self.alpha_delta_network = WeightNetwork(in_channels).cuda()
        else:
            self.alpha_delta_network = nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=bias),
                nn.LeakyReLU(),
                nn.Linear(in_channels, in_channels, bias=bias),
                nn.LeakyReLU(),
                nn.Linear(in_channels, 2, bias=bias),
                nn.LeakyReLU()
            ).cuda()
        self.alpha_delta_network.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.alpha_delta_network.parameters(), lr=0.01, weight_decay=0.0001)
        # input = torch.ones((16, in_channels, 3, 3))
        # if tensor.shape[2] != 1:
        #     out2 = self.alpha_delta_network(tensor)
        # else:
        #     out2 = self.alpha_delta_network(torch.flatten(tensor, start_dim=1, end_dim=3))
        # print("")
        self.epsilon = 0.1
        self.break_var2 = False
        if self.weight.shape == torch.Size([1024, 8192, 1, 1]):
            return

        output = torch.zeros(tensor.size(), device=tensor.device)
        delta = Delta(tensor)
        alpha = Alpha(tensor, delta)
        for i in range(tensor.size()[0]):
            pos_one = (tensor[i] > delta[i]).to(torch.float32)
            neg_one = -1 * (tensor[i] < -delta[i]).to(torch.float32)
            out = torch.add(pos_one, neg_one)
            output[i] = torch.add(output[i], torch.mul(out, alpha[i]))

        lossf = torch.sum((tensor - output) ** 2)

        up = torch.amax(tensor, dim=(1, 2, 3))
        down = torch.amin(tensor, dim=(1, 2, 3))
        rangew = up - down
        self.alpha_delta_network.requires_grad_(True)
        break_var = False
        for i in range(1 if tensor.shape[2] != 1 else 0):
            self.optimizer.zero_grad()
            if tensor.shape[2] != 1:
                w, alpha2 = self.alpha_delta_network(tensor)
            else:
                alpha_delta = self.alpha_delta_network(torch.flatten(tensor, start_dim=1, end_dim=3))
            w_ = self.fw_(w)
            output2 = (alpha2 * rangew / (2 + self.epsilon))[:, None, None, None] * w_
            # output2 = (alpha2 * 40)[:, None, None, None] * w_
            loss = torch.sum((tensor - output2) ** 2)
            # print(f"i: {i}, loss: {loss.item()}, loss3: {1}")
            loss.backward()
            self.optimizer.step()
            if break_var:
                break
        self.alpha_delta_network.requires_grad_(False)

        with torch.no_grad():
            w__ = self.fw__(w_)
            output3 = (alpha2 * rangew / (2 + self.epsilon))[:, None, None, None] * w__
            loss3 = torch.sum((tensor - output3) ** 2)

        print(f"lossf: {lossf.item()}, loss3: {loss3.item()}")
        print("")

    def fw_(self, x):
        _x = 3 * x
        v1 = _x * self.epsilon
        v2 = (_x + 0.5) * (self.epsilon) + (-1 + self.epsilon / 2)
        v3 = (_x - 0.5) * (self.epsilon) + (1 - self.epsilon / 2)
        return torch.logical_and(_x >= -0.5, _x <= 0.5) * v1 + (_x < -0.5) * v2 + (_x > 0.5) * v3

        # if x >= -0.5 and x <= 0.5:
        #     return x * (epsilon)
        # elif x < -0.5:
        #     return (x + 0.5) * (epsilon) + (-1 + epsilon / 2)
        # elif x > 0.5:
        #     return (x - 0.5) * (epsilon) + (1 - epsilon / 2)

    def fw__(self, x):
        _x = x
        v1 = 0
        v2 = -1
        v3 = +1
        return torch.logical_and(_x >= -0.5, _x <= 0.5) * v1 + (_x < -0.5) * v2 + (_x > 0.5) * v3

    def forward(self, x):
        if (self.break_var2 and (self.weight.shape != torch.Size([128, 3, 3, 3]) and self.weight.shape != torch.Size([128, 128, 3, 3])
            and self.weight.shape != torch.Size([256, 128, 3, 3]) and self.weight.shape != torch.Size([256, 256, 3, 3])
            and self.weight.shape != torch.Size([512, 256, 3, 3]) and self.weight.shape == torch.Size([512, 512, 3, 3]))):
            tensor = self.weight.clone().detach().cuda()

            # print("================================")
            output = torch.zeros(tensor.size(), device=tensor.device)
            delta = Delta(tensor)
            alpha = Alpha(tensor,delta)
            for i in range(tensor.size()[0]):
                pos_one = (tensor[i] > delta[i]).to(torch.float32)
                neg_one = -1 * (tensor[i] < -delta[i]).to(torch.float32)
                out = torch.add(pos_one, neg_one)
                output[i] = torch.add(output[i],torch.mul(out, alpha[i]))

            lossf = torch.sum((tensor - output)**2)

            up = torch.amax(tensor, dim=(1, 2, 3))
            down = torch.amin(tensor, dim=(1, 2, 3))
            rangew = up - down
            self.alpha_delta_network.requires_grad_(True)
            break_var = False
            for i in range(200000 if tensor.shape[2] != 1 else 0):
                self.optimizer.zero_grad()
                if tensor.shape[2] != 1:
                    w, alpha2 = self.alpha_delta_network(tensor)
                else:
                    alpha_delta = self.alpha_delta_network(torch.flatten(tensor, start_dim=1, end_dim=3))
                w_ = self.fw_(w)
                output2 = (alpha2 * rangew / (2 + self.epsilon))[:, None, None, None] * w_
                # output2 = (alpha2 * 40)[:, None, None, None] * w_
                loss = torch.sum((tensor - output2)**2)
                with torch.no_grad():
                    w__ = self.fw__(w_)
                    output3 = (alpha2 * rangew / (2 + self.epsilon))[:, None, None, None] * w__
                    loss3 = torch.sum((tensor - output3) ** 2)
                print(f"i: {i}, lossf: {lossf.item()}, loss: {loss.item()}, loss3: {loss3.item()}")
                loss.backward()
                self.optimizer.step()
                if break_var:
                    break
            self.alpha_delta_network.requires_grad_(False)

            with torch.no_grad():
                w__ = self.fw__(w_)
                output3 = (alpha2 * rangew / (2 + self.epsilon))[:, None, None, None] * w__
                loss3 = torch.sum((tensor - output3) ** 2)

            # range_cover_by_alpha = alpha2[0] * (0.1 / 2 + (up[0] - delta2[0]) * (0.1 / delta2[0])) + alpha2[0] * ( 0.1 / 2 + (delta2[0] - down[0]) * (0.1 / delta2[0]))
            #
            # output3 = torch.zeros(tensor.size(), device=tensor.device)
            # for i in range(tensor.size()[0]):
            #     pos_one = (tensor[i] > delta2[i]).to(torch.float32)
            #     neg_one = -1 * (tensor[i] < -delta2[i]).to(torch.float32)
            #     out = torch.add(pos_one, neg_one)
            #     output3[i] = torch.add(output3[i],torch.mul(out, alpha2[i]))
            #
            # loss3 = torch.sqrt(torch.sum((tensor - output3)**2))

            self.lossn_track.append(loss3.item())
            self.lossf = lossf.item()
            # print(f"lossf: {lossf.item()}, lossn: {self.lossn_track}")
            # print("===============")

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
