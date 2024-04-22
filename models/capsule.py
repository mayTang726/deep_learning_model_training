from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils.helpers import softmax # to get transpose softmax function
import psutil

'''Part of format and full model from pytorch examples repo: https://github.com/pytorch/examples/blob/master/mnist/main.py'''
# class net(nn.Module):
#     # input_height, input_width = 224, 224
#     # input_channels = 3
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
def squash(input_tensor):
    '''Squashes an input Tensor so it has a magnitude between 0-1.
        param input_tensor: a stack of capsule inputs, s_j
        return: a stack of normalized, capsule output vectors, v_j
        '''
    squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
    scale = squared_norm / (1 + squared_norm) # normalization coeff
    output_tensor = scale * input_tensor / torch.sqrt(squared_norm)    
    return output_tensor

# convolutional layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        '''Constructs the ConvLayer with a specified input and output size.
           param in_channels: input depth of an image, default value = 1
           param out_channels: output depth of the convolutional layer, default value = 32
           '''
        super(ConvLayer, self).__init__()

        # defining a convolutional layer of the specified size
        self.conv = nn.Conv2d(in_channels, out_channels,    # 第一次的卷积一次之后大小变为(222 * 222)
                              kernel_size=3, stride=1, padding=0)
        print(111)
    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input to the layer; an input image
           return: a relu-activated, convolutional layer
           '''
        # applying a ReLu activation to the outputs of the conv layer
        features = F.relu(self.conv(x)) # will have dimensions (batch_size, 20, 20, 256)
        print(222)
        return features


# primary capsules layer
class PrimaryCaps(nn.Module):
    # 最底层的capsules = 8 个，input_channels 为convlayer的输出，为32
    def __init__(self, num_capsules=8, in_channels=32, out_channels=16):
        '''Constructs a list of convolutional layers to be used in 
           creating capsule output vectors.
           param num_capsules: number of capsules to create
           param in_channels: input depth of features, default value = 256
           param out_channels: output depth of the convolutional layers, default value = 32
           '''
        super(PrimaryCaps, self).__init__()

        # creating a list of convolutional layers for each capsule I want to create
        # all capsules have a conv layer with the same parameters
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, #这是convol层之后的第二次卷积，大小变为(220，220)
                      kernel_size=3, stride=1, padding=0)
            for _ in range(num_capsules)])
        print(333)
    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input; features from a convolutional layer
           return: a set of normalized, capsule output vectors
           '''
        # get batch size of inputs
        batch_size = x.size(0)
        # 上一层卷积之后的大小为 220 * 220，outputs为 16 
        # reshape convolutional layer outputs to be (batch_size, vector_dim=32 * 220 * 220, 1)
        u = [capsule(x).view(batch_size, 16 * 220 * 220, 1) for capsule in self.capsules]
        # stack up output vectors, u, one for each capsule
        u = torch.cat(u, dim=-1)
        # squashing the stack of vectors
        u_squash = squash(u)
        print(444)
        return u_squash


def calculate_cpu(time):
    cpu_usage = psutil.cpu_percent()
    print('第' + time + '：cpu使用率', cpu_usage)
    # 获取内存消耗情况
    memory_usage = psutil.virtual_memory().percent
    print('第' + time + '：内存消耗率', memory_usage)

# digit capsules
class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, previous_layer_nodes= 16 * 220 * 220, 
                 in_channels=8, out_channels=16):
        '''Constructs an initial weight matrix, W, and sets class variables.
           param num_capsules: number of capsules to create
           param previous_layer_nodes: dimension of input capsule vector, default value = 32 * 104 * 104
           param in_channels: number of capsules in previous layer, default value = 8
           param out_channels: dimensions of output capsule vector, default value = 16
           '''
        super(DigitCaps, self).__init__()

        # setting class variables
        self.num_capsules = num_capsules
        self.previous_layer_nodes = previous_layer_nodes # vector input (dim=1152)
        self.in_channels = in_channels # previous layer's number of capsules

        # starting out with a randomly initialized weight matrix, W
        # these will be the weights connecting the PrimaryCaps and DigitCaps layers
        self.W = nn.Parameter(torch.randn(num_capsules, previous_layer_nodes, 
                                          in_channels, out_channels))
        print(555)
    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        # # 获取 CPU 使用率
        calculate_cpu('1')
        # W @ x =
        # (1, out_caps, in_caps, out_dim, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, out_caps, in_caps, out_dims, 1)
        u_hat = torch.matmul(self.W, x)
        calculate_cpu('2')
        # (batch_size, out_caps, in_caps, out_dim)
        u_hat = u_hat.squeeze(-1)
        calculate_cpu('3')
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()
        calculate_cpu('4')
        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1).to(self.device)
        calculate_cpu('5')
        for route_iter in range(3 - 1):
            calculate_cpu('6')
            # (batch_size, out_caps, in_caps, 1) -> Softmax along out_caps
            c = b.softmax(dim=1)
            calculate_cpu('7')
            # element-wise multiplication
            # (batch_size, out_caps, in_caps, 1) * (batch_size, in_caps, out_caps, out_dim) ->
            # (batch_size, out_caps, in_caps, out_dim) sum across in_caps ->
            # (batch_size, out_caps, out_dim)
            s = (c * temp_u_hat).sum(dim=2)
            calculate_cpu('8')
            # apply "squashing" non-linearity along out_dim
            v = squash(s)
            calculate_cpu('9')
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, out_caps, in_caps, out_dim) @ (batch_size, out_caps, out_dim, 1)
            # -> (batch_size, out_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            calculate_cpu('10')
            b += uv
            calculate_cpu('11')
        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        calculate_cpu('12')
        s = (c * u_hat).sum(dim=2)
        calculate_cpu('13')
        # apply "squashing" non-linearity along out_dim
        v = squash(s)
        calculate_cpu('14')
        return v
        # '''Defines the feedforward behavior.
        #    param u: the input; vectors from the previous PrimaryCaps layer
        #    return: a set of normalized, capsule output vectors
        #    '''
        # TRAIN_ON_GPU = torch.cuda.is_available()
        # if(TRAIN_ON_GPU):
        #     print('Training on GPU!')
        # else:
        #     print('Only CPU available')

        # # adding batch_size dims and stacking all u vectors
        # u = u[None, :, :, None, :]
        # calculate_cpu('1')
        # # 4D weight matrix
        # W = self.W[:, None, :, :, :]
        # calculate_cpu('2')
        # # calculating u_hat = W*u
        # u_hat = torch.matmul(u, W)
        # calculate_cpu('3')
        # # getting the correct size of b_ij
        # # setting them all to 0, initially
        # b_ij = torch.zeros(*u_hat.size())
        # calculate_cpu(4)
        # # 获取 CPU 使用率
        

        # # moving b_ij to GPU, if available
        # if TRAIN_ON_GPU:
        #     b_ij = b_ij.cuda()

        # # update coupling coefficients and calculate v_j
        # v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)
        # print(666)
        # return v_j # return final vector outputs

# linear layers reconstructor the data to encoding itself
class Decoder(nn.Module):
    def __init__(self, input_vector_length=16, input_capsules=10, hidden_dim=512):
        '''Constructs an series of linear layers + activations.
           param input_vector_length: dimension of input capsule vector, default value = 16
           param input_capsules: number of capsules in previous layer, default value = 10
           param hidden_dim: dimensions of hidden layers, default value = 512
           '''
        super(Decoder, self).__init__()
        
        # calculate input_dim
        input_dim = input_vector_length * input_capsules
        
        # define linear layers + activations
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # first hidden layer
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2), # second, twice as deep
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, 224 * 224), # can be reshaped into 224*224 image
            nn.Sigmoid() # sigmoid activation to get output pixel values in a range from 0-1, for matching pixels value region
            )
        print(888)
    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input; vectors from the previous DigitCaps layer
           return: two things, reconstructed images and the class scores, y
           '''
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        
        # find the capsule with the maximum vector length
        # here, vector length indicates the probability of a class' existence
        _, max_length_indices = classes.max(dim=1)
        
        # create a sparse class matrix
        sparse_matrix = torch.eye(3) # 10 is the number of classes
        if TRAIN_ON_GPU:
            sparse_matrix = sparse_matrix.cuda()
        # get the class scores from the "correct" capsule
        y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)
        
        # create reconstructed pixels
        x = x * y[:, :, None]
        # flatten image into a vector shape (batch_size, vector_dim)
        flattened_x = x.contiguous().view(x.size(0), -1)
        # create reconstructed image vectors
        reconstructions = self.linear_layers(flattened_x)
        print(999)
        # return reconstructions and the class scores, y
        return reconstructions, y

# dynamic routing
def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    print(1212)
    '''Performs dynamic routing between two capsule layers.
       param b_ij: initial log probabilities that capsule i should be coupled to capsule j
       param u_hat: input, weighted capsule vectors, W u
       param squash: given, normalizing squash function
       param routing_iterations: number of times to update coupling coefficients
       return: v_j, output capsule vectors
       '''    
    # update b_ij, c_ij for number of routing iterations
    for iteration in range(routing_iterations):
        # softmax calculation of coupling coefficients, c_ij
        c_ij = softmax(b_ij, dim=2)

        # calculating total capsule inputs, s_j = sum(c_ij*u_hat)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)

        # squashing to get a normalized vector output, v_j
        v_j = squash(s_j)

        # if not on the last iteration, calculate agreement and new b_ij
        if iteration < routing_iterations - 1:
            # agreement
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            
            # new b_ij
            b_ij = b_ij + a_ij
    
    return v_j # return latest v_j

# complete network
class net(nn.Module):
    def __init__(self):
        '''Constructs a complete Capsule Network.'''
        super(net, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()
                
    def forward(self, images):
        '''Defines the feedforward behavior.
           param images: the original MNIST image input data
           return: output of DigitCaps layer, reconstructed images, class scores
           '''
        primary_caps_output = self.primary_capsules(self.conv_layer(images))
        output = self.digit_capsules(primary_caps_output).squeeze().transpose(0,1)
        reconstructions, y = self.decoder(output) #y暂时没用到
        return output, reconstructions

# loss
class CapsuleLoss(nn.Module):
    def __init__(self):
        '''Constructs a CapsuleLoss module.'''
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='sum') # cumulative loss, equiv to size_average=False
        print(3434)
    def forward(self, x, labels, images, reconstructions):
        '''Defines how the loss compares inputs.
           param x: digit capsule outputs
           param labels: 
           param images: the original MNIST image input data
           param reconstructions: reconstructed MNIST image data
           return: weighted margin and reconstruction loss, averaged over a batch
           '''
        batch_size = x.size(0)

        ##  calculate the margin loss   ##
        
        # get magnitude of digit capsule vectors, v_c
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        # calculate "correct" and incorrect loss
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        
        # sum the losses, with a lambda = 0.5
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        ##  calculate the reconstruction loss   ##
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        print(788)
        # return a weighted, summed loss, averaged over a batch size
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
    

def train(model, device, train_loader, optimizer):
    model.train()
    cost = CapsuleLoss()
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_id, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            # label = label.to(device)
            label = torch.eye(3).index_select(dim=0, index=label)
            print(8989)
            output, reconstructions = model(data)
            loss = cost(output, label, data, reconstructions)
            print(6767)
            loss.backward()
            optimizer.step()
            progress_bar.update(1)

def val(model, device, val_loader, checkpoint=None):
    print(9090)
    if checkpoint is not None: # 恢复状态，调试和分析，恢复训练
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)     
    # Need this line for things like dropout etc.  
    model.eval()
    preds = []
    targets = []
    cost = CapsuleLoss()
    losses = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to(device)
            target = label.clone()
            label = torch.eye(10).index_select(dim=0, index=label)
            output, reconstructions = model(data)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            losses.append(cost(output, label, data, reconstructions))
    loss = np.mean(losses)
    preds = np.argmax(np.concatenate(preds), axis=1)
    print('preds',preds)
    targets  = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return loss, acc

def test(model, device, test_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)    
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            target = label.clone()
            output = model(data)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
        
    preds = np.argmax(np.concatenate(preds), axis=1)
    targets  = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return acc

