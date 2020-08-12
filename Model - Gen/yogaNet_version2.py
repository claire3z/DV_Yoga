import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from reference import yoga_dict
# from percepLoss import VGGPerceptualLoss

import gc

##########################################################################################################
#   Model Classes - ResidualBlock, VUNet                                                                 #
##########################################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1,use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1,stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=False)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)),inplace=False)
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X    # elementwise addition with same depth, not concat
        return F.relu(Y,inplace=False)

# Modified from https://d2l.ai/chapter_convolutional-modern/resnet.html
# Testing
# resiB = ResidualBlock(1,3)
# X = torch.Tensor([[[[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4],[5,5,5,5,5,5],[6,6,6,6,6,6]]]])
# X.shape #torch.Size([1, 1, 6, 6])
# Y = resiB(X)
# Y.shape #torch.Size([1, 3, 6, 6])


class YogaNet(nn.Module):

    def __init__(self, input_channels=3, output_channels = 3, num_channels=64):
        super().__init__()
        self.conv_initial = nn.Conv2d(input_channels,num_channels,kernel_size=1)
        self.conv_final = nn.Conv2d(2*num_channels,output_channels,kernel_size=1)
        self.conv_bottom = nn.Conv2d(num_channels,1,kernel_size=1)     # out_channel set to 1 to reduce size before fc
        self.conv_topup = nn.Conv2d(2, num_channels, kernel_size=1)

        self.downsample_1 = nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=[2,2],padding=1)
        self.downsample_2 = nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=[2,2],padding=1)
        self.downsample_1s = nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=[2,2],padding=1)
        self.downsample_2s = nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=[2,2],padding=1)

        self.upsample_1_subpixel = nn.PixelShuffle(upscale_factor=2)      # utilize subpixel convolution; NOT nn.ConvTranspose2d(num_channels,num_channels,kernel_size=3,stride=[2,2],padding=0)
        self.upsample_1_conv = nn.Conv2d(int(num_channels*2/4),num_channels,kernel_size=3,stride=1,padding=1) #?

        self.upsample_2_subpixel = nn.PixelShuffle(upscale_factor=2)      # utilize subpixel convolution; NOT nn.ConvTranspose2d(num_channels,num_channels,kernel_size=3,stride=[2,2],padding=0)
        self.upsample_2_conv = nn.Conv2d(int(num_channels*2/4),num_channels,kernel_size=3,stride=1,padding=1) #?

        self.res_block_down_1 = ResidualBlock(num_channels,num_channels,strides=1,use_1x1conv=False)
        self.res_block_down_3 = ResidualBlock(num_channels,num_channels,strides=1,use_1x1conv=False)
        self.res_block_down_5 = ResidualBlock(num_channels,num_channels,strides=1,use_1x1conv=False)

        self.res_block_down_1s = ResidualBlock(num_channels,num_channels,strides=1,use_1x1conv=False)
        self.res_block_down_3s = ResidualBlock(num_channels,num_channels,strides=1,use_1x1conv=False)
        self.res_block_down_5s = ResidualBlock(num_channels,num_channels,strides=1,use_1x1conv=False)

        self.res_block_up_1 = ResidualBlock(2*num_channels,num_channels,strides=1,use_1x1conv=True)      # enable use_1x1conv to conform input 128 channels into 64 channels before elementwise addition with output
        self.res_block_up_3 = ResidualBlock(2*num_channels,num_channels,strides=1,use_1x1conv=True)      # enable use_1x1conv to conform input 128 channels into 64 channels before elementwise addition with output
        self.res_block_up_5 = ResidualBlock(2*num_channels,num_channels,strides=1,use_1x1conv=True)      # enable use_1x1conv to conform input 128 channels into 64 channels before elementwise addition with output

        self.fc1 = torch.nn.Linear(25*25+20,512)  # hierarchical class label (6+6+8) one-hot key coding
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 64)         # for mu
        self.fc4 = torch.nn.Linear(256, 64)         # for sigma
        self.fc5 = torch.nn.Linear(64+20, 256)
        self.fc6 = torch.nn.Linear(256, 512)
        self.fc7 = torch.nn.Linear(512, 25*25*1)

    ### The original VUNEt implemented 8 residual blocks; here only 3 as an initial attempt

    def encode_x(self,X,C): #X size([batch_size, 3, 100, 100])
        self.x1 = self.conv_initial(X) #torch.Size([bs, 64, 100, 100])
        self.x2a = self.res_block_down_1(self.x1) #torch.Size([bs, 64, 100, 100])
        self.x3 = self.downsample_1(self.x2a) #torch.Size([bs, 64, 50, 50])
        self.x4a = self.res_block_down_3(self.x3) #torch.Size([bs, 64, 50, 50])
        self.x5 = self.downsample_2(self.x4a) #torch.Size([bs, 64, 25, 25])
        self.x6a = self.res_block_down_5(self.x5) #torch.Size([bs, 64, 25, 25])
        self.x7 = self.conv_bottom(self.x6a) #torch.Size([bs, 1, 25, 25])

        batch_size = X.shape[0]
        x_ = self.x7.view(batch_size,-1)   #torch.Size([bs, 625])
        x_ = torch.cat((x_, C),dim=1) #torch.Size([bs, 625+20])
        x_ = F.relu(self.fc1(x_),inplace=False)  #torch.Size([bs, 512])
        x_ = F.relu(self.fc2(x_),inplace=False)  #torch.Size([bs, 256])
        mu_x = self.fc3(x_)          #torch.Size([bs, 64])
        logvar_x = self.fc4(x_)      #torch.Size([bs, 64])
        return mu_x,logvar_x


    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()  # _ in-place version of torch.exp()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def decode(self,z_x,C):
        z_ = torch.cat((z_x,C),dim=1)
        z_ = F.relu(self.fc5(z_),inplace=False)       # z_=torch.Size([bs, 256])
        z_ = F.relu(self.fc6(z_),inplace=False)       # z_=torch.Size([bs, 512])
        z_ = F.relu(self.fc7(z_),inplace=False)       # z_=torch.Size([bs, 625]) #removed activation torch.tanh(self.fc7(out))

        x_7 = z_.view(-1,1,25,25)               # size (bs,1,25,25)
        x_7 = torch.cat((x_7,self.x7),dim=1)    # x_7=([bs, 1+1+1, 25, 25]), x7=([bs, 1, 25, 25]), s7=torch.Size([bs, 1, 25, 25])

        x_6b = self.conv_topup(x_7)              # to restore channel size from 3 to 64 in order for res-block
        x_6a = torch.cat((x_6b,self.x6a),dim=1)    # torch.Size([bs, 64*3, 25, 25])

        x_5 = self.res_block_up_5(x_6a)            # torch.Size([bs, 64, 25, 25])
        x_5 = torch.cat((x_5,self.x5),dim=1)    # torch.Size([bs, 64*3, 25, 25])

        x_4 = self.upsample_2_subpixel(x_5)       # torch.Size([bs, 64*3/4, 50, 50])
        x_4b = self.upsample_2_conv(x_4)           # torch.Size([bs, 64, 50, 50]) in_channel=32, numchannel = 64 restored
        x_4a = torch.cat((x_4b,self.x4a), dim=1)  # torch.Size([bs, 64*3, 100, 100])

        x_3 = self.res_block_up_3(x_4a)            # torch.Size([bs, 64, 100, 100])
        x_3 = torch.cat((x_3,self.x3), dim=1)  # torch.Size([bs, 64*3, 100, 100])

        x_2 = self.upsample_1_subpixel(x_3)       # torch.Size([bs, 64*3/4, 100, 100])
        x_2b = self.upsample_1_conv(x_2)           # torch.Size([bs, 64, 100, 100]) in_channel=32, numchannel = 64 restored
        x_2a = torch.cat((x_2b,self.x2a), dim=1)  # torch.Size([bs, 64*3, 100, 100])

        x_1 = self.res_block_up_1(x_2a)            # torch.Size([bs, 64, 100, 100])
        x_1 = torch.cat((x_1, self.x1), dim=1)  # torch.Size([bs, 64*3, 100, 100])

        recon_X = self.conv_final(x_1)          #torch.Size([bs, 3, 100, 100])
        return recon_X


    def forward(self,X,C):
        mu_x, logvar_x = self.encode_x(X,C)
        z_x = self.reparameterize(mu_x, logvar_x)

        recon_X = self.decode(z_x,C)
        return recon_X, mu_x, logvar_x, z_x

    def inference(self,z_x,C):
        gen_X = self.decode(z_x,C)
        return gen_X



##########################################################################################################
#   Functions - loss_function, onehot, get_all_data                                                      #
##########################################################################################################


def loss_function_c(X, recon_X, mu_x, logvar_x, batch_size, img_size, img_channels):

    pixLoss = F.mse_loss(X, recon_X)

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_x = -0.5 * torch.sum(1 + logvar_x - mu_x.pow(2) - logvar_x.exp())

    # Normalize
    KLD = KLD_x #/ batch_size * (img_size * img_size * img_channels)

    print('pixLoss=',pixLoss.item(), 'KLD=',KLD.item(),'total=',1*KLD.item() + 100*pixLoss.item())
    return 1*KLD + 100*pixLoss


def onehot(labels_raw,yoga_dict=yoga_dict):
    """
    Convert raw labels (6,20,82) into relative one-hot format, where the first 6: level 1, the next 6: level 2 and the last 8: level 3
    :param labels_raw: torch.Tensor, shape([batch_size, 3])
    :return: torch.Tensor, shape([batch_size, 1, 20])
    """
    # assert labels_raw.dim() == 2
    # assert yoga_dict, 'from reference import yoga_dict'

    batch_size = labels_raw.size(0)
    labels_relative = torch.zeros(batch_size,1,20)
    for i in range(batch_size):
        k = str(int(labels_raw[i][-1].item()))
        relative_idx = yoga_dict[k][-1]
        labels = torch.Tensor(relative_idx).unsqueeze(1)
        labels_relative[i].scatter_(1, labels[0].unsqueeze(1).long(), 1) #self.long() is equivalent to self.to(torch.int64)
        labels_relative[i].scatter_(1, 6 + labels[1].unsqueeze(1).long(), 1)
        labels_relative[i].scatter_(1, 6 + 6 + labels[2].unsqueeze(1).long(), 1)
    return labels_relative.squeeze(dim=1).float() ##NOT.type(torch.DoubleTensor) BUT float()

# Test
# labels_raw = torch.Tensor([[1,8,0],[1,5,1]])
# onehot(labels_raw).shape == torch.Size([2, 20])


# Get (data,labels,keypoints) into DataLoader class

def get_data_txt(dataFile, labelFile, keyptsFile, maskFile):
    data = np.loadtxt(dataFile)
    label = np.loadtxt(labelFile)
    keypoints = np.loadtxt(keyptsFile)
    mask = np.loadtxt(maskFile)
    dataset = []
    for i in range(data.shape[0]):
        dataset.append((data[i],label[i],keypoints[i],mask[i]))
    return dataset


def load_data(dataset,batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def process_data(data_raw,label_raw,keypoints_raw,mask_raw):
    data = data_raw.reshape(-1,img_size,img_size,3).permute(0,3,1,2) ## from np [n,100,100,3] to tensor [n,3,100,100]
    if data.max() >1:
        data /= data.max()
    data = data.float()
    keypoints = keypoints_raw.reshape(-1,17,2)
    label = onehot(label_raw)
    mask = mask_raw.reshape(-1,img_size,img_size,1).permute(0,3,1,2)
    return data, label, keypoints, mask


def mask_on(X, mask, hurdle = 0.5):
    mask = mask > hurdle
    X = X * mask
    return X


def get_keypoints(X):
    """
    :param X: input images, Tensor [bs,3,100,100]
    :return: keypoints, Tensor [bs,17,2]
    """
    assert X.dim() == 4 and X.shape[1] == 3 and X.get_device() == -1  #device(type='cpu')
    if X.max()>1:
        X /= X.max()    # normalization 0-1
    keypoints = torch.zeros([X.shape[0],17,2])
    model_k = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=2, num_keypoints=17, pretrained_backbone=True)
    model_k.eval()
    # model_k.to(device) # take model_k off GPU to save space
    for i, x in enumerate(X):
        predictions = model_k(x.view(-1,3,100,100))
        kpts = predictions[0]['keypoints']
        if kpts.shape[0] == 0:
            kpts = torch.zeros([1, 17, 2])
        keypoints[i] = kpts[0, :, 0:2]
    return keypoints


def draw_stickman(X,keypoints):
    '''
    :param X: Tensor [bs,3,100,100]
    :param keypoints: Tensor [bs,17,2]
    :return: stickman same size as input X, Tensor [bs,3,100,100]
    '''
    # assert X.dim() == 4 and X.shape[1] == 3
    # assert keypoints.dim() == 3 and keypoints.shape[1] == 17
    S = np.zeros((X.shape[0],X.shape[2],X.shape[3],X.shape[1])) # S.shape =(bs,100,100,3)
    for idx, kpts in enumerate(keypoints):
        img = np.zeros(S.shape[1:]) # draw on clean sheet (100,100,3)
        # draw circles at key points
        for k in range(17):
            i = int(kpts[k][0])
            j = int(kpts[k][1])
            cv2.circle(img=img, center=(int(i), int(j)), radius=2, color=(255, 0, 0), thickness=-1, lineType=cv2.FILLED) #cv2 only takes int (255,255,255)
        # connect key points with line segments
        # COCO_PERSON_KEYPOINT_NAMES = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
        connectors = [[15,13],[13,11],[16,14],[14,12],[11,12],[5,11],[6,12],[5,6], [5,7],[6,8],[7,9],[8,10],[1,2],[0,1],[0,2],[1,3],[3,4],[3,5],[4,6]]
        for line in connectors:
            start_idx = line[0]
            end_idx = line[1]
            start_pt = (int(kpts[start_idx][0]),int(kpts[start_idx][1]))
            end_pt = (int(kpts[end_idx][0]),int(kpts[end_idx][1]))
            cv2.line(img,start_pt,end_pt, (0,255,0), 1)
        S[idx] = img
    S = np.transpose(S,(0,3,1,2))/255 #S.shape = (bs, 3, 100,100) for further processing
    return torch.Tensor(S).float()



def plot_pt(X,idx):
    '''
    :param X: Tensor [bs, 3, 100, 100]
    :param idx: int (0 ~ bs-1) index of selected data in the batch
    :return: display image based on selected data
    '''
    assert X.dim() == 4 and X.shape[1] == 3
    X = X.permute(0,2,3,1) # from tensor [n,3,100,100] to np [n,100,100,3]
    img = X[idx].numpy()
    if img.max()>1:
        img /= img.max()
    plt.imshow(img)



def visualize(data,label,stickman, idices:list, saveFile=None):
    '''
    :param data: Tensor [bs,3,100,100]
    :param label: Tensor [bs,3]
    :param keypoints: Tensor [bs,3,100,100]
    :param idices: list of indices of data selected to be displayed
    :param saveFile: file path; if None, do not save
    :return: display selected images and the correspondong stickman
    '''
    assert data.dim() == 4 and data.shape[1] == 3
    n = len(idices)
    for i in range(n):
        plt.subplot(2,n,i+1)
        plot_pt(data,idices[i])
        plt.axis('off')
        # plt.title(label[idices[i]]) #TODO: fix title with proper label

        plt.subplot(2,n,i+1+n)
        plot_pt(stickman.reshape(-1,3,100,100),idices[i])
        plt.axis('off')
    if saveFile:
        plt.savefig(saveFile)
    plt.show()
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


##########################################################################################################
#   INPUTS                                                                                               #
##########################################################################################################

dataFile = 'data_train_10.txt' #'data_test_10.txt'      #'Data/data_100.txt' #'data_test.txt' #685MB  #      #"toy_20_data.pt"     #"toy_5_data.pt"
labelFile = 'labels_train_10.txt' #'labels_test_10.txt'   #'Data/label_100.txt' #'labels_test.txt'             #"toy_20_labels.pt"   #"toy_5_labels.pt"
keyptsFile = 'data_train_keypts_10.txt' #'data_test_keypts_10.txt'  #'Data/keypoints_100.txt' #'data_test_keypts.txt'      #"toy_20_keypts.pt"   #"toy_5_keypts.pt"
maskFile = 'data_train_mask_10.txt' #'data_test_mask_10.txt'

# this takes a long time to execute...
dataset = get_data_txt(dataFile,labelFile,keyptsFile,maskFile)
print(len(dataset),len(dataset[0]),len(dataset[1][0]))


batch_size = 32
img_size = 100
img_channels = 3
learning_rate = 1e-3
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") #device(type='cuda', index=0)

model = YogaNet().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
print('model parameters =',count_parameters(model)/1e6) # 2.6M

data_loader = load_data(dataset,batch_size)
print('size of data_loader =',len(data_loader))

# Issues with CUDA memory
# RuntimeError: CUDA out of memory. Tried to allocate 80.00 MiB (GPU 0; 4.00 GiB total capacity; 2.42 GiB already allocated; 69.90 MiB free; 2.54 GiB reserved in total by PyTorch)
# ...reduce the batch size until your code will run without this error --> 32x 10x 5x doesn't work; batchsize=2 only works for 7 batches. =(
# cmd>>>nvidia-smi
# sys.getsizeof(b.storage())

gc.collect()
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
import datetime

# Training
model.train()
print("Training model >>>")

def training(epoch):
    print('\nTraining Epoch #{}\n'.format(epoch))
    train_loss = 0

    for batch_idx, (data_raw, label_raw, keypoints_raw,mask_raw) in enumerate(data_loader):
        if batch_idx == len(data_loader)-1: # to exclude the last few samples which are not enough to form a batch
            break
        print('>> Batch #{}/{}'.format(batch_idx + 1, len(data_loader) - 1), datetime.datetime.now())
        data, label, keypoints,mask = process_data(data_raw,label_raw,keypoints_raw,mask_raw)
        stickman = draw_stickman(data, keypoints)
        # print('processed', data.shape, label.shape, keypoints.shape, stickman.shape)
        # visualize(data, label, stickman, [0,1,2],saveFile='dog.png')
        # break

        stickman[:,2,:,:] = mask.squeeze(1)
        data_in = stickman

        data_in = data_in.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        recon_X, mu_x, logvar_x, z_x = model(X=data_in,C=label) # forward return
        loss = loss_function_c(data_in, recon_X, mu_x, logvar_x, batch_size, img_size, img_channels)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / (len(data_loader)-1)
    print('Training Set Average Loss: {:.1f}'.format(avg_loss))
    if epoch % 10 ==0:
        torch.save(model, 'Model/model_v10_epoch_{}.pt'.format(epoch))

num_Epoch = 21
for epoch in range(num_Epoch):
    training(epoch)


dataset_test = get_data_txt(dataFile = 'data_test_10.txt',labelFile='labels_test_10.txt',keyptsFile= 'data_test_keypts_10.txt',maskFile= 'data_test_mask_10.txt')
print(len(dataset_test),len(dataset_test[0]))
test_loader = load_data(dataset_test,batch_size=1)
print('size of data_loader =',len(test_loader))

## Inferencing
# model = YogaNet().to(device)
# model = torch.load('Model/model_v10_epoch_0.pt')
model.eval()
print("Inferencing from model")


def generate_pose(num,target_label,fileName): ##[0, 3, 74] = Warrior_II
    with torch.no_grad():
        test_loss = 0
        for i, (data_raw, label_raw,keypoints_raw, mask_raw) in enumerate(test_loader):
            if i == num:
                break
            print('Batch #{}/{}'.format(i+1, len(test_loader) - 1))

            data, label, keypoints, mask = process_data(data_raw, label_raw, keypoints_raw, mask_raw)
            stickman = draw_stickman(data, keypoints)  # already normalized and asfloat

            stickman[:, 2, :, :] = mask.squeeze(1)
            data_in = stickman
            data_in = data_in.to(device)

            target = np.array(target_label).repeat(1, axis=0)
            target = torch.Tensor(target)
            target = onehot(target)
            target = target.to(device)

            recon_X, mu_x, logvar_x, z_x = model(X=data_in, C=target)  # forward return
            loss = loss_function_c(data_in, recon_X, mu_x, logvar_x, batch_size, img_size, img_channels)
            test_loss += loss.item()
            gen_X = model.inference(z_x=torch.ones([1,64]).to(device), C=target)

            plt.subplot(2, 2, 1)
            plt.title("original")
            plt.axis('off')
            plot_pt(data.cpu(), 0)

            plt.subplot(2, 2, 2)
            plt.title("mask_on stickman")
            plt.axis('off')
            plot_pt(data_in.cpu(), 0)

            plt.subplot(2, 2, 3)
            plt.title("reconstructed")
            plt.axis('off')
            plot_pt(recon_X.cpu(), 0)

            plt.subplot(2, 2, 4)
            plot_pt(gen_X.cpu(), 0)
            plt.axis('off')
            plt.title('generated')
            plt.savefig(fileName)
            plt.show()
    print(test_loss/num)

generate_pose(num=1,target_label=[[0, 3, 74]],fileName='v10_output_17.png')