import argparse
import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from models import Generator
from datasets import ImageDataset
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def test():
    ## 超参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='E:\ZRM\data\\nmi_pair\\test', help='root directory of the dataset')
    parser.add_argument('--channels', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=128, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='D:/Python-project/Dual-gan/saved_models/edges2shoes/G_AB_1.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='D:/Python-project/Dual-gan/saved_models/edges2shoes/G_BA_1.pth', help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    #################################
    ##          test准备工作        ##
    #################################

    ## input_shape:(3, 256, 256)
    input_shape = (opt.channels, opt.size, opt.size)
    ## 创建生成器，判别器对象
    netG_A2B = Generator()
    netG_B2A = Generator()

    ## 使用cuda
    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()

    ## 载入训练模型参数
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A))


    ## 设置为测试模式
    netG_A2B.eval()
    netG_B2A.eval()

    ## 创建一个tensor数组
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.channels, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.channels, opt.size, opt.size)

    '''构建测试数据集'''
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
    #                         batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    dataloader = DataLoader(
        ImageDataset("E:\data\zrm_dataset\\nmi_pair\\test", transforms_=transforms_, mode='test'),
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=0
    )

    #################################
    ##           test开始          ##
    #################################

    '''如果文件路径不存在, 则创建一个 (存放测试输出的图片)'''
    if not os.path.exists('output/A'):
        os.makedirs('output/A')
    if not os.path.exists('output/B'):
        os.makedirs('output/B')

    for i, batch in enumerate(dataloader):
        ## 输入数据 real
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        ## 通过生成器生成的 fake
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)
        ## 保存图片
        save_image(fake_A, 'output/A/%04d.png' % (i + 1))
        save_image(fake_B, 'output/B/%04d.png' % (i + 1))
        print('processing (%04d)-th image...' % (i))
    print("测试完成")


if __name__ == '__main__':
    test()
