import torch
from torch import nn

class ForwardWarp_weight(nn.Module):
    """docstring for WarpLayer"""
    def __init__(self,):
        super(ForwardWarp_weight, self).__init__()
        self.eps = 1e-6

    def forward(self, img, counts, flo):
        """
            -img: image (N, 1, H, W)
            counts: (N,1,H,W)
            -flo: optical flow (N, 2, H, W)
            elements of flo is in [0, H] and [0, W] for dx, dy #注意纵轴为x,横轴为y,dx=v,dy=u

        """
        N, C, _, _ = img.size()

        # translate start-point optical flow to end-point optical flow
        y = flo[:, 0:1 :, :]
        x = flo[:, 1:2, :, :]

        x = x.repeat(1, C, 1, 1)
        y = y.repeat(1, C, 1, 1)

        # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
        x1 = torch.floor(x)
        x2 = x1 + 1
        y1 = torch.floor(y)
        y2 = y1 + 1

        # firstly, get gaussian weights
        w11, w12, w21, w22 = self.get_gaussian_weights_2(x, y, x1, x2, y1, y2)

        # secondly, sample each weighted corner 
        img11, o11 = self.sample_one(img, x1, y1, w11, counts)
        img12, o12 = self.sample_one(img, x1, y2, w12, counts)
        img21, o21 = self.sample_one(img, x2, y1, w21, counts)
        img22, o22 = self.sample_one(img, x2, y2, w22, counts)


        imgw = img11 + img12 + img21 + img22
        o = o11 + o12 + o21 + o22
        
        imgw_averaged = imgw/(o+self.eps)

        return imgw_averaged, o

    def get_gaussian_weights(self, x, y, x1, x2, y1, y2):
        w11 = torch.exp(-((x - x1)**2 + (y - y1)**2))
        w12 = torch.exp(-((x - x1)**2 + (y - y2)**2))
        w21 = torch.exp(-((x - x2)**2 + (y - y1)**2))
        w22 = torch.exp(-((x - x2)**2 + (y - y2)**2))

        return w11, w12, w21, w22 #尽量让warp前后的平均像素值保持不变
    
    def get_linear_weights(self, x, y, x1, x2, y1, y2):
        l11 = torch.sqrt((x-x1)**2+(y-y1)**2)
        l12 = torch.sqrt((x-x1)**2+(y-y2)**2)
        l21 = torch.sqrt((x-x2)**2+(y-y1)**2)
        l22 = torch.sqrt((x-x2)**2+(y-y2)**2)
        lsum = l11+l12+l21+l22
        
        w11 = (lsum-l11)/(3*lsum)
        w12 = (lsum-l12)/(3*lsum)
        w21 = (lsum-l21)/(3*lsum)
        w22 = (lsum-l22)/(3*lsum)

        return w11, w12, w21, w22
    
    def get_gaussian_weights_2(self, x, y, x1, x2, y1, y2):
        w11 = torch.exp(-((x - x1)**2 + (y - y1)**2))
        w12 = torch.exp(-((x - x1)**2 + (y - y2)**2))
        w21 = torch.exp(-((x - x2)**2 + (y - y1)**2))
        w22 = torch.exp(-((x - x2)**2 + (y - y2)**2))
        Sumw = w11+w12+w21+w22
        
        w11_norm = w11/Sumw
        w12_norm = w12/Sumw
        w21_norm = w21/Sumw
        w22_norm = w22/Sumw

        return w11_norm, w12_norm, w21_norm, w22_norm 

    def sample_one(self, img, shiftx, shifty, weight, counts):
        """
        Input:
            -img (N, 1, H, W), averaged timestamp image
            -shiftx, shifty (N, c, H, W) #取整之后的像素运动
            counts: event counts for every averaged timestamp
        """
        N, C, H, W = img.size()

        # flatten all (all restored as Tensors)
        flat_shiftx = shiftx.view(-1)#矩阵拉开成向量，按行拉开，x方向移动
        flat_shifty = shifty.view(-1)#矩阵拉开成向量，按行拉开，y方向移动
        '''
        [0 0 0 0 ... 0]
        [1 1 1 1 ... 1]
        ...
        [H-1, H-1, ..., H-1]，然后再按行拉开成向量
        '''
        flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].cuda().long().repeat(N, C, 1, W).view(-1)
        '''
        [0 1 2 3 ... w-1]
        [0 1 2 3 ... w-1]
        ...
        [0 1 2 3 ... w-1]，然后再按行拉开成向量
        '''
        flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].cuda().long().repeat(N, C, H, 1).view(-1)
        flat_weight = weight.view(-1) #每个像素的此次前向移动的比重矩阵，按行拉开成向量
        flat_img = img.contiguous().view(-1) #图片，按行拉开成向量
        flat_counts = counts.contiguous().view(-1)


        # The corresponding positions in I1
        idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).long().cuda().repeat(1, C, H, W).view(-1)
        idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).long().cuda().repeat(N, 1, H, W).view(-1)
        # ttype = flat_basex.type()
        idxx = flat_shiftx.long() + flat_basex # 绝对位置加相对偏移等于变换后的绝对位置坐标
        idxy = flat_shifty.long() + flat_basey


        # recording the inside part the shifted
        mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W) #把移动出边界的点剔除

        # Mask off points out of boundaries
        ids = (idxn*C*H*W + idxc*H*W + idxx*W + idxy)#加上向量化后n和c方向引起的坐标偏移，x方向的偏移也是整行移动的
        ids_mask = torch.masked_select(ids, mask).clone().cuda() #把outliers都剔除掉，返回一个精简后的ids向量

        img_warp = torch.zeros([N*C*H*W, ]).cuda() #准备warp结果的template
        img_warp.put_(ids_mask, torch.masked_select(flat_img*flat_counts*flat_weight, mask), accumulate=True)#把剔除outliers之后的pixel们逐个累加到img_warp的指定座标处

        one_warp = torch.zeros([N*C*H*W, ]).cuda() 
        one_warp.put_(ids_mask, torch.masked_select(flat_counts*flat_weight, mask), accumulate=True)


        return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)

    
class ForwardWarp(nn.Module):
    """docstring for WarpLayer"""
    def __init__(self,):
        super(ForwardWarp, self).__init__()


    def forward(self, img, flo):
        """
            -img: image (N, C, H, W)
            -flo: optical flow (N, 2, H, W)
            elements of flo is in [0, H] and [0, W] for dx, dy #注意纵轴为x,横轴为y,dx=v,dy=u

        """


        # (x1, y1)		(x1, y2)
        # +---------------+
        # |				  |
        # |	o(x, y) 	  |
        # |				  |
        # |				  |
        # |				  |
        # |				  |
        # +---------------+
        # (x2, y1)		(x2, y2)


        N, C, _, _ = img.size()

        # translate start-point optical flow to end-point optical flow
        y = flo[:, 0:1 :, :]
        x = flo[:, 1:2, :, :]

        x = x.repeat(1, C, 1, 1)
        y = y.repeat(1, C, 1, 1)

        # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
        x1 = torch.floor(x)
        x2 = x1 + 1
        y1 = torch.floor(y)
        y2 = y1 + 1

        # firstly, get gaussian weights
        w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2)

        # secondly, sample each weighted corner 
        img11, o11 = self.sample_one(img, x1, y1, w11)
        img12, o12 = self.sample_one(img, x1, y2, w12)
        img21, o21 = self.sample_one(img, x2, y1, w21)
        img22, o22 = self.sample_one(img, x2, y2, w22)


        imgw = img11 + img12 + img21 + img22
        o = o11 + o12 + o21 + o22

        return imgw, o

    def get_gaussian_weights(self, x, y, x1, x2, y1, y2):
        w11 = torch.exp(-((x - x1)**2 + (y - y1)**2))
        w12 = torch.exp(-((x - x1)**2 + (y - y2)**2))
        w21 = torch.exp(-((x - x2)**2 + (y - y1)**2))
        w22 = torch.exp(-((x - x2)**2 + (y - y2)**2))

        return w11, w12, w21, w22 #尽量让warp前后的平均像素值保持不变
    
    def get_linear_weights(self, x, y, x1, x2, y1, y2):
        l11 = torch.sqrt((x-x1)**2+(y-y1)**2)
        l12 = torch.sqrt((x-x1)**2+(y-y2)**2)
        l21 = torch.sqrt((x-x2)**2+(y-y1)**2)
        l22 = torch.sqrt((x-x2)**2+(y-y2)**2)
        lsum = l11+l12+l21+l22
        
        w11 = (lsum-l11)/(3*lsum)
        w12 = (lsum-l12)/(3*lsum)
        w21 = (lsum-l21)/(3*lsum)
        w22 = (lsum-l22)/(3*lsum)

        return w11, w12, w21, w22
    
    def get_gaussian_weights_2(self, x, y, x1, x2, y1, y2):
        w11 = torch.exp(-((x - x1)**2 + (y - y1)**2))
        w12 = torch.exp(-((x - x1)**2 + (y - y2)**2))
        w21 = torch.exp(-((x - x2)**2 + (y - y1)**2))
        w22 = torch.exp(-((x - x2)**2 + (y - y2)**2))
        Sumw = w11+w12+w21+w22
        
        w11_norm = w11/Sumw
        w12_norm = w12/Sumw
        w21_norm = w21/Sumw
        w22_norm = w22/Sumw

        return w11_norm, w12_norm, w21_norm, w22_norm 

    def sample_one(self, img, shiftx, shifty, weight):
        """
        Input:
            -img (N, C, H, W)
            -shiftx, shifty (N, c, H, W) #取整之后的像素运动
        """
#         print("img, shiftx, shifty, weight shape: {}, {}, {}, {}".format(img.shape, shiftx.shape, shifty.shape, weight.shape))
        N, C, H, W = img.size()

        # flatten all (all restored as Tensors)
        flat_shiftx = shiftx.view(-1)#矩阵拉开成向量，按行拉开，x方向移动
        flat_shifty = shifty.view(-1)#矩阵拉开成向量，按行拉开，y方向移动
        '''
        [0 0 0 0 ... 0]
        [1 1 1 1 ... 1]
        ...
        [H-1, H-1, ..., H-1]，然后再按行拉开成向量
        '''
        flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].cuda().long().repeat(N, C, 1, W).view(-1)
        '''
        [0 1 2 3 ... w-1]
        [0 1 2 3 ... w-1]
        ...
        [0 1 2 3 ... w-1]，然后再按行拉开成向量
        '''
        flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].cuda().long().repeat(N, C, H, 1).view(-1)
        flat_weight = weight.view(-1) #每个像素的此次前向移动的比重矩阵，按行拉开成向量
        flat_img = img.contiguous().view(-1) #图片，按行拉开成向量


        # The corresponding positions in I1
        idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).long().cuda().repeat(1, C, H, W).view(-1)
        idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).long().cuda().repeat(N, 1, H, W).view(-1)
        # ttype = flat_basex.type()
        idxx = flat_shiftx.long() + flat_basex # 绝对位置加相对偏移等于变换后的绝对位置坐标
        idxy = flat_shifty.long() + flat_basey


        # recording the inside part the shifted
        mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W) #把移动出边界的点剔除

        # Mask off points out of boundaries
        ids = (idxn*C*H*W + idxc*H*W + idxx*W + idxy)#加上向量化后n和c方向引起的坐标偏移，x方向的偏移也是整行移动的
        ids_mask = torch.masked_select(ids, mask).clone().cuda() #把outliers都剔除掉，返回一个精简后的ids向量

        img_warp = torch.zeros([N*C*H*W, ]).cuda() #准备warp结果的template
        img_warp.put_(ids_mask, torch.masked_select(flat_img*flat_weight, mask), accumulate=True)#把剔除outliers之后的pixel们逐个累加到img_warp的指定座标处

        one_warp = torch.zeros([N*C*H*W, ]).cuda() 
        one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)


        return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)
    

class build_resnet_block(nn.Module):
    """
    a resnet block which includes two general_conv2d
    """
    def __init__(self, channels, layers=2, do_batch_norm=False):
        super(build_resnet_block,self).__init__()
        self._channels = channels
        self._layers = layers

        self.res_block = nn.Sequential(*[general_conv2d(in_channels=self._channels,
                                             out_channels=self._channels,
                                             strides=1,
                                             do_batch_norm=do_batch_norm) for i in range(self._layers)])

    def forward(self,input_res):
        inputs = input_res.clone()
        input_res = self.res_block(input_res)
        return input_res + inputs

class upsample_conv2d_and_predict_flow(nn.Module):
    """
    an upsample convolution layer which includes a nearest interpolate and a general_conv2d
    """
    def __init__(self, in_channels, out_channels, ksize=3, do_batch_norm=False):
        super(upsample_conv2d_and_predict_flow, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._ksize = ksize
        self._do_batch_norm = do_batch_norm

        self.general_conv2d = general_conv2d(in_channels=self._in_channels,
                                             out_channels=self._out_channels,
                                             ksize=self._ksize,
                                             strides=1,
                                             do_batch_norm=self._do_batch_norm,
                                             padding=0)
        
        self.pad = nn.ReflectionPad2d(padding=(int((self._ksize-1)/2), int((self._ksize-1)/2),
                                        int((self._ksize-1)/2), int((self._ksize-1)/2)))#对称padding

        self.predict_flow = general_conv2d(in_channels=self._out_channels,
                                           out_channels=2,
                                           ksize=1,
                                           strides=1,
                                           padding=0,
                                           activation='tanh')

    def forward(self, conv):
        shape = conv.shape
        conv = nn.functional.interpolate(conv,size=[shape[2]*2,shape[3]*2],mode='nearest')#最近邻插值上采样
        conv = self.pad(conv)
        conv = self.general_conv2d(conv)

        flow = self.predict_flow(conv) * 256.
        
        return torch.cat([conv,flow.clone()], dim=1), flow

def general_conv2d(in_channels,out_channels, ksize=3, strides=2, padding=1, do_batch_norm=False, activation='relu'):
    """
    a general convolution layer which includes a conv2d, a relu and a batch_normalize
    """
    if activation == 'relu':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels,eps=1e-5,momentum=0.99)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.ReLU(inplace=True)
            )
    elif activation == 'tanh':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.Tanh(),
                nn.BatchNorm2d(out_channels,eps=1e-5,momentum=0.99)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.Tanh()
            )
    return conv2d

if __name__ == "__main__":
    a = upsample_conv2d_and_predict_flow(1,4)
    b = build_resnet_block(2)
    c = torch.Tensor([[1,2,3,4,5],[4,5,6,2,3],[7,8,9,4,7],[2,3,5,4,6],[4,6,7,4,5]]).reshape(1,1,5,5)
    _, out = a(c)
    out = b(out)
    print(out.shape)
    print(a)
    print(b)
