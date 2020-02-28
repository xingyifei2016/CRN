from __future__ import division
import os,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import imageio
import matplotlib.image as mpimg
from pdb import set_trace as st
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


# {'input': <tf.Tensor 'sub:0' shape=(?, ?, ?, 3) dtype=float32>, 'conv1_1': <tf.Tensor 'Relu:0' shape=(?, ?, ?, 64) dtype=float32>, 'conv1_2': <tf.Tensor 'Relu_1:0' shape=(?, ?, ?, 64) dtype=float32>, 'pool1': <tf.Tensor 'AvgPool:0' shape=(?, ?, ?, 64) dtype=float32>, 'conv2_1': <tf.Tensor 'Relu_2:0' shape=(?, ?, ?, 128) dtype=float32>, 'conv2_2': <tf.Tensor 'Relu_3:0' shape=(?, ?, ?, 128) dtype=float32>, 'pool2': <tf.Tensor 'AvgPool_1:0' shape=(?, ?, ?, 128) dtype=float32>, 'conv3_1': <tf.Tensor 'Relu_4:0' shape=(?, ?, ?, 256) dtype=float32>, 'conv3_2': <tf.Tensor 'Relu_5:0' shape=(?, ?, ?, 256) dtype=float32>, 'conv3_3': <tf.Tensor 'Relu_6:0' shape=(?, ?, ?, 256) dtype=float32>, 'conv3_4': <tf.Tensor 'Relu_7:0' shape=(?, ?, ?, 256) dtype=float32>, 'pool3': <tf.Tensor 'AvgPool_2:0' shape=(?, ?, ?, 256) dtype=float32>, 'conv4_1': <tf.Tensor 'Relu_8:0' shape=(?, ?, ?, 512) dtype=float32>, 'conv4_2': <tf.Tensor 'Relu_9:0' shape=(?, ?, ?, 512) dtype=float32>, 'conv4_3': <tf.Tensor 'Relu_10:0' shape=(?, ?, ?, 512) dtype=float32>, 'conv4_4': <tf.Tensor 'Relu_11:0' shape=(?, ?, ?, 512) dtype=float32>, 'pool4': <tf.Tensor 'AvgPool_3:0' shape=(?, ?, ?, 512) dtype=float32>, 'conv5_1': <tf.Tensor 'Relu_12:0' shape=(?, ?, ?, 512) dtype=float32>, 'conv5_2': <tf.Tensor 'Relu_13:0' shape=(?, ?, ?, 512) dtype=float32>}
# (Pdb) vgg_fake
# {'input': <tf.Tensor 'sub_1:0' shape=(?, 512, 1024, 3) dtype=float32>, 'conv1_1': <tf.Tensor 'Relu_14:0' shape=(?, 512, 1024, 64) dtype=float32>, 'conv1_2': <tf.Tensor 'Relu_15:0' shape=(?, 512, 1024, 64) dtype=float32>, 'pool1': <tf.Tensor 'AvgPool_4:0' shape=(?, 256, 512, 64) dtype=float32>, 'conv2_1': <tf.Tensor 'Relu_16:0' shape=(?, 256, 512, 128) dtype=float32>, 'conv2_2': <tf.Tensor 'Relu_17:0' shape=(?, 256, 512, 128) dtype=float32>, 'pool2': <tf.Tensor 'AvgPool_5:0' shape=(?, 128, 256, 128) dtype=float32>, 'conv3_1': <tf.Tensor 'Relu_18:0' shape=(?, 128, 256, 256) dtype=float32>, 'conv3_2': <tf.Tensor 'Relu_19:0' shape=(?, 128, 256, 256) dtype=float32>, 'conv3_3': <tf.Tensor 'Relu_20:0' shape=(?, 128, 256, 256) dtype=float32>, 'conv3_4': <tf.Tensor 'Relu_21:0' shape=(?, 128, 256, 256) dtype=float32>, 'pool3': <tf.Tensor 'AvgPool_6:0' shape=(?, 64, 128, 256) dtype=float32>, 'conv4_1': <tf.Tensor 'Relu_22:0' shape=(?, 64, 128, 512) dtype=float32>, 'conv4_2': <tf.Tensor 'Relu_23:0' shape=(?, 64, 128, 512) dtype=float32>, 'conv4_3': <tf.Tensor 'Relu_24:0' shape=(?, 64, 128, 512) dtype=float32>, 'conv4_4': <tf.Tensor 'Relu_25:0' shape=(?, 64, 128, 512) dtype=float32>, 'pool4': <tf.Tensor 'AvgPool_7:0' shape=(?, 32, 64, 512) dtype=float32>, 'conv5_1': <tf.Tensor 'Relu_26:0' shape=(?, 32, 64, 512) dtype=float32>, 'conv5_2': <tf.Tensor 'Relu_27:0' shape=(?, 32, 64, 512) dtype=float32>}


def get_weight_bias_torch(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=torch.tensor(weights)
    weights = torch.transpose(weights, 2, 3)
    weights = torch.transpose(weights, 1, 2)
    weights = torch.transpose(weights, 0, 1)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=torch.tensor(bias)
    return torch.nn.Parameter(weights), torch.nn.Parameter(bias)

    
class VGG(nn.Module):
    
    def __init__(self):
        super(VGG, self).__init__()
        vgg_rawnet=scipy.io.loadmat('VGG_Model/imagenet-vgg-verydeep-19.mat')
        vgg_layers=vgg_rawnet['layers'][0]
        
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3), padding=1)
        self.conv1_1.weight, self.conv1_1.bias = get_weight_bias_torch(vgg_layers,0)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv1_2.weight, self.conv1_2.bias = get_weight_bias_torch(vgg_layers,2)
        
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv2_1.weight, self.conv2_1.bias = get_weight_bias_torch(vgg_layers,5)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv2_2.weight, self.conv2_2.bias = get_weight_bias_torch(vgg_layers,7)
        
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv3_1.weight, self.conv3_1.bias = get_weight_bias_torch(vgg_layers,10)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv3_2.weight, self.conv3_2.bias = get_weight_bias_torch(vgg_layers,12)
        self.conv3_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv3_3.weight, self.conv3_3.bias = get_weight_bias_torch(vgg_layers,14)
        self.conv3_4 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv3_4.weight, self.conv3_4.bias = get_weight_bias_torch(vgg_layers,16)
        
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv4_1.weight, self.conv4_1.bias = get_weight_bias_torch(vgg_layers,19)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv4_2.weight, self.conv4_2.bias = get_weight_bias_torch(vgg_layers,21)
        self.conv4_3 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv4_3.weight, self.conv4_3.bias = get_weight_bias_torch(vgg_layers,23)
        self.conv4_4 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv4_4.weight, self.conv4_4.bias = get_weight_bias_torch(vgg_layers,25)
        
        self.conv5_1 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv5_1.weight, self.conv5_1.bias = get_weight_bias_torch(vgg_layers,28)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv5_2.weight, self.conv5_2.bias = get_weight_bias_torch(vgg_layers,30)
        
        self.avgpool = nn.AvgPool2d((2, 2))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x - torch.tensor([123.6800, 116.7790, 103.9390]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x1 = self.relu(x)
        x = self.avgpool(x1)
        
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x2 = self.relu(x)
        x = self.avgpool(x2)
        
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x3 = self.relu(x)
        x = self.conv3_3(x3)
        x = self.relu(x)
        x = self.conv3_4(x)
        x = self.relu(x)
        x = self.avgpool(x)
        
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x4 = self.relu(x)
        x = self.conv4_3(x4)
        x = self.relu(x)
        x = self.conv4_4(x)
        x = self.relu(x)
        x = self.avgpool(x)
        
        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x5 = self.relu(x)
        
        return x1, x2, x3, x4, x5
    

class RecursiveGenerator(nn.Module):
    def __init__(self, sp, label_dim):
        super(RecursiveGenerator, self).__init__()
        self.sp = sp
        self.lrelu = nn.LeakyReLU(0.2)
        
#         self.num_down_sample = down_sample
        
#         #Create convolutions
#         self.convolutions = []
#         iter_down_sample = down_sample
#         prev_dim = label_dim
#         basic_dim = sp
        
#         for i in range((self.num_down_sample)):
#             basic_dim = basic_dim // 2
            
#         while self.num_down_sample >= 1:
            
#             # Only downsample once
#             if self.num_down_sample <= 2:
#                 self.convolutions.append(nn.Conv2d(prev_dim+512, 512, (3, 3), padding=1))
#                 self.convolutions.append(nn.Conv2d(512, 512, (3, 3), padding=1))
#                 self.num_down_sample -= 1
            
            
                
        
        # First layer feature
        self.conv1 = nn.Conv2d(label_dim, 1024, (3, 3), padding=1)
        self.lnorm1 = nn.LayerNorm([1024, 4, 8])
        self.conv2 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.lnorm2 = nn.LayerNorm([1024, 4, 8])
        
        # Upsampler 
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Second layer feature
        self.conv3 = nn.Conv2d(1024+label_dim, 1024, (3, 3), padding=1)
        self.lnorm3 = nn.LayerNorm([1024, 8, 16])
        self.conv4 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.lnorm4 = nn.LayerNorm([1024, 8, 16])
      
        # Third layer feature
        self.conv5 = nn.Conv2d(1024+label_dim, 1024, (3, 3), padding=1)
        self.lnorm5 = nn.LayerNorm([1024, 16, 32])
        self.conv6 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.lnorm6 = nn.LayerNorm([1024, 16, 32])
        
        # Fourth layer feature
        self.conv7 = nn.Conv2d(1024+label_dim, 1024, (3, 3), padding=1)
        self.lnorm7 = nn.LayerNorm([1024, 32, 64])
        self.conv8 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.lnorm8 = nn.LayerNorm([1024, 32, 64])
        
        # Fifth layer feature
        self.conv9 = nn.Conv2d(1024+label_dim, 1024, (3, 3), padding=1)
        self.lnorm9 = nn.LayerNorm([1024, 64, 128])
        self.conv10 = nn.Conv2d(1024, 1024, (3, 3), padding=1)
        self.lnorm10 = nn.LayerNorm([1024, 64, 128])
       
        # Sixth layer feature
        self.conv11 = nn.Conv2d(1024+label_dim, 512, (3, 3), padding=1)
        self.lnorm11 = nn.LayerNorm([512, 128, 256])
        self.conv12 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.lnorm12 = nn.LayerNorm([512, 128, 256])
        
        # Seventh layer feature
        self.conv13 = nn.Conv2d(512+label_dim, 512, (3, 3), padding=1)
        self.lnorm13 = nn.LayerNorm([512, 256, 512])
        self.conv14 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.lnorm14 = nn.LayerNorm([512, 256, 512])
        
        # Eighth layer feature
        self.conv15 = nn.Conv2d(512+label_dim, 128, (3, 3), padding=1)
        self.lnorm15 = nn.LayerNorm([128, 512, 1024])
        self.conv16 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.lnorm16 = nn.LayerNorm([128, 512, 1024])
        
        self.lastconv = nn.Conv2d(128, 1, (1, 1))
        

    
    def forward(self, x):
        down_sampled_256_512 = torch.nn.functional.interpolate(x, (self.sp//2,self.sp), mode='bilinear', align_corners=False)
        down_sampled_128_256 = torch.nn.functional.interpolate(x, (self.sp//2//2,self.sp//2), mode='bilinear', align_corners=False)
        down_sampled_64_128 = torch.nn.functional.interpolate(x, (self.sp//2//2//2,self.sp//2//2), mode='bilinear', align_corners=False)
        down_sampled_32_64 = torch.nn.functional.interpolate(x, (self.sp//2//2//2//2,self.sp//2//2//2), mode='bilinear', align_corners=False)
        down_sampled_16_32 = torch.nn.functional.interpolate(x, (self.sp//2//2//2//2//2,self.sp//2//2//2//2), mode='bilinear', align_corners=False)
        down_sampled_8_16 = torch.nn.functional.interpolate(x, (self.sp//2//2//2//2//2//2,self.sp//2//2//2//2//2), mode='bilinear', align_corners=False)
        down_sampled_4_8 = torch.nn.functional.interpolate(x, (self.sp//2//2//2//2//2//2//2,self.sp//2//2//2//2//2//2), mode='bilinear', align_corners=False)
        
        # First Layer 4*8
        base_feature_4_8 = self.lrelu(self.lnorm1(self.conv1(down_sampled_4_8)))
        base_feature_4_8 = self.lrelu(self.lnorm2(self.conv2(base_feature_4_8)))
        
        # Upsample and Concat
        upsampled_8_16 = self.up(base_feature_4_8)
        base_feature_8_16 = torch.cat((upsampled_8_16, down_sampled_8_16), 1)
        
        # Second Layer 8*16
        base_feature_8_16 = self.lrelu(self.lnorm3(self.conv3(base_feature_8_16)))
        base_feature_8_16 = self.lrelu(self.lnorm4(self.conv4(base_feature_8_16)))
        
        # Upsample and Concat
        upsampled_16_32 = self.up(base_feature_8_16)
        base_feature_16_32 = torch.cat((upsampled_16_32, down_sampled_16_32), 1)
        
        # Third Layer 16*32
        base_feature_16_32 = self.lrelu(self.lnorm5(self.conv5(base_feature_16_32)))
        base_feature_16_32 = self.lrelu(self.lnorm6(self.conv6(base_feature_16_32)))
        
        # Upsample and Concat
        upsampled_32_64 = self.up(base_feature_16_32)
        base_feature_32_64 = torch.cat((upsampled_32_64, down_sampled_32_64), 1)
        
        # Fourth Layer
        base_feature_32_64 = self.lrelu(self.lnorm7(self.conv7(base_feature_32_64)))
        base_feature_32_64 = self.lrelu(self.lnorm8(self.conv8(base_feature_32_64)))
        
        # Upsample and Concat
        upsampled_64_128 = self.up(base_feature_32_64)
        base_feature_64_128 = torch.cat((upsampled_64_128, down_sampled_64_128), 1)
        
        # Fifth Layer
        base_feature_64_128 = self.lrelu(self.lnorm9(self.conv9(base_feature_64_128)))
        base_feature_64_128 = self.lrelu(self.lnorm10(self.conv10(base_feature_64_128)))
        
        # Upsample and Concat
        upsampled_128_256 = self.up(base_feature_64_128)
        base_feature_128_256 = torch.cat((upsampled_128_256, down_sampled_128_256), 1)
        
        # Sixth Layer
        base_feature_128_256 = self.lrelu(self.lnorm11(self.conv11(base_feature_128_256)))
        base_feature_128_256 = self.lrelu(self.lnorm12(self.conv12(base_feature_128_256)))
        
        # Upsample and Concat
        upsampled_256_512 = self.up(base_feature_128_256)
        base_feature_256_512 = torch.cat((upsampled_256_512, down_sampled_256_512), 1)
        
        # Seventh Layer
        base_feature_256_512 = self.lrelu(self.lnorm13(self.conv13(base_feature_256_512)))
        base_feature_256_512 = self.lrelu(self.lnorm14(self.conv14(base_feature_256_512)))
        
        # Upsample and Concat
        upsampled_512_1024 = self.up(base_feature_256_512)
        base_feature_512_1024 = torch.cat((upsampled_512_1024, x), 1)
        
        # Eighth Layer
        base_feature_512_1024 = self.lrelu(self.lnorm15(self.conv15(base_feature_512_1024)))
        base_feature_512_1024 = self.lrelu(self.lnorm16(self.conv16(base_feature_512_1024)))
        
        base_feature_512_1024 = self.lastconv(base_feature_512_1024)
        base_feature_512_1024 = (base_feature_512_1024+1)/2.0*255.
        
        return base_feature_512_1024
       
        
        

def compute_error(real,fake,label):
    #return tf.reduce_sum(tf.reduce_mean(label*tf.expand_dims(tf.reduce_mean(tf.abs(fake-real),reduction_indices=[3]),-1),reduction_indices=[1,2]))#diversity loss
    return tf.reduce_mean(tf.abs(fake-real))#simple loss

# #os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
# #os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))#select a GPU with maximum available memory
# #os.system('rm tmp')
# sess=tf.Session()
# is_training=False
# sp=512#spatial resolution: 512x1024
# with tf.variable_scope(tf.get_variable_scope()):
#     label=tf.placeholder(tf.float32,[None,None,None,20])
#     real_image=tf.placeholder(tf.float32,[None,None,None,3])
#     fake_image=tf.placeholder(tf.float32,[None,None,None,3])
#     generator=recursive_generator(label,sp)
#     weight=tf.placeholder(tf.float32)
#     vgg_real=build_vgg19(real_image)
#     vgg_fake=build_vgg19(generator,reuse=True)
#     p0=compute_error(vgg_real['input'],vgg_fake['input'],label)
#     p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'],label)/2.6
#     p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'],tf.image.resize_area(label,(sp//2,sp)))/4.8
#     p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'],tf.image.resize_area(label,(sp//4,sp//2)))/3.7
#     p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'],tf.image.resize_area(label,(sp//8,sp//4)))/5.6
#     p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'],tf.image.resize_area(label,(sp//16,sp//8)))*10/1.5
#     G_loss=p0+p1+p2+p3+p4+p5
# lr=tf.placeholder(tf.float32)
# G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in tf.trainable_variables()])
# sess.run(tf.global_variables_initializer())
# # from pdb import set_trace as st
# # st()
# ckpt=tf.train.get_checkpoint_state("result_512p")
# if ckpt:
#     print('loaded '+ckpt.model_checkpoint_path)
#     saver=tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
#     saver.restore(sess,ckpt.model_checkpoint_path)
# else:
#     ckpt_prev=tf.train.get_checkpoint_state("result_256p")
#     saver=tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_') and not var.name.startswith('g_512')])
#     print('loaded '+ckpt_prev.model_checkpoint_path)
#     saver.restore(sess,ckpt_prev.model_checkpoint_path)
# saver=tf.train.Saver(max_to_keep=1000)

use_cuda = torch.cuda.is_available()
torch.manual_seed(4222222)
device = torch.device("cuda" if use_cuda else "cpu")
# kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
model1=RecursiveGenerator(512, 20).cuda()
model2=VGG().cuda()

for epoch in range(1, 30 + 1):
    train_acc = 0
    train_loss = 0
#     for it,(local_batch, local_labels) in enumerate(train_loader):
    label = torch.ones((30, 20, 512, 1024)).cuda()
    images = torch.ones((30, 3, 512, 1024)).cuda()
    fake = model1(label)
    fake1, fake2, fake3, fake4, fake5 = VGG(fake)
    real1, real2, real3, real4, real5 = VGG(images)
    st()
    
#         batch = torch.tensor(local_batch, requires_grad=True).cuda()
#         labels = local_labels.cuda()
#         optimizer.zero_grad()
#         out = model(batch)
#         _, predicted = torch.max(out, 1)
#         total = labels.shape[0]
#         train_acc += (predicted == labels).sum().item()
#         criterion = nn.CrossEntropyLoss()
#         loss = criterion(out, labels)
#         train_loss += loss
#         loss.backward()
#         optimizer.step()

g_loss=np.zeros(3000,dtype=float)
input_images=[None]*3000
label_images=[None]*3000
for epoch in range(1,21):
    if os.path.isdir("result_512p/%04d"%epoch):
        continue
    cnt=0
#     for ind in np.random.permutation(2975)+1:
    for ind in [100027, 100167, 100338, 100343, 100437, 100499]:
        st=time.time()
#         cnt+=1
#         if input_images[ind] is None:
        label_images[cnt]=helper.get_semantic_map("data/cityscapes/Label512Full/%08d.png"%ind)#training label
        st()
        print(np.concatenate((label_images[ind],np.expand_dims(1-np.sum(label_images[ind],axis=3),axis=3)),axis=3))
        input_images[cnt]=np.expand_dims(np.float32(imageio.imread("data/cityscapes/RGB512Full_vivid/%08d.png"%ind)), \
                                         axis=0)#training image with vivid appearance. see "optional_preprocessing"
        cnt+=1
        st()    
        _,G_current,l0,l1,l2,l3,l4,l5=sess.run([G_opt,G_loss,p0,p1,p2,p3,p4,p5],feed_dict={label:np.concatenate((label_images[ind],np.expand_dims(1-np.sum(label_images[ind],axis=3),axis=3)),axis=3),real_image:input_images[ind],lr:1e-4})
#         g_loss[ind]=G_current
#         print("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),np.mean(l0),np.mean(l1),np.mean(l2),np.mean(l3),np.mean(l4),np.mean(l5),time.time()-st))
#     os.makedirs("result_512p/%04d"%epoch)
#     target=open("result_512p/%04d/score.txt"%epoch,'w')
#     target.write("%f"%np.mean(g_loss[np.where(g_loss)]))
#     target.close()
#     saver.save(sess,"result_512p/model.ckpt")
#     if epoch%20==0:
#         saver.save(sess,"result_512p/%04d/model.ckpt"%epoch)
#     for ind in range(100001,100051):
#         if not os.path.isfile("data/cityscapes/Label512Full/%08d.png"%ind):#test label
#             continue            
#         semantic=helper.get_semantic_map("data/cityscapes/Label512Full/%08d.png"%ind)#test label
#         output=sess.run(generator,feed_dict={label:np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3)})
#         output=np.minimum(np.maximum(output,0.0),255.0)


#         scipy.misc.toimage(output[0,:,:,:],cmin=0,cmax=255).save("result_512p/%04d/%06d_output.jpg"%(epoch,ind))

# if not os.path.isdir("result_512p/final"):
#     os.makedirs("result_512p/final")
# for ind in range(100001,100501):
#     if not os.path.isfile("data/cityscapes/Label512Full/%08d.png"%ind):#test label
#         continue    
#     semantic=helper.get_semantic_map("data/cityscapes/Label512Full/%08d.png"%ind)#test label
#     output=sess.run(generator,feed_dict={label:np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3)})
#     output=np.minimum(np.maximum(output,0.0),255.0)
#     from pdb import set_trace as st
#     mpimg.imsave("result_512p/final/%06d_output.jpg"%ind, output[0,:,:,:] / 255.)
