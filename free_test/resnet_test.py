from torchvision import datasets, models, transforms
import torch
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet18 = models.resnet18(pretrained=True) # in (18, 34, 50, 101, 152)
print(resnet18)
resnet18 = resnet18.to(device)

hand_dir1 = '/data_ssd/hsr_dropobject/data/10_11_20:20:05/data/img/d/10.png'
hand_dir2 = '/data_ssd/hsr_dropobject/data/10_11_20:20:05/data/img/d/11.png'
hand_dir3 = '/data_ssd/hsr_dropobject/data/10_11_20:20:05/data/img/d/12.png'

starttime = time.time()

hand_im1 = Image.open(hand_dir1)
hand_im2 = Image.open(hand_dir2)
hand_im3 = Image.open(hand_dir3)

compose = transforms.Compose([transforms.Resize((224, 224))])
hand_im1 = compose(hand_im1)
hand_im2 = compose(hand_im2)
hand_im3 = compose(hand_im3)

base_hand_arr = np.array(hand_im1)
#base_hand_arr = np.repeat(base_hand_arr[...,np.newaxis], 3, -1).transpose((2, 0, 1))
hand_np2 = np.array(hand_im2)
#hand_np2 = np.repeat(hand_np2[...,np.newaxis], 3, -1).transpose((2, 0, 1))
hand_np3 = np.array(hand_im3)
#hand_np3 = np.repeat(hand_np3[...,np.newaxis], 3, -1).transpose((2, 0, 1))



base_hand_arr = np.concatenate(([base_hand_arr], [hand_np2]), axis=0)
base_hand_arr = np.concatenate((base_hand_arr, [hand_np3]), axis=0)
base_hand_arr = np.repeat(base_hand_arr[...,np.newaxis], 3, -1)

print(base_hand_arr.shape)
base_hand_arr = base_hand_arr.transpose((0, 3, 1, 2))
print(base_hand_arr.shape)

# hand_dir1 = '/data_ssd/hsr_dropobject/data/10_11_20:20:05/data/img/hand/10.png'
# hand_dir2 = '/data_ssd/hsr_dropobject/data/10_11_20:20:05/data/img/hand/11.png'
# hand_dir3 = '/data_ssd/hsr_dropobject/data/10_11_20:20:05/data/img/hand/12.png'
#
# starttime = time.time()
#
# hand_im1 = Image.open(hand_dir1)
# hand_im2 = Image.open(hand_dir2)
# hand_im3 = Image.open(hand_dir3)
#
# compose = transforms.Compose([transforms.Resize((49, 49))]) #(224, 224)
# hand_im1 = compose(hand_im1)
# hand_im2 = compose(hand_im2)
# hand_im3 = compose(hand_im3)
#
# base_hand_arr = np.array(hand_im1)
# #base_hand_arr = np.repeat(base_hand_arr[...,np.newaxis], 3, -1).transpose((2, 0, 1))
# hand_np2 = np.array(hand_im2)
# #hand_np2 = np.repeat(hand_np2[...,np.newaxis], 3, -1).transpose((2, 0, 1))
# hand_np3 = np.array(hand_im3)
# #hand_np3 = np.repeat(hand_np3[...,np.newaxis], 3, -1).transpose((2, 0, 1))
#
#
#
# base_hand_arr = np.concatenate(([base_hand_arr], [hand_np2]), axis=0)
# base_hand_arr = np.concatenate((base_hand_arr, [hand_np3]), axis=0)
#
#
# print(base_hand_arr.shape)
# base_hand_arr = base_hand_arr.transpose((0, 3, 1, 2))
# print(base_hand_arr.shape)
#
# hand_arr = torch.FloatTensor(base_hand_arr)
hand_arr = torch.FloatTensor(base_hand_arr)
#
dataloader = torch.utils.data.DataLoader(hand_arr, batch_size=3,num_workers=3)

for inputs in dataloader:
    inputs = inputs.to(device)
    output = resnet18(inputs)
    print(output)
    print(output.shape)

print(time.time() - starttime)


# n_features = 100
# resnet = models.resnet18()
# resnet.fc = nn.Linear(resnet.fc.in_features, n_features)
# torch.save(resnet, 'model.pth')
# model = torch.load('model.pth')
#
# inp = torch.randn(1, 3, 224, 224)
# out = model(inp)
# print(out.size()) -> torch.Size([1, 100])



# inputs torch.Size([4, 3, 224, 224])
# inp (224, 224, 3)