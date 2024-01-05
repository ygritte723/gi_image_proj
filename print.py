# -*- coding: utf-8 -*-
# @Time    : 15/12/23 21:21
# @Author  : zxl
# @Site    : 
# @File    : print.py.py
# @Software: PyCharm
# Import the necessary libraries
import torch
from models.renet import RENet
from common.utils import setup_run

args = setup_run(arg_mode="train")

# Create an instance of the model
model = RENet(args)

sample = torch.rand((16, 3, 84, 84))
print('Input: ')
print(sample.shape)
sample_encoded = model.encode(sample, False)
print('After encoding (without gap): ')
print(sample_encoded.shape)
sample_encoded_ = model.encode(sample, True)
print('After encoding (with gap): ')
print(sample_encoded_.shape)

data_support, data_query = sample_encoded[:10], sample_encoded[10:]
print('After query/support splitting, shape of support: ')
print(data_support.shape)
print('After query/support splitting, shape of query: ')
print(data_query.shape)

data_support = data_support.unsqueeze(0).repeat(1, 1, 1, 1, 1)
result = model.cca_blra(data_support, data_query)
# result = model.cca(data_support, data_query)
print('Final loss metric: ')
print(result[0].shape)
print('Final loss anchor: ')
print(result[1].shape)
# Input:
# torch.Size([16, 3, 84, 84])
# After encoding (without gap):
# torch.Size([16, 640, 5, 5])
# After encoding (with gap):
# torch.Size([16, 640, 1, 1])
# After query/support splitting, shape of support:
# torch.Size([10, 640, 5, 5])
# After query/support splitting, shape of query:
# torch.Size([6, 640, 5, 5])
# Final loss metric:
# torch.Size([6, 2])
# Final loss anchor:
# torch.Size([6, 4])