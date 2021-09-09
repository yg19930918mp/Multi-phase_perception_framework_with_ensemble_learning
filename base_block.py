import torch
import torch.nn as nn
import torch.nn.functional as F
from tcn import TemporalConvNet, TemporalConvNet_Single
import numpy as np

class BasicConv2d(nn.Module):
	def __init__(self,in_channels, out_channels, bn2d, dropout, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv2d = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
		if bn2d:
			self.bn2d = nn.BatchNorm2d(out_channels)
		else:
			self.bn2d = False
		if dropout:
			self.dropout=nn.Dropout(dropout)
		else:
			self.dropout = False
		self.relu = nn.ReLU()

	def forward(self,x):
		x = self.conv2d(x)
		if self.bn2d:
			x = self.bn2d(x)
		if self.dropout:
			x=self.dropout(x)
		x = self.relu(x)
		
		return x

class CNN_block(nn.Module):
	def __init__(self, in_channels, out_channels, bn2d, dropout, kernel_size, padding):
		super(CNN_block, self).__init__()
		self.layer_num = len(in_channels)
		self.layers = nn.ModuleList()
		for i in range(self.layer_num):
			self.layers.append(BasicConv2d(in_channels=in_channels[i], out_channels=out_channels[i], bn2d=bn2d[i], dropout=dropout[i], kernel_size=kernel_size[i], padding=padding[i]))		
		self.pooling = nn.AdaptiveMaxPool2d((4,4))



	def forward(self, x):

		for i in range(self.layer_num):
			x = self.layers[i](x)
		output = self.pooling(x)
		return output, x

class CNN_block1(nn.Module):
	def __init__(self, in_channels, out_channels, bn2d, dropout, kernel_size, padding):
		super(CNN_block1, self).__init__()
		self.layer_num = len(in_channels)
		self.layers = nn.ModuleList()
		for i in range(self.layer_num):
			self.layers.append(BasicConv2d(in_channels=in_channels[i], out_channels=out_channels[i], bn2d=bn2d[i], dropout=dropout[i], kernel_size=kernel_size[i], padding=padding[i]))		
		self.pooling = nn.AdaptiveMaxPool2d((4,4))

	def forward(self, x):
		for i in range(self.layer_num):
			x = self.layers[i](x)
		output = self.pooling(x)
		return output, x

class BasicConv3d(nn.Module):
	def __init__(self, in_channels, out_channels, dropout, **kwargs):
		super(BasicConv3d, self).__init__()
		self.conv3d = nn.Conv3d(in_channels, out_channels, **kwargs)
		self.relu = nn.ReLU()
		if dropout:
			self.dropout=nn.Dropout3d(dropout)
		else:
			self.dropout = False
	
	def forward(self,x):
		x = self.conv3d(x)
		if self.dropout:
			x=self.dropout(x)
		x = self.relu(x)
		
		return x

class CNN3D_block(nn.Module):
	def __init__(self, in_channels, out_channels, dropout, kernel_size, padding, stride):
		super(CNN3D_block, self).__init__()
		self.layer_num = len(in_channels)
		self.layers = nn.ModuleList()
		for i in range(self.layer_num):
			self.layers.append(BasicConv3d(in_channels=in_channels[i], out_channels=out_channels[i], dropout = dropout[i], kernel_size=kernel_size[i], padding=padding[i], stride=stride[i]))
		self.pooling = nn.AdaptiveAvgPool3d((2,4,4))

	def forward(self, x):
		for i in range(self.layer_num):
			x = self.layers[i](x)
		output = self.pooling(x)
		
		return output, x

class LSTM_block(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, bias, batch_first, dropout):
		super(LSTM_block, self).__init__()
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout)


	def forward(self,x):
		output, (hn, cn) = self.lstm(x) 
		
		return output	

class tcn_single_block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, dropout):
		super(tcn_single_block, self).__init__()
		self.tcn = TemporalConvNet_Single(in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, dropout = dropout)

	def forward(self, x):
		output = self.tcn(x) 
		
		return output

class CNN_LSTM_block(nn.Module):
	def __init__(self, in_channels, out_channels, bn2d_CNN, dropout_CNN, kernel_size, padding, input_size, hidden_size, num_layers, batch_first, dropout_LSTM):
		super(CNN_LSTM_block, self).__init__()
		self.CNN_block = CNN_block1(in_channels=in_channels, out_channels=out_channels, bn2d=bn2d_CNN, dropout=dropout_CNN, kernel_size=kernel_size, padding=padding)
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout_LSTM)
		self.fc = nn.Linear(4*4*out_channels[-1], 64)
		self.relu = nn.ReLU()

	def forward(self, x):
		feature_dic = {}
		feature_list=[]
		for timestep in range(len(x[0])):
			feature_dic["timestep_"+str(timestep)], feature_maps = self.CNN_block(x[:,timestep,:,:,:])
			feature_dic["timestep_"+str(timestep)] = self.relu(self.fc(torch.flatten(feature_dic["timestep_"+str(timestep)],1)))
			feature_list.append(feature_dic["timestep_"+str(timestep)])
			timestep += 1				
		lstm_input = torch.stack(feature_list, axis=1)
		lstm_output, (h_n, c_n) = self.lstm(lstm_input)
		
		return lstm_output

class CNN_TCN_block(nn.Module):
	def __init__(self, in_channels, out_channels, bn2d_CNN, dropout_CNN, kernel_size_CNN, padding,in_channels_tcn, out_channels_tcn, kernel_size_tcn, dropout_tcn, single_block=True):
		super(CNN_TCN_block, self).__init__()
		self.CNN_block = CNN_block(in_channels=in_channels, out_channels=out_channels, bn2d=bn2d_CNN, dropout=dropout_CNN, kernel_size=kernel_size_CNN, padding=padding)
		if single_block:
			self.tcn = tcn_single_block(in_channels=in_channels_tcn, out_channels=out_channels_tcn, kernel_size = kernel_size_tcn, dropout = dropout_tcn)
		else:
			self.tcn=tcn_block(in_channels=in_channels_tcn, out_channels=out_channels_tcn, kernel_size = kernel_size_tcn, dropout = dropout_tcn)
		self.fc = nn.Linear(4*4*out_channels[-1], 64)
		self.relu = nn.ReLU()

	def forward(self, x):
		feature_dic = {}
		feature_list = []
		for timestep in range(len(x[0])):
			feature_dic["timestep_"+str(timestep)], feature_maps = self.CNN_block(x[:,timestep,:,:,:])
			feature_dic["timestep_"+str(timestep)] = self.relu(self.fc(torch.flatten(feature_dic["timestep_"+str(timestep)],1)))
			feature_list.append(feature_dic["timestep_"+str(timestep)])
			timestep += 1
			tcn_input = torch.stack(feature_list, axis=2)
			tcn_output = self.tcn(tcn_input)
		
		return tcn_output
	


