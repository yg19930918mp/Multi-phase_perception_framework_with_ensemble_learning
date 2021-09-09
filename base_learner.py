import torch
import torch.nn as nn
import torch.nn.functional as F
from base_block import *

class CNN(nn.Module):
	def __init__(self, in_channels, out_channels, bn2d, dropout, kernel_size, padding):
		super(CNN, self).__init__()
		self.CNN_part = CNN_block(in_channels=in_channels, out_channels=out_channels, bn2d=bn2d, dropout=dropout, kernel_size=kernel_size, padding=padding)
		self.fc = nn.Linear(4*4*out_channels[-1], 2)
		self.fc_mp = nn.Linear(4*4*out_channels[-1]*2, 2)
		#self.skip = nn.Conv2d(in_channels[0], out_channels[-1], kernel_size=1, stride=1, bias=False)

	def forward(self, x, pooling_load=None):
		pooling_featuremaps, CNN_featuremaps = self.CNN_part(x)
		#skip = self.skip(x)
		#CNN_featuremaps += skip
		if pooling_load == None:
			output = torch.flatten(pooling_featuremaps, 1)
			output = self.fc(output)
		else:
			pooling_features_mp = torch.cat((pooling_load, pooling_featuremaps),axis=1)
			output = torch.flatten(pooling_features_mp, 1)
			output = self.fc_mp(output)
		
		return output, pooling_featuremaps

class Conv3D(nn.Module):
	def __init__(self, in_channels, out_channels, dropout, kernel_size, padding, stride):
		super(Conv3D, self).__init__()
		self.Conv3D_part = CNN3D_block(in_channels=in_channels, out_channels=out_channels, dropout=dropout, kernel_size=kernel_size, padding=padding, stride=stride)
		self.fc = nn.Linear(2*4*4*out_channels[-1], 2)
		self.fc_mp = nn.Linear(2*4*4*out_channels[-1]*2, 2)

	def forward(self,x, pooling_load=None):
		pooling_maps, CNN_featuremaps = self.Conv3D_part(x)
		if pooling_load == None:
			pooling_flatten = torch.flatten(pooling_maps, 1)
			output = self.fc(pooling_flatten)
		else:
			pooling_features_mp = torch.cat((pooling_load, pooling_maps),axis=1)
			output = torch.flatten(pooling_features_mp, 1)
			output = self.fc_mp(output)

		return output, pooling_maps

class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, bias, batch_first, dropout):
		super(LSTM, self).__init__()
		self.lstm_part = LSTM_block(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout)
		self.fc = nn.Linear(hidden_size, 2)
		self.fc_mp = nn.Linear(hidden_size*2, 2)

	def forward(self,x, feature_load=None):
		lstm_output = self.lstm_part(x)
		if feature_load == None:
			output = self.fc(lstm_output[:,-1,:])
		else:
			feature_mp = torch.cat((feature_load[:,-1,:], lstm_output[:,-1,:]),axis=1)
			output = self.fc_mp(feature_mp)

		return output, lstm_output

class TCN(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, dropout):
		super(TCN, self).__init__()
		self.tcn = tcn_single_block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dropout=dropout)
		self.fc = nn.Linear(out_channels[-1][-1], 2)
		self.fc_mp = nn.Linear(out_channels[-1][-1]*2, 2)
		
	def forward(self, x, feature_load=None):
		tcn_output = self.tcn(x)
		if feature_load == None:
			output = self.fc(tcn_output[:,:,-1])
		else:
			feature_mp = torch.cat((feature_load[:,:,-1], tcn_output[:,:,-1]),axis=1)
			output = self.fc_mp(feature_mp)

		return output, tcn_output

class CNN_LSTM(nn.Module):
	def __init__(self, in_channels, out_channels, bn2d_CNN, dropout_CNN, kernel_size, padding, input_size, hidden_size, num_layers, batch_first, dropout_LSTM):
		super(CNN_LSTM, self).__init__()
		self.CNN_LSTM = CNN_LSTM_block(in_channels=in_channels, out_channels=out_channels, bn2d_CNN=bn2d_CNN, dropout_CNN=dropout_CNN, kernel_size=kernel_size, padding=padding, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, dropout_LSTM=dropout_LSTM)
		self.fc = nn.Linear(hidden_size, 2)
		self.fc_mp = nn.Linear(hidden_size*2, 2)
		
		
	def forward(self,x, features_load=None):
		lstm_output = self.CNN_LSTM(x)
		if features_load == None:
			output = self.fc(lstm_output[:,-1,:])
		else:
			feature_mp = torch.cat((features_load[:,-1,:], lstm_output[:,-1,:]),axis=1)
			output = self.fc_mp(feature_mp)

		return output, lstm_output

class CNN_TCN(nn.Module):
	def __init__(self, in_channels, out_channels, bn2d_CNN, dropout_CNN, kernel_size_CNN, padding, in_channels_tcn, out_channels_tcn, kernel_size_tcn, dropout_tcn, single_block=True):
		super(CNN_TCN, self).__init__()
		self.CNN_TCN = CNN_TCN_block(in_channels=in_channels, out_channels=out_channels, bn2d_CNN=bn2d_CNN, dropout_CNN=dropout_CNN, kernel_size_CNN=kernel_size_CNN, padding=padding, in_channels_tcn=in_channels_tcn, out_channels_tcn=out_channels_tcn, kernel_size_tcn=kernel_size_tcn, dropout_tcn=dropout_tcn)
		self.fc = nn.Linear(out_channels_tcn[-1][-1], 2)
		self.fc_mp = nn.Linear(out_channels_tcn[-1][-1]*2, 2)

	def forward(self,x, features_load=None):
		tcn_output = self.CNN_TCN(x)
		if features_load == None:
			output = self.fc(tcn_output[:,:,-1])
		else:
			feature_mp = torch.cat((features_load[:,:,-1], tcn_output[:,:,-1]),axis=1)
			output = self.fc_mp(feature_mp)

		return output, tcn_output

