import torch
import torch.nn as nn
import torch.nn.functional as F
from base_block import *
from base_learner import *
#Stacked models based on base_learner

class CNN_MP(nn.Module):
	def __init__(self, in_channels_load, out_channels_load, bn2d_load, dropout_load, kernel_size_load, padding_load, in_channels_lift, out_channels_lift, bn2d_lift, dropout_lift, kernel_size_lift, padding_lift):
		super(CNN_MP, self).__init__()
		self.CNN_load = CNN(in_channels=in_channels_load, out_channels=out_channels_load, bn2d=bn2d_load, dropout=dropout_load, kernel_size=kernel_size_load, padding=padding_load)
		self.CNN_lift = CNN(in_channels=in_channels_lift, out_channels=out_channels_lift, bn2d=bn2d_lift, dropout=dropout_lift, kernel_size=kernel_size_lift, padding=padding_lift)

	def forward(self,x):
		print(x.shape)
		timesteps_per_phase = int(x.shape[1]/2)
		output_load, feamaps_load = self.CNN_load(x[:,0:timesteps_per_phase,:,:])
		output_lift, feamaps_lift = self.CNN_lift(x[:,timesteps_per_phase:,:,:], pooling_load=feamaps_load)

		return output_load, output_lift


class C3D_MP(nn.Module):
	def __init__(self, in_channels_load, out_channels_load, dropout_load, kernel_size_load, padding_load, stride_load, in_channels_lift, out_channels_lift, dropout_lift, kernel_size_lift, padding_lift, stride_lift):
		super(C3D_MP, self).__init__()
		self.C3D_load = Conv3D(in_channels=in_channels_load, out_channels=out_channels_load, dropout=dropout_load, kernel_size=kernel_size_load, padding=padding_load, stride=stride_load)
		self.C3D_lift = Conv3D(in_channels=in_channels_lift, out_channels=out_channels_lift, dropout=dropout_lift, kernel_size=kernel_size_lift, padding=padding_lift, stride=stride_lift)

	def forward(self, x):
		timesteps_per_phase = int(x.shape[2]/2)
		output_load, feamaps_load = self.C3D_load(x[:,:,0:timesteps_per_phase,:,:])
		output_lift, feamaps_lift = self.C3D_lift(x[:,:, timesteps_per_phase:,:,:], pooling_load=feamaps_load)

		return output_load, output_lift	

class LSTM_MP(nn.Module):
	def __init__(self, input_size_load, hidden_size_load, num_layers_load, bias, batch_first, dropout_load, input_size_lift, hidden_size_lift, num_layers_lift, dropout_lift):
		super(LSTM_MP, self).__init__()
		self.LSTM_load = LSTM(input_size=input_size_load, hidden_size=hidden_size_load, num_layers=num_layers_load, bias=bias, batch_first=batch_first, dropout=dropout_load)
		self.LSTM_lift = LSTM(input_size=input_size_lift, hidden_size=hidden_size_lift, num_layers=num_layers_lift, bias=bias, batch_first=batch_first, dropout=dropout_lift)
	
	def forward(self, x):
		timesteps_per_phase = int(x.shape[1]/2)
		output_load, feature_load = self.LSTM_load(x[:,0:timesteps_per_phase,:]) 
		output_lift, feature_lift = self.LSTM_lift(x[:,timesteps_per_phase:,:], feature_load = feature_load)

		return output_load, output_lift

class TCN_MP(nn.Module):
	def __init__(self, in_channels_load, out_channels_load, kernel_size_load, dropout_load, in_channels_lift, out_channels_lift, kernel_size_lift, dropout_lift):
		super(TCN_MP, self).__init__()
		self.TCN_load = TCN(in_channels=in_channels_load, out_channels=out_channels_load, kernel_size=kernel_size_load, dropout=dropout_load)
		self.TCN_lift = TCN(in_channels=in_channels_lift, out_channels=out_channels_lift, kernel_size=kernel_size_lift, dropout=dropout_lift)

	def forward(self, x):
		timesteps_per_phase = int(x.shape[2]/2)
		output_load, features_load = self.TCN_load(x[:,:,0:timesteps_per_phase])
		output_lift, features_lift = self.TCN_lift(x[:,:,timesteps_per_phase:], feature_load = features_load)

		return output_load, output_lift

class CNN_LSTM_MP(nn.Module): 
	def __init__(self, in_channels, out_channels, bn2d_CNN, dropout_CNN, kernel_size, padding, input_size_load, hidden_size_load, num_layers, batch_first, dropout_LSTM_load, input_size_lift, hidden_size_lift, dropout_LSTM_lift):
		super(CNN_LSTM_MP, self).__init__()
		self.CNN_LSTM_load = CNN_LSTM(in_channels=in_channels, out_channels=out_channels, bn2d_CNN=bn2d_CNN, dropout_CNN=dropout_CNN, kernel_size=kernel_size, padding=padding, input_size = input_size_load, hidden_size=hidden_size_load, num_layers=num_layers, batch_first=batch_first, dropout_LSTM=dropout_LSTM_load)
		self.CNN_LSTM_lift = CNN_LSTM(in_channels=in_channels, out_channels=out_channels, bn2d_CNN=bn2d_CNN, dropout_CNN=dropout_CNN, kernel_size=kernel_size, padding=padding, input_size = input_size_lift, hidden_size=hidden_size_lift, num_layers=num_layers, batch_first=batch_first, dropout_LSTM=dropout_LSTM_lift)

	def forward(self, x, features_load=None):
		timesteps_per_phase = int(x.shape[1]/2)
		output_load, features_load = self.CNN_LSTM_load(x[:,0:timesteps_per_phase,:,:,:], features_load=None)
		output_lift, features_lift = self.CNN_LSTM_lift(x[:,timesteps_per_phase:,:,:,:], features_load=features_load)

		return output_load, output_lift

class CNN_TCN_MP(nn.Module):
	def __init__(self, in_channels, out_channels, bn2d_CNN, dropout_CNN, kernel_size_CNN, padding, in_channels_tcn_load, out_channels_tcn_load, kernel_size_tcn_load, dropout_tcn_load, in_channels_tcn_lift, out_channels_tcn_lift, kernel_size_tcn_lift, dropout_tcn_lift):
		super(CNN_TCN_MP, self).__init__()
		self.CNN_TCN_load = CNN_TCN(in_channels=in_channels, out_channels=out_channels, bn2d_CNN=bn2d_CNN, dropout_CNN=dropout_CNN, kernel_size_CNN=kernel_size_CNN, padding=padding, in_channels_tcn=in_channels_tcn_load, out_channels_tcn=out_channels_tcn_load, kernel_size_tcn=kernel_size_tcn_load, dropout_tcn=dropout_tcn_load)
		self.CNN_TCN_lift = CNN_TCN(in_channels=in_channels, out_channels=out_channels, bn2d_CNN=bn2d_CNN, dropout_CNN=dropout_CNN, kernel_size_CNN=kernel_size_CNN, padding=padding, in_channels_tcn=in_channels_tcn_lift, out_channels_tcn=out_channels_tcn_lift, kernel_size_tcn=kernel_size_tcn_lift, dropout_tcn=dropout_tcn_lift)

	def forward(self, x, features_load=None):
		timesteps_per_phase = int(x.shape[1]/2)
		output_load, features_load = self.CNN_TCN_load(x[:,0:timesteps_per_phase,:,:,:])
		output_lift, features_lift = self.CNN_TCN_lift(x[:,timesteps_per_phase:,:,:,:], features_load=features_load)

		return output_load, output_lift
		
		
			

