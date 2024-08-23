%% Demo code for training and testing the CDFTSVM on an artifical dataset
clc
clear
cd ../

%% load train data

cd D:\MDFTSVM-master

load data/synthtr
traindata=synthtr(:,1:2);%训练集
trainlabel=synthtr(:,3)*(-2)+1;
% 
%% load test data
load data/synthte

testdata=synthte(:,1:2);
testlabel=synthte(:,3)*(-2)+1;

% Nolinear MDFTSVM
%% seting parameters

Parameter.ker = 'rbf';
Parameter.CC = 2^1;
Parameter.CR = 2^1;
Parameter.p1 = 2^(-1);
Parameter.v = 10;
Parameter.algorithm = 'CD';    
% Parameter.showplots = true;
Parameter.showplots = false;
%%
%%开始计时
tic
%% training rbf cdftsvm
[ftsvm_struct] = ftsvmtrain(traindata,trainlabel,Parameter);

%% testing rbf cdftsvm
[acc]= ftsvmclass(ftsvm_struct,testdata,testlabel);
%%
%%结束计时
toc


