function [sp,sn,time]=massfuzzy(Xp,Xn,Lp,Ln,Parameter)
%function [sp,sn,XPnoise,XNnoise,time]=fuzzy(Xp,Xn,Parameter)
% Function:  compute fuzzy membership
% Input:      
% Xp                        -  the positive samples
% Xn                        -  the  negative samples 
% Parameter         -  the parameters 
%
% Output:    
% sp                         - the fuzzy mebership vlaue for Xp
% sn                         - the fuzzy mebership vlaue for Xn
%


% if ( nargin>3||nargin<3) % check correct number of arguments
%     help Gbbftsvm
% else
tic
% Parameter settings for iForest
HeightLimit=8; % the height limit of each iTree, the subsample size = 2^HeightLimit
NumTree=100; % the number of iTrees
e=1; % 1 is average over each iTree (Arithmetic Mean)

% Calculating Dissimilarity Matrix
traindata=[Xp;Xn]
trainlabel=[Lp;Ln]
MassMatrix=meMatrix(traindata,NumTree,HeightLimit,e); 
[row,col]=size(traindata);
rowp=find(trainlabel==1);
rown=find(trainlabel==-1);
score=zeros(row,1);
sp=zeros(size(rowp,1),1);
sn=zeros(size(rown,1),1);
for i=1:size(rowp,1)
    me=sum(MassMatrix(i,rown));
    mp=sum(MassMatrix(i,rowp));
    sp(i)=me-mp;
end
for i=1:size(rown,1)
    me=sum(MassMatrix(i,rowp));
    mp=sum(MassMatrix(i,rown));
    sn(i)=me-mp;
end
time=toc;
%sp=mapminmax(sp',eps,1)'
sp=(sp-min(sp))/(max(sp)-min(sp));
sn=(sn-min(sn))/(max(sn)-min(sn));
% sn=mapminmax(sn',eps,1)'
% end

