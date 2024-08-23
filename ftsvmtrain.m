function  [ftsvm_struct] = ftsvmtrain(Traindata,Trainlabel,Parameter)
% Function:  train mdftsvm
% Input:      
% Traindata         -  the train data where the feature are stored
% Trainlabel        -  the  lable of train data  
% Parameter         -  the parameters for ftsvm
%
% Output:    
% ftsvm_struct      -  ftsvm model
%


% check correct number of arguments
if ( nargin>3||nargin<3) 
    help  ftsvmtrain
end

ker=Parameter.ker;
CC=Parameter.CC;
CR=Parameter.CR;
Parameter.autoScale=0;
%Parameter.showplots=0;
autoScale=Parameter.autoScale;


st1 = cputime;%记录当前时间
%把训练标签集划分两类，Index记录分类的下标
[groupIndex, groupString] = grp2idx(Trainlabel);
%把类别标志修改为-1和1，原来是2的修改为-1，原来是1的修改为1
groupIndex = 1 - (2* (groupIndex-1));
scaleData = [];

% normalization
if autoScale
    scaleData.shift = - mean(Traindata);
    stdVals = std(Traindata);
    scaleData.scaleFactor = 1./stdVals;
    % leave zero-variance data unscaled:
    scaleData.scaleFactor(~isfinite(scaleData.scaleFactor)) = 1;
    % shift and scale columns of data matrix:
    for k = 1:size(Traindata, 2)
        scTraindata(:,k) = scaleData.scaleFactor(k) * ...
            (Traindata(:,k) +  scaleData.shift(k));
    end
else
    scTraindata= Traindata;
end


Xp=scTraindata(groupIndex==1,:);%正例样本数据
Lp=Trainlabel(groupIndex==1);%正例标签集
Xn=scTraindata(groupIndex==-1,:);%负例样本数据
Ln=Trainlabel(groupIndex==-1);%负例标签集
X=[Xp;Xn];%正例样本和负例样本叠加
L=[Lp;Ln];%正类标签和负类标签叠加
% compute fuzzy membership
[sp,sn]=massfuzzy(Xp,Xn,Lp,Ln,Parameter);
%[sp,sn]=fuzzy(Xp,Xn,Parameter);

lp=sum(groupIndex==1);%表示正类样本数量
ln=sum(groupIndex==-1);%负类样本数量
% kernel matrix
switch ker
    case 'linear'
        kfun = @linear_kernel;kfunargs ={};
    case 'quadratic'
        kfun = @quadratic_kernel;kfunargs={};
    case 'radial'
        p1=Parameter.p1;
        kfun = @rbf_kernel;kfunargs = {p1};
    case 'rbf'
        p1=Parameter.p1;
        kfun = @rbf_kernel;kfunargs = {p1};
    case 'polynomial'
        p1=Parameter.p1;
        kfun = @poly_kernel;kfunargs = {p1};
    case 'mlp'
        p1=Parameter.p1;
        p2=Parameter.p2;
        kfun = @mlp_kernel;kfunargs = {p1, p2};
end
% kernel function
switch ker
    case 'linear'
        Kpx=Xp;Knx=Xn;
    case 'rbf'
        Kpx = feval(kfun,Xp,X,kfunargs{:});%K(X+,X)
        Knx = feval(kfun,Xn,X,kfunargs{:});%K(X-,X)
end
%实际上在Kpx矩阵最后面增加一列全1 Knx最后面增加一列全1
S=[Kpx ones(lp,1)];R=[Knx ones(ln,1)];%对应文章里面的公式12下面的S+和S-

CC1=CC*sn;%对应约束条件中的C3S-
CC2=CC*sp;%对应约束条件中的C4S+ 因为设定C3=C4，所以CC代表C3/C4

fprintf('Optimising ...\n');
switch  Parameter.algorithm
    case  'CD'
        [alpha ,vp] =  L1CD(S,R,CR,CC1);
        [beta , vn] =  L1CD(R,S,CR,CC2);
        vn=-vn;
    case  'qp'
        QR=(S'*S+CR*eye(size(S'*S)))\R';
        RQR=R*QR;
        RQR=(RQR+RQR')/2;
        
        QS=(R'*R+CR*eye(size(R'*R)))\S';
        SQS=S*QS;
        SQS=(SQS+SQS')/2;

        [alpha,~,~]=qp(RQR,-ones(ln,1),[],[],zeros(ln,1),CC1,ones(ln,1));
        [beta,~,~] =qp(SQS,-ones(lp,1),[],[],zeros(lp,1),CC2,ones(lp,1));
        
        vp=-QR*alpha;
        vn=QS*beta;
    case  'QP'
        QR=(S'*S+CR*eye(size(S'*S)))\R';
        RQR=R*QR;
        RQR=(RQR+RQR')/2;
        
        QS=(R'*R+CR*eye(size(R'*R)))\S';
        SQS=S*QS;
        SQS=(SQS+SQS')/2;
        % Solve the Optimisation Problem  
        qp_opts = optimset('display','off');
        [alpha,~,~]=quadprog(RQR,-ones(ln,1),[],[],[],[],zeros(ln,1),CC1,zeros(ln,1),qp_opts);
        [beta,~,~]=quadprog(SQS,-ones(lp,1),[],[],[],[],zeros(lp,1),CC2,zeros(lp,1),qp_opts);
        
        vp=-QR*alpha;%对应公式中的u+
        vn=QS*beta;%对应公式中的u-
end
ExpendTime=cputime - st1;

ftsvm_struct.scaleData=scaleData;

ftsvm_struct.X = X;
ftsvm_struct.L = L;
ftsvm_struct.sp = sp;
ftsvm_struct.sn = sn;


ftsvm_struct.alpha = alpha;
ftsvm_struct.beta  = beta;
ftsvm_struct.vp = vp;
ftsvm_struct.vn = vn;

ftsvm_struct.KernelFunction = kfun;
ftsvm_struct.KernelFunctionArgs = kfunargs;
ftsvm_struct.Parameter = Parameter;
ftsvm_struct.groupString=groupString;
ftsvm_struct.time=ExpendTime;

% sp_center=mean(sp,1)
% sn_center=mean(sn,1)
% NXpv=find(sp>sn_center)
% 
% NXnv=find(sn>sp_center)
% 
% ftsvm_struct.NXpv=NXpv;
% ftsvm_struct.NXnv=NXnv;
% ftsvm_struct.nv=length(NXpv)+length(NXnv);
if  Parameter.showplots
    ftsvmplot(ftsvm_struct,Traindata,Trainlabel);
end   
end






