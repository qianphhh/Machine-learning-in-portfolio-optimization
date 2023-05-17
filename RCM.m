%% main2.m
% 数值模拟数据生成1
% =====模型设定======
% 随机波动率因子模型：dX_{it}=mu_i d_t + rho_i sigma_{it} dB_{it} + sqrt(1-(rho_i)^2)sigma_{it} dW_t + v_i dZ_t i=1,2,...,p
% 其中: X_{it}为对数价格, B,W,Z均为独立的标准布朗运动
% 瞬时波动率(spot volatility)服从独立OU过程: dVrho_{it}=alpha_i (beta_{i0}-Vrho_{it}) dt + beta_{i1} dU_{it}
% 其中 Vrho_{it}=log(sigma_{it}), U_{it}为独立的布朗运动
% =====参数设置和选取=====
% 每日N=60*4*60=14400个等长度区间，即每隔1/N时间产生一个有效对数价格
% p只股票,生成初始值
p=300;
X0=100*ones(100,1);
% 模拟50天的数据, 共生成60*242=14520条有效对数价格
N=60*242;
days=50;
deltaT=1/(N*days);% 时间增量
% 参数设置
x=rand(p,5)*0.6+0.7; % 生成每只证券固定参数的x
mu=x(:,1).*0.03;
beta0=-x(:,2);
beta1=0.75.*x(:,3);
alpha=-(1/40).*x(:,4);
rho=-0.7;
v=exp(beta0);
% 高斯噪声
% w=0;
% w=0.001;
w=0.001;
gamma=0.01;
% epsilon=randn(p,days*N)*w;
% 对数价格过程和对数瞬时波动率过程
Vrho0=0; % 初始对数波动率为 0
X0=100.*ones(p,1); % 初始对数价格
Vrho=[];
for i=1:N*days
% 生成独立标准布朗运动 B W Z 的维纳增量
B=randn(p,1);
W=randn(1,1);
Z=randn(1,1);
U=randn(p,1);
    if i==1
  deltaVrho=alpha.*(beta0-Vrho0).*deltaT+beta1.*U.*sqrt(deltaT);
  Vrho(:,i)=Vrho0+deltaVrho;
  deltaX=mu.*deltaT+rho.*exp(Vrho(:,i)).*B.*sqrt(deltaT)+sqrt(1-rho^2).*exp(Vrho(:,i)).*W.*sqrt(deltaT)+v.*Z.*sqrt(deltaT);
  X(:,i)=X0+deltaX;
    else
  deltaVrho=alpha.*(beta0-Vrho(:,i-1)).*deltaT+beta1.*U.*sqrt(deltaT);
  Vrho(:,i)=Vrho(:,i-1)+deltaVrho;
  deltaX=mu.*deltaT+rho.*exp(Vrho(:,i)).*B.*sqrt(deltaT)+sqrt(1-rho^2).*Vrho(:,i).*W.*sqrt(deltaT)+v.*Z.*sqrt(deltaT);
  X(:,i)=X(:,i-1)+deltaX;
    end
end
X_NoNoise=X; % 无噪声价格序列
 X=X+rand(p,days*N).*(gamma^2*sqrt(1/(N*days)*sum(exp(Vrho).^4,2)));
% X=X+epsilon; % 有噪声价格序列
marker=[1:60:60*242*days]; % 模拟一分钟高频数据
marker2=[1:15:60*242*days]; % 模拟15秒高频数据

% 将数据处理成一分钟高频和15秒数据的方法
simudata=X(:,marker);
simudata_15s=X(:,marker2);
Noisefree_simudata=X_NoNoise(:,marker);
Noisefree_simudata_15s=X_NoNoise(:,marker2);

for i=1:days
%     r=X_NoNoise(:,(i-1)*242*60+2:242*60*i)-X_NoNoise(:,(i-1)*242*60+1:242*60*i-1);
%     r=Noisefree_simudata_15s(:,(i-1)*968+2:i*968)-Noisefree_simudata_15s(:,(i-1)*968+1:i*968-1);
r=Noisefree_simudata(:,(i-1)*242+2:242*i)-Noisefree_simudata(:,(i-1)*242+1:242*i-1);
    real_cov(:,:,i)=r*r';
end
save('A:\临时MATLAB任务_高维投资组合\数据\模拟数据_300stocks.mat','simudata','simudata_15s','Noisefree_simudata','Noisefree_simudata_15s','real_cov');
% save('A:\临时MATLAB任务_高维投资组合\数据\模拟数据_300stocks_highNoise.mat','simudata','simudata_15s','Noisefree_simudata','Noisefree_simudata_15s','real_cov')

% 模拟数据生成2
clear
% 随机波动率因子模型：dX_{it}=mu_i d_t + rho_i sigma_{it} dB_{it} + sqrt(1-(rho_i)^2)sigma_{it} dW_t + v_i dZ_t i=1,2,...,p
% 其中: X_{it}为对数价格, B,W,Z均为独立的标准布朗运动

% 瞬时波动率(spot volatility)服从独立OU过程: dVrho_{it}=alpha_i (beta_{i0}-Vrho_{it}) dt + beta_{i1} dU_{it}
% 其中 Vrho_{it}=log(sigma_{it}), U_{it}为独立的布朗运动

% =====参数设置和选取=====
% 每日N=60*4*60=14400个等长度区间，即每隔1/N时间产生一个有效对数价格
% p只股票,生成初始值
p=300;
Y0=100*ones(100,1);
% 模拟50天的数据, 共生成60*242=14520条有效对数价格
N=60*242;
days=50;
deltaT=1/(N*days);% 时间增量
% 参数设置
x=rand(p,5)*0.6+0.7; % 生成每只证券固定参数的x
b=[0.005 0.003 0.001];
beta0=-x(:,2);
beta1=0.75.*x(:,3);
alpha=-(1/40).*x(:,4);
rho=-0.7;
v=exp(beta0);
% 高斯噪声
% w=0;
% w=0.001;
w=0.001;
gamma=0.01;
% 对角分块
left=p;
block=[];
for i=1:p
    block(i)=randi([1,50],1);
    left=left-block(i);
    if left==0
        break
    else if left<0
           block(i)=block(i)+left;
         break
        end
    end
end
block=cumsum(block);block=[0 block];
Residuals=zeros(p,p);
Residuals(logical(diag(ones(p,1))))=0.05+rand(p,1)*(0.09-0.05);
for i=1:size(block,2)-1 % 生成分块内元素
    size=block(i+1)-block(i);
    corr=0.15+rand(size,size)*(0.6-0.15);% 相关系数
    corr=(corr+corr')/2;
    corr(logical(diag(ones(size,1))))=1; % 相关系数
    Residuals(block(i)+1:block(i+1),block(i)+1:block(i+1))=...
        diag(diag(Residuals(block(i)+1:block(i+1),block(i)+1:block(i+1))))*corr*...
        diag(diag(Residuals(block(i)+1:block(i+1),block(i)+1:block(i+1))));
end
  Residuals=Transform(Residuals,1e-10);
[V,D]=eig(Residuals); %特征分解 Residuals=V*D*V';
Compom=sqrt(D)*V'; % Residuals=Compom'*Compom

% 对数价格过程和对数瞬时波动率过程
Vrho0=0; % 初始对数波动率为 0
Y0=100.*ones(p,1); % 初始对数价格
Vrho=[];
for i=1:N*days 
    % 生成独立标准布朗运动 B W Z 的维纳增量
    B=randn(p,1);
    W=randn(1,1);
    Z=randn(1,1);
    U=randn(p,1);
    % 模拟对角分块
    if i==1
        deltaVrho=alpha.*(beta0-Vrho0).*deltaT+beta1.*U.*sqrt(deltaT);
        Vrho(:,i)=Vrho0+deltaVrho;
        %       deltaZ=
        deltaY=b*deltaX+deltaZ;
        deltaY=mu.*deltaT+rho.*exp(Vrho(:,i)).*B.*sqrt(deltaT)+sqrt(1-rho^2).*exp(Vrho(:,i)).*W.*sqrt(deltaT)+v.*Z.*sqrt(deltaT);
        X(:,i)=Y0+deltaY;
    else
        deltaVrho=alpha.*(beta0-Vrho(:,i-1)).*deltaT+beta1.*U.*sqrt(deltaT);
        Vrho(:,i)=Vrho(:,i-1)+deltaVrho;
        deltaY=mu.*deltaT+rho.*exp(Vrho(:,i)).*B.*sqrt(deltaT)+sqrt(1-rho^2).*Vrho(:,i).*W.*sqrt(deltaT)+v.*Z.*sqrt(deltaT);
        X(:,i)=X(:,i-1)+deltaY;
    end
end
X_NoNoise=X; % 无噪声价格序列
X=X+rand(p,days*N).*(gamma^2*sqrt(1/(N*days)*sum(exp(Vrho).^4,2)));
% X=X+epsilon; % 有噪声价格序列
marker=[1:60:60*242*days]; % 模拟一分钟高频数据
marker2=[1:15:60*242*days]; % 模拟15秒高频数据
% 将数据处理成一分钟高频和15秒数据的方法
simudata=X(:,marker);
simudata_15s=X(:,marker2);
Noisefree_simudata=X_NoNoise(:,marker);
Noisefree_simudata_15s=X_NoNoise(:,marker2);
for i=1:days
%     r=X_NoNoise(:,(i-1)*242*60+2:242*60*i)-X_NoNoise(:,(i-1)*242*60+1:242*60*i-1);
%     r=Noisefree_simudata_15s(:,(i-1)*968+2:i*968)-Noisefree_simudata_15s(:,(i-1)*968+1:i*968-1);
r=Noisefree_simudata(:,(i-1)*242+2:242*i)-Noisefree_simudata(:,(i-1)*242+1:242*i-1);
    real_cov(:,:,i)=r*r';
end
% save('A:\临时MATLAB任务_高维投资组合\数据\模拟数据_300stocks.mat','simudata','simudata_15s','Noisefree_simudata','Noisefree_simudata_15s','real_cov');
% save('A:\临时MATLAB任务_高维投资组合\数据\模拟数据_300stocks_highNoise.mat','simudata','simudata_15s','Noisefree_simudata','Noisefree_simudata_15s','real_cov')


% 补充non-linear和linear两个结果
% 从R处理好的数据中提取NonLinear-poet和Linear-I的结果
% di=dir('A:\临时MATLAB任务_高维投资组合\数据\处理完的数据\R处理完的数据_无噪声');
di=dir('A:\临时MATLAB任务_高维投资组合\数据\处理完的数据\R处理完的数据_高噪');
di(1:2,:)=[];
% linear-i部分
Linear_I=zeros(300,300,50);
for i=1:50%size(di,1)
    name=di(i).name;
 Linear(:,:,str2num(name(regexp(name,'\d'))))=load([di(i).folder,'\',di(i).name]);
end

% nonlinear部分
nonLinear=zeros(300,300,50);
for i=51:size(di,1)
    name=di(i).name;
 nonLinear(:,:,str2num(name(regexp(name,'\d'))))=load([di(i).folder,'\',di(i).name]);
end
% save('A:\临时MATLAB任务_高维投资组合\数据\处理完的数据\R_two_estimator_NoNoise.mat','nonLinear','Linear')
% save('A:\临时MATLAB任务_高维投资组合\数据\处理完的数据\R_two_estimator.mat','nonLinear','Linear')
save('A:\临时MATLAB任务_高维投资组合\数据\处理完的数据\R_two_estimator_highNoise.mat','nonLinear','Linear')


% 数值模拟部分结果，收益和RMSE
% % =====表：不同估计量在模拟情形下进行组合投资的结果比较=====
tic
% load('4_Simu_Six_estimatiors_Noisy.mat')
load('5_find_optimal_POET_Parats.mat')
% load('4_Simu_Six_estimatiors_NoNoise.mat')

% 得到全部的字段名
fnames=fieldnames(RV,'-full');

% 得到日收益率矩阵 r - 证券数x天数(从旧到新)
r=getfield(RV,fnames{end})';
% r=[r.data]';

% 得到样本的日期
% date=RV.dailyre(1).times;

% 证券数和样本天数
num=size(r,1);
days=size(r,2);
pr_result_temp={};
gr_result_temp={};
fnames(end)=[];
fnames{end}='erc';
real_cov=RV(1).real_cov(:,:,2:end);
RV(1).real_cov=real_cov;

for i=1:size(fnames,1) % i为要进行组合优化的矩阵类型个数 -2是因为times和dailyre并非矩阵
     if strcmp(fnames{i},'erc')
     matrix=ones(num,num,days-2);
     else
     matrix=getfield(RV(1),fnames{i});
     end
 % parfor timer
N = size(matrix,3);
parfor_progress(N); % Initialize

x=[];
r_matrix=[];
parfor j=1:size(matrix,3)-1 % t为样本的日期数
%>>>>>>>>>>>>>基于RCOV的策略<<<<<<<<<<<<<<第一个v为稀疏估计矩阵，第二个为正常已实现协方差矩阵
v=matrix(:,:,j);
options=optimoptions(@quadprog,'Display','off');
x(:,j)=quadprog(v,[],[],[],ones(size(v,1),1)',1,zeros(size(v,1),1),ones(size(v,1),1),[],options); % mv 有非卖空约束
% x(:,j)=quadprog(v,[],[],[],ones(size(v,1),1)',1,[],[],[],options); % mv 无非卖空约束
% x(:,j)=inv(v)*ones(size(v,1),1)/(ones(1,size(v,1))*inv(v)*ones(size(v,1),1));% 无非卖空约束的等价形式
% x(:,j)=x(:,j)./sum(x(:,j));
r_matrix(j)=x(:,j)'*r(:,j+1);
F_dis(i,j)=sqrt(sum(sum((v-real_cov(:,:,j)).^2,1),2)); % 与真实矩阵之间的Fronbenius距离
% count=count+1*(r_n>=r_rcov && (r_n>=r_p));

% parfor timer
pause(rand); % Replace with real code
parfor_progress; % Count
end
 parfor_progress(0);
 % 将每日的收益保存为结构数组 pr-portfolio return  第二行为相应的最优权重
pr_result(1).(fnames{i})=r_matrix;
pr_result(2).(fnames{i})=x;
% pr_result_temp{i,:}=[fnames{i} r_matrix x];

% 将总收益保存为结构数组 gr-gross return
% gr_result(1).(fnames{i})=exp(sum(log(1+r_matrix)))-1;
gr_result(2).(fnames{i})=mean(r_matrix);

gr_result(3).(fnames{i})=max(r_matrix);
gr_result(4).(fnames{i})=min(r_matrix);
gr_result(5).(fnames{i})=mean(F_dis(i,:));
gr_result(6).(fnames{i})=std(F_dis(i,:));
gr_result(7).(fnames{i})=std(r_matrix);
% gr_result(6).(fnames{i})=getfield(RV(2),fnames{i});
end

% =====图：不同估计量在模拟情形下进行组合投资的结果比较=====
% 实证部分结果
% =========得到所有股票的不同行业分类=========
load('1分钟收益_293只股票_2018年.mat')
w=windmatlab;
for i=1:size(dailydata,2)
    stkid=dailydata(i).stockid;
[w_wsd_data,w_wsd_codes,w_wsd_fields,w_wsd_times,w_wsd_errorid,w_wsd_reqid]=... % 获取行业名称和代码 分别为：申万 中信 中证 证监会
    w.wsd(stkid,'industry_sw,industry_citic,industry_csi,industry_CSRC12','2019-01-01','2019-01-02','industryType=1');
Ind(i).SW=w_wsd_data(1);
Ind(i).ZX=w_wsd_data(2);
Ind(i).ZZ=w_wsd_data(3);
Ind(i).ZJH=w_wsd_data(4);
end
for i=1:size(Ind,2)
    if i==1
    Ind(i).SW_code=1;
    Ind(i).ZX_code=1;
    Ind(i).ZZ_code=1;
    Ind(i).ZJH_code=1;
    else
        for j=1:i-1
            if Ind(i).SW_code==0
    Ind(i).SW_code=Ind(j).SW_code*(strcmp(Ind(i).SW,Ind(j).SW));
            end
            if Ind(i).ZX_code==0
    Ind(i).ZX_code=Ind(j).SW_code*(strcmp(Ind(i).ZX,Ind(j).ZX));
            end
            if Ind(i).ZZ_code==0
    Ind(i).ZZ_code=Ind(j).SW_code*(strcmp(Ind(i).ZZ,Ind(j).ZZ));
            end
            if Ind(i).ZJH_code==0
    Ind(i).ZJH_code=Ind(j).SW_code*(strcmp(Ind(i).ZJH,Ind(j).ZJH));
            end
        end
        if Ind(i).SW_code==0
    Ind(i).SW_code=i;
        end
            if Ind(i).ZX_code==0
    Ind(i).ZX_code=i;
            end
            if Ind(i).ZZ_code==0
    Ind(i).ZZ_code=i;
            end
            if Ind(i).ZJH_code==0
    Ind(i).ZJH_code=i;
            end
    end
    Ind(i).Stkid=dailydata(i).stockid;
end
Sortby_Ind=Ind;
save('293只股票的行业分类.mat','Sortby_Ind')

% =========数据清洗：删去在一定时间内无交易价格变化的股票, 绘制"正定矩阵比率与删除标准之间的曲线"
load('1分钟收益_293只股票_2018年.mat')
days=size(times,1)/242;% 交易日数
temp=[hfdata.data];
% 1分钟收益率
hf_return=temp(:,1:4:end);% 从旧到新
% 交易量
hf_vol=temp(:,2:4:end);
% 区间收盘价
hf_price=temp(:,4:4:end);

% 在293只证券中检查是否存在超过多少个区间无交易 即收益率为0
delete_data=struct;
num_missing=[200:242];
parfor ii=1:size(num_missing,2)
    delete=[];
    count=0;
for i=1:size(hf_return,2)
    for j=1:size(hf_return,1)
count=(hf_return(j,i)==0)*(count+1);
if count>=num_missing(ii)
    delete=[delete;i j];
    count=0;
    break
end
    end
end
delete_data(ii).delete=delete;

end

% 得到不同删除标准对应的1分钟半已实现协方差的符合条件比率

for i=1:size(delete_data,2)%size(delete_data,2)
    temp=hf_return;
    delete=delete_data(i).delete;
    temp(:,delete(:,1)')=[];
    i
    count_indef=0; % 非正定的个数
    for j=1:243 % size(hfdata,1)
temp_p_raw=log(1+temp((j-1)*242+1:j*242,:));
r_raw=temp_p_raw(2:end,:)-temp_p_raw(1:end-1,:);% 将1分钟高频价格数据处理成收益率
r_raw(r_raw<-1)=0;r_raw(r_raw>1)=0;
r0n=(r_raw-abs(r_raw))/2;
r0p=(r_raw+abs(r_raw))/2;

if any(sum(abs(r0n'*r0n),1)<=0.001) || any(sum(abs(r0p'*r0p),1)<=0.001)
    count_indef=count_indef+1;
end
    end
delete_data(i).num_indef=count_indef;
end

% 画图1：剔除最大连续无交易股票
set(gcf,'color','white')
plot(num_missing,[delete_data.num_indef],'--k')
hold on
scatter(num_missing,[delete_data.num_indef],'ok','filled')
box off
ylabel('存在全零列或行的交易日数','FontSize',15)
xlabel('无交易价格变化的最大时间','FontSize',15)
set(gca,'XTick',[200 225 236 242])
set(gca,'XTickLabel',{'200','225','236','242'})
grid on
set(gca,'YTick',[0 1 2 3 4 12])
set(gca,'YTickLabel',{'0','1','2','3','4','12'})

% ==========残差矩阵可视化展示===========
load('2_PRVM_RV_subsampled.mat')
% 得到全部的字段名
% fnames=fieldnames(RV,'-full');
fnames1={'RCOV_PRVM_POET_res_1';'RCOV_PRVM_POET_res_2';'RCOV_PRVM_POET_res_3';'RCOV_PRVM_POET_res_4';...
    'RCOV_PRVM_POET_res_5';'RCOV_PRVM_POET_res_6';'RCOV_PRVM_POET_res_7';'RCOV_PRVM_POET_res_8'};
fnames2={'P_PRVM';'P_sample';'N_PRVM';'N_sample'};
fnames3={'POET_hard_res';'POET_soft_res';'POET_AL_res'};
% 得到日收益率矩阵 r - 证券数x天数(从旧到新)
r=getfield(RV,'dailyre')';
% 将所有协方差大小分为几个类？
% type=10;
% 按照P和N矩阵重新排列残差矩阵
for i=1:size(fnames2,1)
    matrix=getfield(RV,fnames2{i});
    order=[];
    size_type=[];
    for j=1:size(matrix,3)
        if j<20
            temp_matrix=sum(matrix(:,:,1:j),3);
            [ rearrange_matrix,rearrange_order,serial] = rearrange( type,temp_matrix,0.9);

            order=[order rearrange_order];
            temp_size_type=[];
            for kk=1:size(serial,2)
                temp_size_type(kk,1)=size(serial{kk},1);
            end
            size_type=[size_type;temp_size_type];
% 画图比较排列后的P
% imagesc(rearrange_matrix)
% caxis([0 1])
% colormap(parula)

        else
            temp_matrix=sum(matrix(:,:,j-20+1:j),3);
            [ rearrange_matrix,rearrange_order,serial ] = rearrange( type,temp_matrix,0.9);
            order=[order rearrange_order];
            temp_size_type=[];
            for kk=1:size(serial,2)
                temp_size_type(kk,1)=size(serial{kk},1);
            end
            size_type=[size_type;temp_size_type];
        end
    end
residuals(i).order=order;
residuals(i).size_type=size_type;
end
% 按4种行业进行分组
temp=[[1:size(Sortby_Ind,2)]' [Sortby_Ind.SW_code]'];
[B,I]=sort(temp(:,2)); % I为原元素排列后的位置
Groups.SW=temp(I,:);
[a,ia,ic]=unique(Groups.SW(:,2));
Groups.SW_location=[ia;size(Sortby_Ind,2)]; % 各组开始的位置
Groups.SW_GroupNum=size(unique(Groups.SW(:,2)),1);
Groups.SW_GroupName=[Sortby_Ind.SW]';Groups.SW_GroupName=Groups.SW_GroupName(I,:);

temp=[[1:size(Sortby_Ind,2)]' [Sortby_Ind.ZX_code]'];
[B,I]=sort(temp(:,2)); % I为原元素排列后的位置
Groups.ZX=temp(I,:);
[a,ia,ic]=unique(Groups.ZX(:,2));
Groups.ZX_location=[ia;size(Sortby_Ind,2)]; % 各组开始的位置
Groups.ZX_GroupNum=size(unique(Groups.ZX(:,2)),1);
Groups.ZX_GroupName=[Sortby_Ind.ZX]';Groups.ZX_GroupName=Groups.ZX_GroupName(I,:);

temp=[[1:size(Sortby_Ind,2)]' [Sortby_Ind.ZZ_code]'];
[B,I]=sort(temp(:,2)); % I为原元素排列后的位置
Groups.ZZ=temp(I,:);
[a,ia,ic]=unique(Groups.ZZ(:,2));
Groups.ZZ_location=[ia;size(Sortby_Ind,2)]; % 各组开始的位置
Groups.ZZ_GroupNum=size(unique(Groups.ZZ(:,2)),1);
Groups.ZZ_GroupName=[Sortby_Ind.ZZ]';Groups.ZZ_GroupName=Groups.ZZ_GroupName(I,:);

temp=[[1:size(Sortby_Ind,2)]' [Sortby_Ind.ZJH_code]'];
[B,I]=sort(temp(:,2)); % I为原元素排列后的位置
Groups.ZJH=temp(I,:);
[a,ia,ic]=unique(Groups.ZJH(:,2));
Groups.ZJH_location=[ia;size(Sortby_Ind,2)]; % 各组开始的位置
Groups.ZJH_GroupNum=size(unique(Groups.ZJH(:,2)),1);
Groups.ZJH_GroupName=[Sortby_Ind.ZJH]';Groups.ZJH_GroupName=Groups.ZJH_GroupName(I,:);

% 标记残差矩阵中连续一半以上天数的相关系数（注意是相关系数！！）均大于特定值的位置
value=0.15;
for i=1:size(fnames1,1)
    matrix=getfield(RV,fnames1{i});
    for j=1:size(matrix,3)
        temp_m=matrix(:,:,j);
        diagonal=diag(temp_m);
        diagonal(diagonal<0)=mean(diagonal);
        temp_m=inv(sqrt(diag(diagonal)))*temp_m*inv(sqrt(diag(diagonal))); %相关系数矩阵
        temp_m(temp_m<value)=0;
        temp_m(temp_m>=value)=value; % 将大于等于value的均赋予一个大于value的定值
        matrix(:,:,j)=temp_m;
    end
    temp_sum=sum(matrix,3);
    temp_sum(temp_sum<value*size(matrix,3)/2)=0;
    temp_sum(logical(diag(ones(size(temp_sum,1),1))))=1;
    residuals(i).res=temp_sum;
    % residuals(i).factor_num=str2num(fnames1{i}(regexp(fnames1{i},'\d')));
end
save('C:\Users\qianp\Desktop\论文\研究内容\2018-12-23\编程相关\数据\处理完的数据\data.mat','Sortby_Ind','-append')
clear
load('C:\Users\qianp\Desktop\论文\研究内容\2018-12-23\编程相关\数据\处理完的数据\data.mat')
% =======图4 不同因子个数的残差矩阵=======
figure('units','normalized','position',[0.1,0.1,0.8,0.32])
subplot(1,4,1)
im=imagesc(residuals(1).res); %matrix(:,:,100)
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gcf,'color','white')
set(gca,'ticklength',[0 0])
title({'(a) $$r=1$$'},'Interpreter','latex','FontSize',12)
axis xy
subplot(1,4,2)
im=imagesc(residuals(3).res);
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gcf,'color','white')
set(gca,'ticklength',[0 0])
title({'(b) $$r=3$$'},'Interpreter','latex','FontSize',12)
axis xy
subplot(1,4,3)
im=imagesc(residuals(5).res);
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gcf,'color','white')
set(gca,'ticklength',[0 0])
title({'(c) $$r=5$$'},'Interpreter','latex','FontSize',12)
axis xy
subplot(1,4,4)
im=imagesc(residuals(7).res);
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gcf,'color','white')
set(gca,'ticklength',[0 0])
title({'(d) $$r=7$$'},'Interpreter','latex','FontSize',12)
axis xy
% =======图5：按照不同行业标准进行重排=======
figure('units','normalized','position',[0.1,0.1,0.8,0.32])
subplot(1,3,1)
im=imagesc(residuals(6).res(Groups.SW(:,1),Groups.SW(:,1))); %matrix(:,:,100)
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gcf,'color','white')
title({'(a) SW'},'Interpreter','latex','FontSize',12)
axis xy
set(gca,'ytick',[10 106 172],'yticklabel',[{'银行'},{'非银金融'},{'国防军工'}])
set(gca,'xtick',[10 106 172],'xticklabel',[{'银行'},{'非银金融'},{'国防军工'}])
for i=1:size(Groups.SW_location,1)-1
    line([Groups.SW_location(i) Groups.SW_location(i)],[Groups.SW_location(i) Groups.SW_location(i+1)-1],'color','r','LineWidth',0.1)
    line([Groups.SW_location(i) Groups.SW_location(i+1)-1],[Groups.SW_location(i) Groups.SW_location(i)],'color','r','LineWidth',0.1)
    line([Groups.SW_location(i+1)-1 Groups.SW_location(i+1)-1],[Groups.SW_location(i) Groups.SW_location(i+1)-1],'color','r','LineWidth',0.1)
    line([Groups.SW_location(i) Groups.SW_location(i+1)-1],[Groups.SW_location(i+1)-1 Groups.SW_location(i+1)-1],'color','r','LineWidth',0.1)
end
% 稀疏度
loc=[];
for i=1:size(Groups.SW_location,1)-1
    j=Groups.SW_location(i+1)-Groups.SW_location(i);
    temp=residuals(6).res(Groups.SW(:,1),Groups.SW(:,1));
    loc(i,:)=[Groups.SW_location(i) Groups.SW_location(i+1)-1 ...
        (sum(sum(temp(Groups.SW_location(i):Groups.SW_location(i+1)-1,...
        Groups.SW_location(i):Groups.SW_location(i+1)-1)==0,1),2))/(j*j)];
end
mean(loc(~isnan(loc(:,3)),3)) 
subplot(1,3,2)
im=imagesc(residuals(6).res(Groups.ZZ(:,1),Groups.ZZ(:,1)));
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gcf,'color','white')
title({'(b) CSI'},'Interpreter','latex','FontSize',12)
axis xy
set(gca,'ytick',[29 114 194 259],'yticklabel',[{'金融地产'},{'原材料'},{'能源'},{'主要消费'}])
set(gca,'xtick',[29 114 194 259],'xticklabel',[{'金融地产'},{'原材料'},{'能源'},{'主要消费'}])
for i=1:size(Groups.ZZ_location,1)-1
    line([Groups.ZZ_location(i) Groups.ZZ_location(i)],[Groups.ZZ_location(i) Groups.ZZ_location(i+1)-1],'color','r','LineWidth',0.1)
    line([Groups.ZZ_location(i) Groups.ZZ_location(i+1)-1],[Groups.ZZ_location(i) Groups.ZZ_location(i)],'color','r','LineWidth',0.1)
    line([Groups.ZZ_location(i+1)-1 Groups.ZZ_location(i+1)-1],[Groups.ZZ_location(i) Groups.ZZ_location(i+1)-1],'color','r','LineWidth',0.1)
    line([Groups.ZZ_location(i) Groups.ZZ_location(i+1)-1],[Groups.ZZ_location(i+1)-1 Groups.ZZ_location(i+1)-1],'color','r','LineWidth',0.1)
end

loc=[];
for i=1:size(Groups.ZZ_location,1)-1
    j=Groups.ZZ_location(i+1)-Groups.ZZ_location(i);
    temp=residuals(6).res(Groups.ZZ(:,1),Groups.ZZ(:,1));
    loc(i,:)=[Groups.ZZ_location(i) Groups.ZZ_location(i+1)-1 ...
        (sum(sum(temp(Groups.ZZ_location(i):Groups.ZZ_location(i+1)-1,...
        Groups.ZZ_location(i):Groups.ZZ_location(i+1)-1)==0,1),2))/(j*j)];
end
mean(loc(~isnan(loc(:,3)),3)) 
subplot(1,4,3)
im=imagesc(residuals(6).res(Groups.ZX(:,1),Groups.ZX(:,1)));
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gcf,'color','white')
title({'ZX'},'Interpreter','latex','FontSize',12)
axis xy
       set(gca,'ytick',[10 115 182],'yticklabel',[{'银行'},{'非银金融'},{'国防军工'}])
       set(gca,'xtick',[10 115 182],'xticklabel',[{'银行'},{'非银金融'},{'国防军工'}])
for i=1:size(Groups.ZX_location,1)-1
 line([Groups.ZX_location(i) Groups.ZX_location(i)],[Groups.ZX_location(i) Groups.ZX_location(i+1)-1],'color','r','LineWidth',0.1)
 line([Groups.ZX_location(i) Groups.ZX_location(i+1)-1],[Groups.ZX_location(i) Groups.ZX_location(i)],'color','r','LineWidth',0.1)
 line([Groups.ZX_location(i+1)-1 Groups.ZX_location(i+1)-1],[Groups.ZX_location(i) Groups.ZX_location(i+1)-1],'color','r','LineWidth',0.1)
 line([Groups.ZX_location(i) Groups.ZX_location(i+1)-1],[Groups.ZX_location(i+1)-1 Groups.ZX_location(i+1)-1],'color','r','LineWidth',0.1)
end

loc=[];
for i=1:size(Groups.ZX_location,1)-1
    j=Groups.ZX_location(i+1)-Groups.ZX_location(i);
    loc(i,:)=[Groups.ZX_location(i) Groups.ZX_location(i+1)-1 ...
        (sum(sum(residuals(6).res(Groups.ZX_location(i):Groups.ZX_location(i+1)-1,...
        Groups.ZX_location(i):Groups.ZX_location(i+1)-1)==0,1),2))/(j*j-j)];
end
mean(loc(~isnan(loc(:,3)),3)) % 0.992305297621356
subplot(1,3,3)
im=imagesc(residuals(6).res(Groups.ZJH(:,1),Groups.ZJH(:,1)));
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gcf,'color','white')
title({'(c) CSRC'},'Interpreter','latex','FontSize',12)
axis xy
set(gca,'ytick',[23 128 202],'yticklabel',[{'金融业'},{'制造业'},{'采矿业'}])
set(gca,'xtick',[23 128 202],'xticklabel',[{'金融业'},{'制造业'},{'采矿业'}])
for i=1:size(Groups.ZJH_location,1)-1
    line([Groups.ZJH_location(i) Groups.ZJH_location(i)],[Groups.ZJH_location(i) Groups.ZJH_location(i+1)-1],'color','r','LineWidth',0.1)
    line([Groups.ZJH_location(i) Groups.ZJH_location(i+1)-1],[Groups.ZJH_location(i) Groups.ZJH_location(i)],'color','r','LineWidth',0.1)
    line([Groups.ZJH_location(i+1)-1 Groups.ZJH_location(i+1)-1],[Groups.ZJH_location(i) Groups.ZJH_location(i+1)-1],'color','r','LineWidth',0.1)
    line([Groups.ZJH_location(i) Groups.ZJH_location(i+1)-1],[Groups.ZJH_location(i+1)-1 Groups.ZJH_location(i+1)-1],'color','r','LineWidth',0.1)
end

loc=[];
for i=1:size(Groups.ZJH_location,1)-1
    j=Groups.ZJH_location(i+1)-Groups.ZJH_location(i);
    temp=residuals(6).res(Groups.ZJH(:,1),Groups.ZJH(:,1));
    loc(i,:)=[Groups.ZJH_location(i) Groups.ZJH_location(i+1)-1 ...
        (sum(sum(temp(Groups.ZJH_location(i):Groups.ZJH_location(i+1)-1,...
        Groups.ZJH_location(i):Groups.ZJH_location(i+1)-1)==0,1),2))/(j*j)];
end
mean(loc(~(isnan(loc(:,3))+(loc(:,3)==0)),3)) % 0.992305297621356
% ======图6：RCM算法的结果
figure('units','normalized','position',[0.1,0.1,0.8,0.32])
subplot(1,4,1)
% [p,~,~,~,~,~] = dmperm(residuals(6).res);
% block_m = residuals(6).res(p,p);
p =genrcm(residuals(6).res);
block_m = residuals(6).res(p,p);
xyticklabel=[{'103'},{'232'},{'18'},{'65'},{'224'},{'201'}];
xytick=1:50:280;
% 检测对角线上的非稀疏方块
max_block=round(size(block_m,1)*0.5); % 最大允许不超过多大的方块
sparsity=0.1;
loc=[];
log=[];
next=1;
for i=1:size(block_m,1)
    log_temp=[];
    if i==next
        for j=size(block_m,1)-i+1:-1:3
            if ((sum(sum(block_m(i:i+j-1,i:i+j-1)==0,1),2))/(j*j))<=sparsity
                %                loc=[loc;i i+j-1];
                log_temp=[log_temp;i i+j-1 (sum(sum(block_m(i:i+j-1,i:i+j-1)==0,1),2))/(j*j)];
                %                next=i+j-1;
                %                break
                %               else next=i+1;
            end
        end
        if ~isempty(log_temp)
            %              next=log_temp(find(log_temp(:,3)==min(log_temp(:,3))),2); % 找满足稀疏条件中最小的
            next=log_temp(:,2); % 找满足稀疏条件中方块最大的
            next=next(1);
            loc=[loc;i next];
        else
            next=i+1;
        end
    end
end
im=imagesc(block_m);
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gca,'ticklength',[0.01 0.01])
set(gcf,'color','white')
title({'(a) $$s=0.1$$'},'Interpreter','latex','FontSize',12)
set(gca,'ytick',xytick,'yticklabel',xyticklabel)
set(gca,'xtick',xytick,'xticklabel',xyticklabel)
axis xy
for i=1:size(loc,1)
    line([loc(i,1) loc(i,1)],[loc(i,1) loc(i,2)],'color','r','LineWidth',0.1)
    line([loc(i,1) loc(i,2)],[loc(i,1) loc(i,1)],'color','r','LineWidth',0.1)
    line([loc(i,2) loc(i,2)],[loc(i,1) loc(i,2)],'color','r','LineWidth',0.1)
    line([loc(i,1) loc(i,2)],[loc(i,2) loc(i,2)],'color','r','LineWidth',0.1)
end
% 稀疏度
s=[];
for i=1:size(loc,1)-1
    j=loc(i,2)-loc(i,1)+1;
    s(i)=(sum(sum(block_m(loc(i,1):loc(i,2),...
        loc(i,1):loc(i,2))==0,1),2))/(j*j);
end
mean(s)
subplot(1,4,2)
p =genrcm(residuals(6).res);
block_m = residuals(6).res(p,p);
% 检测对角线上的非稀疏方块
max_block=round(size(block_m,1)*0.5); % 最大允许不超过多大的方块
sparsity=0.2;
loc=[];
log=[];
next=1;
for i=1:size(block_m,1)
    log_temp=[];
    if i==next
        for j=size(block_m,1)-i+1:-1:3
            if ((sum(sum(block_m(i:i+j-1,i:i+j-1)==0,1),2))/(j*j))<=sparsity
                %                loc=[loc;i i+j-1];
                log_temp=[log_temp;i i+j-1 (sum(sum(block_m(i:i+j-1,i:i+j-1)==0,1),2))/(j*j)];
                %                next=i+j-1;
                %                break
                %               else next=i+1;
            end
        end
        if ~isempty(log_temp)
            %              next=log_temp(find(log_temp(:,3)==min(log_temp(:,3))),2); % 找满足稀疏条件中最小的
            next=log_temp(:,2); % 找满足稀疏条件中方块最大的
            next=next(1);
            loc=[loc;i next];
        else
            next=i+1;
        end
    end
end
im=imagesc(block_m);
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gca,'ticklength',[0.01 0.01])
set(gcf,'color','white')
title({'(b) $$s=0.2$$'},'Interpreter','latex','FontSize',12)
set(gca,'ytick',xytick,'yticklabel',xyticklabel)
set(gca,'xtick',xytick,'xticklabel',xyticklabel)
axis xy
for i=1:size(loc,1)
    line([loc(i,1) loc(i,1)],[loc(i,1) loc(i,2)],'color','r','LineWidth',0.1)
    line([loc(i,1) loc(i,2)],[loc(i,1) loc(i,1)],'color','r','LineWidth',0.1)
    line([loc(i,2) loc(i,2)],[loc(i,1) loc(i,2)],'color','r','LineWidth',0.1)
    line([loc(i,1) loc(i,2)],[loc(i,2) loc(i,2)],'color','r','LineWidth',0.1)
end
% 稀疏度
s=[];
for i=1:size(loc,1)-1
    j=loc(i,2)-loc(i,1)+1;
    s(i)=(sum(sum(block_m(loc(i,1):loc(i,2),...
        loc(i,1):loc(i,2))==0,1),2))/(j*j);
end
mean(s)
subplot(1,4,3)
p =genrcm(residuals(6).res);
block_m = residuals(6).res(p,p);
% 检测对角线上的非稀疏方块
max_block=round(size(block_m,1)*0.5); % 最大允许不超过多大的方块
sparsity=0.3;
loc=[];
log=[];
next=1;
for i=1:size(block_m,1)
    log_temp=[];
    if i==next
        for j=size(block_m,1)-i+1:-1:3
            if ((sum(sum(block_m(i:i+j-1,i:i+j-1)==0,1),2))/(j*j))<=sparsity
                %                loc=[loc;i i+j-1];
                log_temp=[log_temp;i i+j-1 (sum(sum(block_m(i:i+j-1,i:i+j-1)==0,1),2))/(j*j)];
                %                next=i+j-1;
                %                break
                %               else next=i+1;
            end
        end
        if ~isempty(log_temp)
            %              next=log_temp(find(log_temp(:,3)==min(log_temp(:,3))),2); % 找满足稀疏条件中最小的
            next=log_temp(:,2); % 找满足稀疏条件中方块最大的
            next=next(1);
            loc=[loc;i next];
        else
            next=i+1;
        end
    end
end
im=imagesc(block_m);
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gca,'ticklength',[0.01 0.01])
set(gcf,'color','white')
title({'(c) $$s=0.3$$'},'Interpreter','latex','FontSize',12)
set(gca,'ytick',xytick,'yticklabel',xyticklabel)
set(gca,'xtick',xytick,'xticklabel',xyticklabel)
axis xy
for i=1:size(loc,1)
    line([loc(i,1) loc(i,1)],[loc(i,1) loc(i,2)],'color','r','LineWidth',0.1)
    line([loc(i,1) loc(i,2)],[loc(i,1) loc(i,1)],'color','r','LineWidth',0.1)
    line([loc(i,2) loc(i,2)],[loc(i,1) loc(i,2)],'color','r','LineWidth',0.1)
    line([loc(i,1) loc(i,2)],[loc(i,2) loc(i,2)],'color','r','LineWidth',0.1)
end
ans=[Sortby_Ind.ZJH_code];
ans(p(20:36))
% 稀疏度
s=[];
for i=1:size(loc,1)-1
    j=loc(i,2)-loc(i,1)+1;
    s(i)=(sum(sum(block_m(loc(i,1):loc(i,2),...
        loc(i,1):loc(i,2))==0,1),2))/(j*j);
end
mean(s)
subplot(1,4,4)
p =genrcm(residuals(6).res);
block_m = residuals(6).res(p,p);
% 检测对角线上的非稀疏方块
max_block=round(size(block_m,1)*0.5); % 最大允许不超过多大的方块
sparsity=0.4;
loc=[];
log=[];
next=1;
for i=1:size(block_m,1)
    log_temp=[];
    if i==next
        for j=size(block_m,1)-i+1:-1:3
            if ((sum(sum(block_m(i:i+j-1,i:i+j-1)==0,1),2))/(j*j))<=sparsity
                %                loc=[loc;i i+j-1];
                log_temp=[log_temp;i i+j-1 (sum(sum(block_m(i:i+j-1,i:i+j-1)==0,1),2))/(j*j)];
                %                next=i+j-1;
                %                break
                %               else next=i+1;
            end
        end
        if ~isempty(log_temp)
            %              next=log_temp(find(log_temp(:,3)==min(log_temp(:,3))),2); % 找满足稀疏条件中最小的
            next=log_temp(:,2); % 找满足稀疏条件中方块最大的
            next=next(1);
            loc=[loc;i next];
        else
            next=i+1;
        end
    end
end
im=imagesc(block_m);
map = [ones(3999,3);[70 130 180]./255];colormap(map)
caxis([0.0001-0.00000001 0.0001])
set(gca,'ticklength',[0.01 0.01])
set(gcf,'color','white')
title({'(d) $$s=0.4$$'},'Interpreter','latex','FontSize',12)
set(gca,'ytick',xytick,'yticklabel',xyticklabel)
set(gca,'xtick',xytick,'xticklabel',xyticklabel)
axis xy
for i=1:size(loc,1)
    line([loc(i,1) loc(i,1)],[loc(i,1) loc(i,2)],'color','r','LineWidth',0.1)
    line([loc(i,1) loc(i,2)],[loc(i,1) loc(i,1)],'color','r','LineWidth',0.1)
    line([loc(i,2) loc(i,2)],[loc(i,1) loc(i,2)],'color','r','LineWidth',0.1)
    line([loc(i,1) loc(i,2)],[loc(i,2) loc(i,2)],'color','r','LineWidth',0.1)
end
% 稀疏度
s=[];
for i=1:size(loc,1)-1
    j=loc(i,2)-loc(i,1)+1;
    s(i)=(sum(sum(block_m(loc(i,1):loc(i,2),...
        loc(i,1):loc(i,2))==0,1),2))/(j*j);
end
mean(s)

%% UTAT
function [matrix] =UTAT( matrix,type,theta,n )
% Universal Thresholding / Adaptive Thresholding
% Kim D et al. Adaptive thresholding for large volatility matrix estimation based on high-frequency financial data [J]. 
% Journal of Econometrics, 2018, 203(1): 69-79.
lambda=1;

switch type
    case 'uni'
     w=mean(mean(matrix));
     diag_ele=diag(matrix);
     matrix=(matrix-sign(matrix).*w).*(abs(matrix)>=w);
     matrix(logical(diag(ones(280,1))))=diag_ele.*double((diag_ele>=0));   
        
    case 'ada'
     w=lambda*sqrt(theta*n^(-0.5)*log(280));
     diag_ele=diag(matrix);
     matrix=(matrix-sign(matrix).*w).*(abs(matrix)>=w);
     matrix(logical(diag(ones(280,1))))=diag_ele.*double((diag_ele>=0));
        
         
end
end


%% rearrange_DM
function [ rearrange_matrix,rearrange_order,serial] = rearrange_DM( type,temp_matrix)
% rearrange 按照分类数目重排协方差矩阵
% serials-想要排列的顺序 输入矩阵的顺序为1,2,3,...,size(matrix,1)
% matrix--协方差矩阵
% left----[1:股票数], 即原始矩阵中的顺序
% pct-----每次寻找最大分块的百分比

            serial={};
           left=1:size(temp_matrix,1);
           
for k=1:type
    serial_temp=left((k-1)*size(temp_matrix,1)/type+1:k*size(temp_matrix,1)/type);
    temp_m=temp_matrix(serial_temp,serial_temp);
        [p,~,~,~,~,~] = dmperm(temp_m);
         block_m = temp_m(p,p);
        serial{k}=serial_temp(p)'; % 保留组成方块的股票在原协方差中的位置
end

rearrange_order=[];
for l=1:size(serial,2)
rearrange_order=[rearrange_order;serial{l}];
end

rearrange_matrix=temp_matrix(rearrange_order,rearrange_order);

end

%% POET-v
function [ s_cov,residual_opt,residual_t,eigenvalues] = POET_v( v,k,c,thres,t) % 
% POET: Principal Orthogonal ComplEment Thresholding (POET)
% v----n*n的已实现协方差矩阵
% k----因子个数 默认为 2
% c----阈值常数 默认为 0.5
% thres阈值方法(高频版本) 取值--'hard','soft','AL'(adaptive lasso) 
% residual_opt---------PCA后的残差矩阵
% residual_t---------PCA后+阈值处理后的残差矩阵
if nargin<2
     k=2;c=0.5;thres_type=1;% 默认为soft
else thres_type=strcmp(thres,'hard')*1+strcmp(thres,'soft')*2+strcmp(thres,'AL')*3+strcmp(thres,'diag')*4;
end

% [t,n]=size(r);
n=size(v,1);
[eigenvectors,eigenvalues] = eig(v);
eigenvalues=repmat(diag(eigenvalues)',n,1); % 特征值的顺序从小到大
cov_hat=(eigenvalues(:,end-k+1:end).*eigenvectors(:,end-k+1:end))*eigenvectors(:,end-k+1:end)'; % 主成分矩阵
cov_hat=(cov_hat+cov_hat')/2; % 去除浮点误差 保证对称
residual=v-cov_hat; % 残差矩阵，或者称为正交互余矩阵(orthogonal complement)
residual_opt=residual;

% 对残差矩阵进行门限/阈值处理 保证gamma的稀疏性 
ii=kron(diag(residual),ones(1,n));  ii=(abs(ii)+ii)/2;
jj=kron(ones(n,1),diag(residual)'); jj=(abs(jj)+jj)/2;
% w=c*sqrt(t^(-1/2)*log(n).*(ii).*jj); % 阈值方案
w=c*sqrt((ii).*jj);

% adaptive 
if thres_type==3
residual(residual==0)=1e-10;
eta=4;
diagn=diag(residual);
diagn(diagn<0)=0;
residual_t=residual.*(abs(1-abs(w./residual)^eta)+(1-abs(w./residual)^eta))/2;
residual_t(logical(eye(n)))=diagn;

% soft 
else if thres_type==2
diagn=diag(residual);
diagn(diagn<0)=0;
residual_t=(abs(residual)>w).*(residual-sign(residual).*w);
residual_t(logical(eye(n)))=diagn;    

% hard
    else if thres_type==1
 diagn=diag(residual);diagn(diagn<0)=0;
residual_t=residual.*(abs(residual)>w);
residual_t(logical(eye(n)))=diagn; 
        else
% diag ---仅保留对角元素
residual_t=diag(diag(residual));            
        end
    end   
end
        
% 确保对称
residual_t=(residual_t+residual_t')/2;

% 得到已实现协方差估计量
s_cov=residual_t+cov_hat;

end

%% Transform
function V_T = Transform( V,Delta )
% 输出任何对称伪协方差矩阵的最近协方差矩阵(相关系数矩阵)
%   V----输入矩阵
%   Delta-----最近相关系数矩阵最小特征值的下限, 默认为1e-8（大于0）
if isempty(Delta)
    Delta=1e-8;
end
R=diag(diag(1./sqrt(V)))*V*diag(diag(1./sqrt(V)));
R=(R+R')./2;
% R=round(R,8);
R=nearcorr_aa(R,[],2,100000,'u',Delta); 
V_T=inv(diag(diag(1./sqrt(V))))*R*inv(diag(diag(1./sqrt(V))));
end

%% block-diagonal POET_block_diag
function [ s_cov,residual_opt,residual_thred ] = POET_block_diag( v,k,block_type,Ind,sparsity)
% 基于行业的分类方法/使用RCM算法进行分类 得到 POET估计量

% v----n*n的已实现协方差矩阵
% k----因子个数
% block_type------'ind'使用Ait-Sahalia和Xiu(2017)基于行业的分类方法;
%           ------'rcm'使用RCM算法进行分类
% Ind-------行业分类 对应Sortby_Ind变量  293只股票的行业分类.mat
% sparsity--RCM寻找分块大小的稀疏度
% residual_thred--对对角分块外元素作门限处理后的残差矩阵

n=size(v,1);
[eigenvectors,eigenvalues] = eig(v);
eigenvalues=repmat(diag(eigenvalues)',n,1); % 特征值的顺序从小到大
cov_hat=(eigenvalues(:,end-k+1:end).*eigenvectors(:,end-k+1:end))*eigenvectors(:,end-k+1:end)'; % 主成分矩阵
cov_hat=(cov_hat+cov_hat')/2; % 去除浮点误差 保证对称
residual=v-cov_hat; % 残差矩阵，或者称为正交互余矩阵(orthogonal complement)
residual_opt=residual;
% 对残差矩阵分块对角外的元素取零处理
switch block_type
    case 'ind'
    temp=[[1:size(Ind,2)]' [Ind.SW_code]'];
    [~,I]=sort(temp(:,2)); % I为原元素排列后的位置
    Groups.SW=temp(I,:); 
    [~,ia,~]=unique(Groups.SW(:,2));
    Groups.SW_location=[ia;size(Ind,2)]; % 各组开始的位置
    Groups.SW_GroupNum=size(unique(Groups.SW(:,2)),1);
    Groups.SW_GroupName=[Ind.SW]';
    Groups.SW_GroupName=Groups.SW_GroupName(I,:);
        
% 标记残差矩阵中连续一半以上天数的相关系数（注意是相关系数！！）均大于特定值的位置
    value=0.15;
    % 得到相关系数矩阵 
    diagonal=diag(residual_opt);
    CORR=inv(sqrt(diag(diagonal)))*residual_opt*inv(sqrt(diag(diagonal)));
    
    residual_opt(CORR<value)=0;
    residual_sorted=residual_opt(Groups.SW(:,1),Groups.SW(:,1)); % 按照行业分类重新排列过的残差矩阵
    residual_thred=zeros(size(v,1),size(v,1))+diag(diag(residual_sorted));                   % 按照行业分类重新排列过的残差矩阵
    for i=1:size(Groups.SW_location,1)-1
       residual_thred(Groups.SW_location(i):Groups.SW_location(i+1)-1,Groups.SW_location(i):Groups.SW_location(i+1)-1)...
       =residual_sorted(Groups.SW_location(i):Groups.SW_location(i+1)-1,Groups.SW_location(i):Groups.SW_location(i+1)-1);  
    end
    % 确保对称
    residual_thred=(residual_thred+residual_thred')/2;   
    [~,original_loc]=sort(Groups.SW(:,1));
    residual_thred=residual_thred(original_loc,original_loc); % 按原来顺序排列
    % 得到已实现协方差估计量
     s_cov=residual_thred+cov_hat;  
    case 'rcm' 
        % 标记残差矩阵中连续一半以上天数的相关系数（注意是相关系数！！）均大于特定值的位置
        value=0.15;
        % 得到相关系数矩阵 
        diagonal=diag(residual_opt);
        CORR=inv(sqrt(diag(diagonal)))*residual_opt*inv(sqrt(diag(diagonal)));
        residual_opt(CORR<value)=0;
        p =genrcm(residual_opt);
        block_m =residual_opt(p,p); % 按照行业分类重新排列过的残差矩阵 block_m
        
        % 寻找符合条件的对角分块
        max_block=round(size(block_m,1)*0.5); % 最大允许不超过多大的方块
        loc=[];
        log=[];
        next=1;
        for i=1:size(block_m,1)
             log_temp=[];
             if i==next
                 for j=size(block_m,1)-i+1:-1:3
                      if ((sum(sum(block_m(i:i+j-1,i:i+j-1)==0,1),2))/(j*j))<=sparsity  
                        log_temp=[log_temp;i i+j-1 (sum(sum(block_m(i:i+j-1,i:i+j-1)==0,1),2))/(j*j)];
                      end
                 end
                 if ~isempty(log_temp)
                 next=log_temp(:,2); % 找满足稀疏条件中方块最大的
                 next=next(1);
                 loc=[loc;i next];
                 else 
                    next=i+1; 
                 end
             end
        end
        residual_thred=zeros(size(v,1),size(v,1))+diag(diag(block_m));
        
            for i=1:size(loc,1)
               residual_thred(loc(i,1):loc(i,2),loc(i,1):loc(i,2))=block_m(loc(i,1):loc(i,2),loc(i,1):loc(i,2));  
            end    
        % 确保对称
        residual_thred=(residual_thred+residual_thred')/2;        
        [~,original_loc]=sort(p);
        residual_thred=residual_thred(original_loc,original_loc);
        % 得到已实现协方差估计量
        s_cov=residual_thred+cov_hat;       
end
end