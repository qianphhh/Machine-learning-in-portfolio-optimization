%% data processing
clear
% 处理用于检验FF或投资组合绩效的 按因子分的 n x n组合
rawdapath='H:\工作台\系统-MTRL-修改-2020-12-30\原始数据';
datapath='H:\工作台\系统-MTRL-修改-2020-12-30\A股数据';% 数据所在路径
savepath='H:\工作台\系统-MTRL-修改-2020-12-30\A股数据\因子组合数据'; % 数据存储路径
%%%%%%%%%%% 将原始股票数据处理为“日期.csv' %%%%%%%%%%%%%%%%
di=dir([rawdapath,'\A股日数据']);
di(1:2)=[];
tic
for i=1:size(di,1)
    [~,~,da]=xlsread([di(i).folder,'\',di(i).name]);
    hearerline=da(1,:);
    da(1,:)=[];
    uni_date=unique(da(:,1));
    
    for j=1:size(uni_date,1)
        [num2str(i/size(di,1)*100),' ',num2str(j/size(uni_date,1)*100)]
        temp=da(ismember(da(:,1),uni_date(j)),[1 2 4 5 6]); 
        date_string=datestr(datetime(uni_date(j)),'yyyy-mm-dd');
        file=fopen([datapath,'\A股日数据\',date_string,'.csv'],'a'); % w为写入的清空现有并写入的意思 a为补充写入
        if exist([datapath,'\A股日数据\',date_string,'.csv'],'file')~=0
        writecell([temp],[datapath,'\A股日数据\',date_string,'.csv'],'FileType','text','WriteMode','append');
        else
        writecell([hearerline(:,[1 2 4 5 6]);temp],[datapath,'\A股日数据\',date_string,'.csv'],'FileType','text');    
        end
    end
    fclose all
end
toc


%选择时间段
start_date='2000-01-04';
end_date='2019-12-31';


%%%%%%%%%% 得到全样本的所有股票代码 %%%%%%%%%%%%%%%%%%

di=dir([datapath,'\A股日数据']);
di(1:2)=[];
temp=struct2cell(di);
filename=temp(1,:)';
start_loc=find(ismember(filename,{[start_date,'.csv']})~=0); % 想要的开始日期
end_loc=find(ismember(filename,{[end_date,'.csv']})~=0); % 想要的开始日期
stk_list=[];
kk=1;
total_cic=end_loc-start_loc+1;
for i=start_loc:end_loc
    
    % 进度条
    percent=kk/total_cic*100;
    kk=kk+1;
    perc = sprintf('%3.0f%%', percent); % 4 characters wide, percentage
    disp([repmat(char(8), 1, (50+9)), char(10), perc, '[', repmat('=', 1, round(percent*50/100)), '>', repmat(' ', 1, 50 - round(percent*50/100)), ']']);
    
    file=fopen([di(i).folder,'\',di(i).name]);
    da=textscan(file,'%{yyyy/MM/dd}D %f %f %f %f','Delimiter',',','HeaderLines',1);
    fclose(file);
    temp=[da{1,2} da{1,3} da{1,4} da{1,5}];
    temp(find(isnan(temp(:,2))+isnan(temp(:,3))==1),:)=[]; % 删除有缺失值的股票
    temp((floor(temp(:,1)./100000)==2)|(floor(temp(:,1)./100000)==9),:)=[]; % 删除B股股票
    % 为全样本的数据收集做准备
    stk_list=unique([stk_list;temp(:,1)]); % 更新股票列表 构建全样本的所有股票组成
end
save([savepath,'\过程数据\daily_stk_list_all.mat'],'stk_list')

%%%%%%% 生成全部股票收益率数据/市值数据/账面市值比数据 %%%%%%%%%%

load([savepath,'\过程数据\daily_stk_list_all.mat']);
di=dir([datapath,'\A股日数据']);
di(1:2)=[];
temp=struct2cell(di);
filename=temp(1,:)';
start_loc=find(ismember(filename,{[start_date,'.csv']})~=0); % 想要的开始日期
end_loc=find(ismember(filename,{[end_date,'.csv']})~=0); % 想要的开始日期

kk=1;

daily_re=nan(size(stk_list,1),end_loc-start_loc+1);
daily_size=nan(size(stk_list,1),end_loc-start_loc+1);
daily_bm=nan(size(stk_list,1),end_loc-start_loc+1);

k=1;
total_cic=end_loc-start_loc+1;
for i=start_loc:end_loc
    % 进度条
    percent=kk/total_cic*100;
    kk=kk+1;
    perc = sprintf('%3.0f%%', percent); % 4 characters wide, percentage
    disp([repmat(char(8), 1, (50+9)), char(10), perc, '[', repmat('=', 1, round(percent*50/100)), '>', repmat(' ', 1, 50 - round(percent*50/100)), ']']);

    file=fopen([di(i).folder,'\',di(i).name]);
    da=textscan(file,'%{yyyy/MM/dd}D %f %f %f %f','Delimiter',',','HeaderLines',1);
    fclose(file);
    temp=[da{1,2} da{1,3} da{1,4} da{1,5}];
    temp(find(isnan(temp(:,2))+isnan(temp(:,3))==1),:)=[]; % 删除有缺失值的股票
    temp((floor(temp(:,1)./100000)==2)|(floor(temp(:,1)./100000)==9),:)=[]; % 删除B股股票
    
    if ~isempty(temp(abs(temp(:,4))>=2,:))
    temp(abs(temp(:,4))>=2,4)=0; % 删除涨跌幅过大的股票（可能未除权）
    end
    
    daily_re(ismember(stk_list,temp(:,1)),k)=temp(:,4);
    daily_size(ismember(stk_list,temp(:,1)),k)=temp(:,3);
    daily_bm(ismember(stk_list,temp(:,1)),k)=1./temp(:,2);
    date{:,k}=datestr(unique(da{1,1}),'yyyymmdd');
    
    k=k+1;


end
save([savepath,'\过程数据\daily_re.mat'],'daily_re','date','stk_list')
save([savepath,'\过程数据\daily_size.mat'],'daily_size','date','stk_list')
save([savepath,'\过程数据\daily_bm.mat'],'daily_bm','date','stk_list')
%% main 
clear
% 读取数据
addpath('I:\数据\A股数据\因子组合数据\')
load('BM20.mat') % 不存在收益率缺失的组!
data(1).BM20=date;
data(2).BM20=re_group;
load('S20.mat') % 不存在收益率缺失的组!
data(1).S20=date;
data(2).S20=re_group;
load('S2BM3.mat')  % 不存在收益率缺失的组!
data(1).S2BM3=date;
data(2).S2BM3=re_group;
load('S5BM5.mat');  % 不存在收益率缺失的组!
data(1).S5BM5=date;
data(2).S5BM5=re_group;
load('S6BM6.mat')  % 存在收益率缺失的组有14个，删去这些组.
loc=find(sum(isnan(re_group),1)>0);
re_group(:,loc)=[];
data(1).S6BM6=date;
data(2).S6BM6=re_group;
load('S8BM8.mat')  % 存在收益率缺失的组有14个，删去这些组.
loc=find(sum(isnan(re_group),1)>0);
re_group(:,loc)=[];
data(1).S8BM8=date;
data(2).S8BM8=re_group;
load('S10BM10.mat')  % 存在收益率缺失的组有14个，删去这些组.
loc=find(sum(isnan(re_group),1)>0);
re_group(:,loc)=[];
data(1).S10BM10=date;
data(2).S10BM10=re_group;
load('S11BM11.mat')  % 存在收益率缺失的组有14个，删去这些组.
loc=find(sum(isnan(re_group),1)>0);
re_group(:,loc)=[];
data(1).S11BM11=date;
data(2).S11BM11=re_group;
% 参数设置
% 参数
% rolling_names={'day','week','month'};
% rolling_window=[240 100 48]; % 不同步长 日 周 月
% day=[1 5 20]; % 不同步长分别对应多少天
% eva_window=[120 50 12]; % 评估步长
rolling_names={'day'};
rolling_window=[240]; % 不同步长 日 周 月
train_window=5; % 是rolling_window的几倍？
day=[1]; % 不同步长分别对应多少天
eva_window=[120]; % 评估步长
tic
strategy={'MTL_GMV','GMV','MV'}; %
fname=fieldnames(data);
para_lambda_1=[1];
para_lambda_2=[1];
num_para=1;
for num_para_i=1:size(para_lambda_1,2)
for num_para_j=1:size(para_lambda_2,2)
    lambda_1=para_lambda_1(num_para_i);
    lambda_2=para_lambda_2(num_para_j);
% lambda_1=0;
% lambda_2=0;
for i=1:size(fname,1)
    temp=data(2).(fname{i});
    % 选择实际步长最长的rolling_window 来选择起点日期
    [est_index]=max(rolling_window.*day);
    % 选择评估步长最长的来确定结束的日期
    [eva_index]=max(eva_window.*day);
    % 初始化
    post_r={};
    date_est=[];
    weight=struct;
    % 初始化进度条
    tt=1;
    total_cic=size(temp,1)-eva_index-est_index+1;
    for t=est_index:est_index+100 % :size(temp,1)-eva_index
        
        % 进度条
        percent=tt/total_cic*100;
        perc = sprintf('%3.0f%%', percent); % 4 characters wide, percentage
        disp([repmat(char(8), 1, (50+9)), char(10), perc, '[', repmat('=', 1, round(percent*50/100)), '>', repmat(' ', 1, 50 - round(percent*50/100)), ']']);
        est_r={};
        Y={};
        % 将日数据矩阵转换为3D阵以便于转化为不同频率的数据
        for k=1:size(rolling_window,2)
            est_r{k}=exp(squeeze(sum(permute(reshape(log(1+temp(t-rolling_window(k)*day(k)+1:t,:))',[],day(k),rolling_window(k)),[2 1 3]),1))')-1;
            Y{k}=est_r{k}*inv(est_r{k}'*est_r{k})*ones(size(temp,2),1);
            post_r{k}(:,:,tt)=temp(t+1:t+eva_window(k)*day(k),:);
            date_est{tt}=date(t,:);
        end
        %%%%%%%%% 输入投资组合策略，得到权重 %%%%%%%%%%
        for stra_num=1:size(strategy,2)
                [weight.(strategy{stra_num})(:,:,tt),omegainv{stra_num}]=Port_weight_calcu(Y,est_r,strategy{stra_num},lambda_1,lambda_2);
        end
      tt=tt+1;  
    end
    weight.MTL_MV=weight.MTL_GMV(:,2,:);
    weight.MTL_GMV(:,2,:)=[];
    %%%%%%%%%%%% 评估投资策略 %%%%%%%%%%
    [perf.(fname{i}),perf_detail.(fname{i})]=Port_evaluation(weight,post_r,rolling_names);
end
result(num_para).(fname{i})=perf.(fname{i});
result(num_para).lambda1=lambda_1;
result(num_para).lambda2=lambda_2;
num_para=num_para+1;
end
end
toc
% sharpe=[];
% para=[];
% for i=1:size(result,2)
%     temp=result(i).BM20;
%     para(i,:)=[result(i).lambda1 result(i).lambda2];
%     temp=temp{:,:};
%     sharpe(i,:)=[temp(1,1) temp(4,1)];
% end
% plot(para(:,2),sharpe(:,1),'-b')
% hold on
% plot(para(:,2),sharpe(:,2),'-k')
% hold on
% plot(para(:,2),repmat(temp(2,1),1,size(para(:,1),1)),'--b')
% hold on
% plot(para(:,2),repmat(temp(3,1),1,size(para(:,1),1)),'--k')
% legend({'MTL-GMV','MTL-MV','GMV','MV'})
function [w,omegainv]=Port_weight_calcu(Y,est_r,strategy_type,lambda_1,lambda_2)
% 计算权重
switch strategy_type
    case 'MTL_GMV'
        [w,~,omegainv]=MTL_strategies(Y,est_r,lambda_1,lambda_2,1e-05);
    case 'GMV'
        opts = optimoptions('quadprog','Display','off');
        for i=1:size(est_r,2)
            V=cov(est_r{i}); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
            w(:,i)=quadprog((V+V')/2,[],[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),[],ones(size(V,1),1)./size(V,1),opts);
        end
        omegainv=nan;
    case 'MV'
        opts = optimoptions('quadprog','Display','off');
        for i=1:size(est_r,2)
            V=cov(est_r{i}); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
            mu=mean(est_r{i},1);
            w(:,i)=quadprog((V+V')/4,-mu',[],[],[],[],zeros(size(V,1),1),[],ones(size(V,1),1)./size(V,1),opts);
            w(:,i)=w(:,i)./sum(w(:,i),1);
        end
        omegainv=nan;
end

end
function [perf,perf_detail]=Port_evaluation(weight,post_r,rolling_names)
perf=table;
% 统计投资组合的样本外绩效
fnames=fieldnames(weight);
for i=1:size(fnames,1)
        realized_r=[];
        reb_w=[];
        kk=1;
        t=1;
        for t=1:size(post_r{1},3)
            realized_r(:,:,t)=post_r{1}(:,:,t)*weight.(fnames{i})(:,:,t);
            % 每隔一个评估期间 重新平衡投资组合
            %             if mod(t-1,size(post_r{i},1))==0
            %                 kk=kk+1;
            %             temp=exp(sum(log(1+post_r{i}(:,:,t)),1))'.*weight.(fnames{i})(:,:,t);
            %             reb_w(kk,:)=sum(abs(temp./sum(temp,1)-weight.(fnames{i})(:,:,t)),1);
            %             end    
            % 每隔t天 平衡一次投资组合
            T=10;
            if t<=size(post_r{1},3)-T
                        temp=exp(sum(log(1+post_r{1}(1:T,:,t)),1))'.*weight.(fnames{i})(:,:,t);
                        reb_w(kk,:)=sum(abs(temp./sum(temp,1)-weight.(fnames{i})(:,:,t+T)),1);
                        kk=kk+1;
            end
        end
        if ~isempty(reb_w)
        rownames=cellfun(@(x,y) [x,'_',y],repmat(fnames(i),size(rolling_names,2),1),rolling_names','UniformOutput',false);
        perf=[perf;table(mean(squeeze(mean(realized_r,1)./std(realized_r,1)),1),...
            mean(squeeze(std(realized_r,1)),1),...
            mean(squeeze(mean(realized_r,1)),1),...
            mean(reb_w,1)','VariableNames',{'sharpe','std','m_r','turnover'},'rownames',rownames)];
        perf_detail.(fnames{i})={'sharpe','std','m_r','turnover';...
            squeeze(mean(realized_r,1)./std(realized_r,1))',squeeze(std(realized_r,1))',squeeze(mean(realized_r,1))',reb_w};
        else 
        rownames=cellfun(@(x,y) [x,'_',y],repmat(fnames(i),size(rolling_names,2),1),rolling_names','UniformOutput',false);
        perf=[perf;table(mean(squeeze(mean(realized_r,1)./std(realized_r,1)),1),...
            mean(squeeze(std(realized_r,1)),1),...
            mean(squeeze(mean(realized_r,1)),1),'VariableNames',{'sharpe','std','m_r'},'rownames',rownames)];
        perf_detail.(fnames{i})={'sharpe','std','m_r';...
            squeeze(mean(realized_r,1)./std(realized_r,1))',squeeze(std(realized_r,1))',squeeze(mean(realized_r,1))'};
        end
end
end


%% main2
% 提前估计omegainv
% 参数
rolling_names={'day'};
rolling_window=[240]; % 不同步长 日 周 月
train_size=2; % 是rolling_window的几倍？
day=[1]; % 不同步长分别对应多少天
eva_window=[120]; % 评估步长
interval=60; 
strategy={'MTL','GMV','MV','TZ','EW','BS'}; %
MTL_substr={'MTL_GMV','MTL_MV'}; %
%%%%%%%%%%%% 第一步：提前估计omegainv %%%%%%%%%%%
pct_data=0.5; % 用来估计omegainv的数据比例
% 读取数据
fname=fieldnames(data);
tic
for i1=1:size(fname,1)
    temp=data(2).(fname{i1});
    temp(round(size(temp,1)*pct_data)+1:end,:)=[]; % 删去第二步的数据
    
    total_cic=floor((size(temp,1)-rolling_window)/interval)+1; % 循环次数
    pre_weight=struct;
    for i=1:total_cic
        % 将日数据矩阵转换为3D阵以便于转化为不同频率的数据
        est_r=temp((i-1)*interval+1:(i-1)*interval+rolling_window,:);
        for j=1:size(MTL_substr,2)
            pre_weight.(MTL_substr{j}(5:end))(:,i)=Port_weight_calcu([],{est_r},MTL_substr{j}(5:end),[],[]);
        end
    end
    Omegainv.(fname{i1})(1,1)=mean(var(pre_weight.GMV,0,2));
    Omegainv.(fname{i1})(2,2)=mean(var(pre_weight.MV,0,2));
    for i=1:size(temp,2)
        temp=cov(pre_weight.GMV(i,:),pre_weight.MV(i,:));
        nondiag(i)=temp(1,2);
    end
    Omegainv.(fname{i1})(1,2)=mean(nondiag);
    Omegainv.(fname{i1})(2,1)=mean(nondiag);
end
%%%%%%%%%% 第二步：插入估计好的omegainv 进行样本外投资 %%%%%%%%%%%
% 读取数据
fname=fieldnames(data);
date(1:round(size(temp,1)*pct_data),:)=[]; % 删去第1步的数据
tic
for i1=1:size(fname,1)
    temp=data(2).(fname{i1});
    temp(1:round(size(temp,1)*pct_data),:)=[]; % 删去第1步的数据
    % 初始化
    post_r={};
    date_est=[];
    weight={};
    % 初始化进度条
    %     tt=1;
    total_cic=size(temp,1)-eva_window-train_size*(rolling_window+eva_window)+1;
    lambda_log=[];
    weight_MTL=[];
    weight_normal=[];
    %     profile on
    parfor_progress(size([1:total_cic],2));
    num_loop=1;
    omegainv={};
    err={};
    parfor i=1:total_cic  
        pause(rand); % Replace with real code
        parfor_progress;
        if mod(i-1,interval)~=0
            continue
        end
        t=i+train_size*(rolling_window+eva_window)-1;
        est_r={};
        Y={};    
        % 训练,寻找最优的超参数
        test_r=temp(t-(train_size*(rolling_window+eva_window))+1:t,:);% 训练集
        lambda_reuslt=[];
        for stra_num=find(cellfun(@(x) strcmp(x(1:2),'MT'),strategy,'UniformOutput',true)==1) % 只对MTL进行训练            
            for num_train=1:train_size
                temp_test=test_r((num_train-1)*(rolling_window+eva_window)+1:num_train*(rolling_window+eva_window),:);
                train_weight=[];
                perf=[];
                % 先令lambda_1为0，寻找sharpe率最优的lambda_2
                for para_searc=1:50 % 50
                    lambda_1=0;
                    lambda_2(para_searc)=0+2e-05*(para_searc-1);   % 2e-05
                    train_weight.(strategy{stra_num})(:,:,1)=Port_weight_calcu({temp_test(1:rolling_window,:)},strategy{stra_num},lambda_1,lambda_2(para_searc)); % 得到投资权重
                    [perf(para_searc,:),~]=Port_evaluation_train(train_weight,{temp_test(rolling_window+1:rolling_window+eva_window,:)},rolling_names); % 得到评估
                end
                % 再固定lambda_2，寻找最优的lambda_1
                [~,loc]=max(sum(perf,2));
                lambda_2=lambda_2(loc);
                for para_searc=1:50 % 50
                    lambda_1(para_searc)=0+2e-05*(para_searc-1);   % 2e-05
                    train_weight.(strategy{stra_num})(:,:,1)=Port_weight_calcu({temp_test(1:rolling_window,:)},strategy{stra_num},lambda_1(para_searc),lambda_2); % 得到投资权重
                    [perf(para_searc,:),~]=Port_evaluation_train(train_weight,{temp_test(rolling_window+1:rolling_window+eva_window,:)},rolling_names); % 得到评估
                end
                [~,loc]=max(sum(perf,2));
                lambda_1=lambda_1(loc);
                lambda_reuslt(num_train,:)=[lambda_1 lambda_2];
            end
        end
        temp_lamda=[mean(lambda_reuslt,1)];
        lambda_log=[lambda_log;temp_lamda];
        lambda_1=temp_lamda(1,1);
        lambda_2=temp_lamda(1,2);
        %         lambda_1=lambda_log(num_loop,1);
        %         lambda_2=lambda_log(num_loop,2);
        %
        % 将日数据矩阵转换为3D阵以便于转化为不同频率的数据
        k=1;
        est_r=exp(squeeze(sum(permute(reshape(log(1+temp(t-rolling_window(k)*day(k)+1:t,:))',[],day(k),rolling_window(k)),[2 1 3]),1))')-1;
        Y=est_r*inv(est_r'*est_r)*ones(size(temp,2),1);
        date_est=[date_est;{date(t,:)}];
     %%%%%%%%% 输入投资组合策略，得到权重 %%%%%%%%%%
        for stra_num=1:size(strategy,2)
            if strcmp(strategy{stra_num},'MTL')
                [temp_w]=Port_weight_calcu(Y,{est_r},strategy{stra_num},lambda_1,lambda_2);
                weight_MTL=[weight_MTL temp_w];
            else
                [temp_w]=Port_weight_calcu(Y,{est_r},strategy{stra_num},lambda_1,lambda_2);
                weight_normal=[weight_normal temp_w];
            end
        end       
        %     num_loop=num_loop+1;
    end 
    for ii=1:total_cic
        if mod(ii-1,interval)~=0
            continue
        end
        t=ii+train_size*(rolling_window+eva_window)-1;
        post_r=[post_r;{temp(t+1:t+eva_window(1)*day(1),:)}];
    end
    % 分别存储不同策略的权重
    kk_1=1;
    kk_2=1;
    for num_stra=1:size(strategy,2)
        if strcmp(strategy{num_stra},'MTL')
            for j=1:size(MTL_substr,2)
                weight.(MTL_substr{j})=weight_MTL(:,j:2:end);
            end
        else
            weight.(strategy{num_stra})=weight_normal(:,kk_2:(size(strategy,2)-1):end);
            kk_2=kk_2+1;
        end
    en   
    %%%%%%%%%%%% 评估投资策略 %%%%%%%%%%
    [performance.(fname{i1}),perf_detail.(fname{i1}),total_r.(fname{i1})]=Port_evaluation(weight,post_r,rolling_names); %
    clear post_r
    save(['I:\工作台\临时数据\interval_',num2str(interval),'\给定omega\lam1和lam2均不固定\parfor_result_',fname{i1},'.mat'])
end
toc
function [w]=Port_weight_calcu(Y,est_r,strategy_type,lambda_1,lambda_2,Omega)
% 计算权重
switch strategy_type
    case 'MTL'
        [w]=MTL(est_r,lambda_1,lambda_2,Omega);
    case 'GMV'
        opts = optimoptions('quadprog','Display','off');
        for i=1:size(est_r,2)
            V=cov(est_r{i}); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
            w(:,i)=quadprog((V+V')/2,[],[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),[],ones(size(V,1),1)./size(V,1),opts);
        end
    case 'MV'
        opts = optimoptions('quadprog','Display','off');
        for i=1:size(est_r,2)
            V=cov(est_r{i}); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
            mu=mean(est_r{i},1);
            w(:,i)=quadprog((V+V')/4,-mu',[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),[],ones(size(V,1),1)./size(V,1),opts);
        end
    case 'TZ' % Tu and Zhou (2011)
        gamma=1;
        opts = optimoptions('quadprog','Display','off');
        for i=1:size(est_r,2)
            T=size(est_r{i},1);
            V=cov(est_r{i}); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
            num_asset=size(V,1);
            mu=mean(est_r{i},1);
            mv_w=quadprog((V+V')/4,-mu',[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),[],ones(size(V,1),1)./size(V,1),opts); % mv
            eq_w=ones(size(V,1),1)./size(V,1); % EW
            theta=mu*inv(V)*mu';
            pi1=eq_w'*V*eq_w-(2/gamma)*eq_w'*mu'+(1/gamma^2)*theta^2;
            c=(T-2)*(T-num_asset-2)/((T-num_asset-1)*(T-num_asset-4));
            pi2=(1/gamma^2)*(c-1)*theta^2+(c/gamma^2)*(num_asset/T);
            delta=pi1/(pi1+pi2);
            w(:,i)=(1-delta).*eq_w+delta.*mv_w; % 组合权重
        end
        
    case 'EW'
        for i=1:size(est_r,2)
            num_asset=size(est_r{i},2); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
            w(:,i)=ones(num_asset,1)./num_asset;
        end
    case 'BS'
        gamma=1;
        opts = optimoptions('quadprog','Display','off');
        for i=1:size(est_r,2)
            T=size(est_r{i},1);
            V=cov(est_r{i}); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
            num_asset=size(V,1);
            re=mean(est_r{i},1)';
            Y0=ones(1,num_asset)*(T-1)*inv(V)/(T-num_asset-2)*re/(ones(1,num_asset)*(T-1)*inv(V)/(T-num_asset-2)*ones(num_asset,1)); % 收益率的shrinkage target
            si=(num_asset+2)/((num_asset+2)+(re-Y0*ones(num_asset,1))'*T*(T-1)*inv(V)/(T-num_asset-2)*(re-Y0*ones(num_asset,1))); % shrinkage intensity
            mu=(1-si)*re+si*ones(num_asset,1)*Y0; % 收益率的shrinkage估计量
            lambda=(num_asset+2)/((re-Y0*ones(num_asset,1))'*(T-1)*inv(V)/(T-num_asset-2)*(re-Y0*ones(num_asset,1)));
            % 协方差的shrinkage估计量
            coVar=((T-1)*V/(T-num_asset-2))*(1+1/(T+lambda))+(lambda/(T*(T+1+lambda)))*ones(num_asset,1)*ones(1,num_asset)/(ones(1,num_asset)*(T-1)*inv(V)/(T-num_asset-2)*ones(num_asset,1));
            w=quadprog(gamma.*coVar,-mu,[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),ones(size(V,1),1),ones(size(V,1),1)./size(V,1),opts);
        end       
end
end
function [perf,perf_detail,total_r]=Port_evaluation(weight,post_r,rolling_names)
perf=table;
% 统计投资组合的样本外绩效
fnames=fieldnames(weight);
for i=1:size(fnames,1)
    realized_r=[];
    reb_w=[];
    kk=1;
    t=1;
    for t=1:size(post_r,1)
        realized_r(:,:,t)=post_r{t}*weight.(fnames{i})(:,t);
        total_r.(fnames{i})(:,t)=exp(sum(log(1+realized_r(:,:,t))))-1;
        %         sum_realized_post_r=
        % 每隔一个评估期间 重新平衡投资组合
        if t<=size(post_r,1)-1 && size(post_r,1)>1
            temp=exp(sum(log(1+post_r{t}),1))'.*weight.(fnames{i})(:,t);
            reb_w(kk,:)=sum(abs(temp./sum(temp,1)-weight.(fnames{i})(:,t+1)),1);
            kk=kk+1;
        end       
        % 每隔t天 平衡一次投资组合
        %         T=10;
        %         if t<=size(post_r,1)-T
        %             temp=exp(sum(log(1+post_r{t}(1:T,:)),1))'.*weight.(fnames{i})(:,t);
        %             reb_w(kk,:)=sum(abs(temp./sum(temp,1)-weight.(fnames{i})(:,t+T)),1);
        %             kk=kk+1;
        %         end
    end
    if ~isempty(reb_w)
        rownames=cellfun(@(x,y) [x,'_',y],repmat(fnames(i),size(rolling_names,2),1),rolling_names','UniformOutput',false);
        perf=[perf;table(mean(squeeze(mean(realized_r,1)./std(realized_r,1)),1),...
            mean(squeeze(std(realized_r,1)),1),...
            mean(squeeze(mean(realized_r,1)),1),...
            mean(reb_w,1)','VariableNames',{'sharpe','std','m_r','turnover'},'rownames',rownames)];
        perf_detail.(fnames{i})={'sharpe','std','m_r','turnover';...
            squeeze(mean(realized_r,1)./std(realized_r,1))',squeeze(std(realized_r,1))',squeeze(mean(realized_r,1))',reb_w};
    else
        perf=mean(squeeze(mean(realized_r,1)./std(realized_r,1)),1);
        perf_detail=[];
    end
end
end
function [perf,perf_detail]=Port_evaluation_train(weight,post_r,rolling_names)
% 统计投资组合的样本外绩效
for i=1:size(weight.MTL,2)
    realized_r=[];
    for t=1:size(post_r,1)
        realized_r(:,:,t)=post_r{t}*weight.MTL(:,i);
    end
    perf(1,i)=mean(squeeze(mean(realized_r,1)./std(realized_r,1)),1);
    perf_detail=[];
end
end

%% MTRL
function [w,err]=MTL(Y,X,lambda_1,lambda_2,tol)
% X <------ cell数组，包含m个数据集
m=size(X,2); % 数据集个数
n=size(X{1},2); % 求解向量的维度
% 分两步求解优化问题
err=1;
% tol=1e-04;
XY=zeros(n*m,m);
XX=zeros(n*m,n*m);
for i=1:m
    %    XY((i-1)*n+1:i*n,i)=-2*X{i}'*Y{i}/size(Y{i},1); % 回归形式
    %    XX((i-1)*n+1:i*n,(i-1)*n+1:i*n)=X{i}'*X{i}/size(X{i},1); % 回归形式
    XX((i-1)*n+1:i*n,(i-1)*n+1:i*n)=cov(X{i}); % 直接使用二次型
end

omegainv=eye(m)/m; % Omega的初始值
k=1;
opts = optimoptions('quadprog','Display','off');
while err>=tol
    % 第一步：omegainv假设为常数
    %     H=XX+lambda_1/2*eye(m*n)+lambda_2/2*kron(omegainv,eye(n)); % 回归形式
    %     vec_w=quadprog((H+H')/2,XY*ones(m,1),[],[],[],[],zeros(n*m,1),[],zeros(n*m,1),opts); % 回归形式
    H=XX+lambda_1/2*eye(m*n)+lambda_2/2*kron(omegainv,eye(n)); % 直接使用二次型
    vec_w=quadprog((H+H')/2,[],[],[],kron(eye(m),ones(1,n)),ones(m,1),zeros(n*m,1),[],zeros(n*m,1),opts); % 直接使用二次型
    % 第2步：已知w，求omegainv的解析解
    temp=(reshape(vec_w,n,m)'*reshape(vec_w,n,m))^(1/2)/trace((reshape(vec_w,n,m)'*reshape(vec_w,n,m))^(1/2));
    err(k)=sqrt(sum(sum((omegainv-temp).^2,1),2));
    k=k+1;
    omegainv=temp;
    if k>1000
        break
    end
end

w=reshape(vec_w,n,m);
% w=w./sum(w,1);
end

%% facotr
% 数据集1:因子组合
di = dir('H:\工作台\系统-MTRL-修改-2020-12-30\美股数据\因子组合收益率');
di(1:2) = [];
for i = 1:size(di,1)
    [~,~,da ] = xlsread([di(1).folder,'\',di(i).name]);
    temp_re = cell2mat(da(2:end,2:end)) * 0.01;
    loc = find(sum(isnan(temp_re),1) > 0 | sum(temp_re < -0.9,1));
    temp_re(:,loc) = [];
    temp_date = da(2:end,1);
    data(1).(di(i).name(1:end-4)) = temp_date;
    data(2).(di(i).name(1:end-4)) = temp_re;
end
save('H:\工作台\系统-MTRL-修改-2020-12-30\美股数据\临时数据\data.mat','data')
% 参数
rolling_names={'day'};
rolling_window=[240]; % 不同步长 日 周 月
train_size=2; % 是rolling_window的几倍？
day=[1]; % 不同步长分别对应多少天
eva_window=[120]; % 评估步长
interval=120; % 评估的间隔 120结果已有
strategy={'MTL','GMV','MV','TZ','EW','BS'}; %
MTL_substr={'MTL_GMV','MTL_MV'}; %
% 读取数据
fname=fieldnames(data);
tic
for i1=1:size(fname,1)
    temp=data(2).(fname{i1}); 
    % 初始化
    post_r={};
    date_est=[];
    weight={};
    % 初始化进度条
    %     tt=1;
    total_cic=size(temp,1)-eva_window-train_size*(rolling_window+eva_window)+1;
    lambda_log=[];
    weight_MTL=[];
    weight_normal=[];
    %     profile on
    parfor_progress(size([1:total_cic],2));
    num_loop=1;
    omegainv={};
    err={};    
%     numIterations = total_cic;
%     ppm = ParforProgressbar(numIterations,'showWorkerProgress',true,'progressBarUpdatePeriod',3,'title',['第',num2str(i1),'个数据集']); 
%     pauseTime = 60/numIterations;
    for i=1:total_cic
%     pause(pauseTime);
    % increment counter to track progress
%     ppm.increment();
        if mod(i-1,interval)~=0
            continue
        end
        t=i+train_size*(rolling_window+eva_window)-1;       
        est_r={};
        Y={};   
        % 训练,寻找最优的超参数
        test_r=temp(t-(train_size*(rolling_window+eva_window))+1:t,:);% 训练集
        lambda_reuslt=[];
        for stra_num=find(cellfun(@(x) strcmp(x(1:2),'MT'),strategy,'UniformOutput',true)==1) % 只对MTL进行训练
            
            for num_train=1:train_size
                temp_test=test_r((num_train-1)*(rolling_window+eva_window)+1:num_train*(rolling_window+eva_window),:);
                train_weight=[];
                perf=[];
                % 先令lambda_1为0，寻找sharpe率最优的lambda_2
                for para_searc=1:50 % 50
                    lambda_1=0;
                    lambda_2(para_searc)=0+2e-05*(para_searc-1);   % 2e-05
                    train_weight.(strategy{stra_num})(:,:,1)=Port_weight_calcu(Y,{temp_test(1:rolling_window,:)},strategy{stra_num},lambda_1,lambda_2(para_searc)); % 得到投资权重
                    [perf(para_searc,:),~]=Port_evaluation_train(train_weight,{temp_test(rolling_window+1:rolling_window+eva_window,:)},rolling_names); % 得到评估
                end
                % 再固定lambda_2，寻找最优的lambda_1
                [~,loc]=max(sum(perf,2));
                lambda_2=lambda_2(loc);
                for para_searc=1:50 % 50
                    lambda_1(para_searc)=0+2e-05*(para_searc-1);   % 2e-05
                    train_weight.(strategy{stra_num})(:,:,1)=Port_weight_calcu(Y,{temp_test(1:rolling_window,:)},strategy{stra_num},lambda_1(para_searc),lambda_2); % 得到投资权重
                    [perf(para_searc,:),~]=Port_evaluation_train(train_weight,{temp_test(rolling_window+1:rolling_window+eva_window,:)},rolling_names); % 得到评估
                end
                [~,loc]=max(sum(perf,2));
                lambda_1=lambda_1(loc);
                lambda_reuslt(num_train,:)=[lambda_1 lambda_2];
            end
        end
        temp_lamda=[mean(lambda_reuslt,1)];
        lambda_log=[lambda_log;temp_lamda];
        lambda_1=temp_lamda(1,1);
        lambda_2=temp_lamda(1,2);
        % 将日数据矩阵转换为3D阵以便于转化为不同频率的数据
        k=1;
        est_r=exp(squeeze(sum(permute(reshape(log(1+temp(t-rolling_window(k)*day(k)+1:t,:))',[],day(k),rolling_window(k)),[2 1 3]),1))')-1;
        Y=est_r*inv(est_r'*est_r)*ones(size(temp,2),1);
%         date_est=[date_est;{date(t,:)}];
        
        %%%%%%%%% 输入投资组合策略，得到权重 %%%%%%%%%%
        for stra_num=1:size(strategy,2)
            if strcmp(strategy{stra_num},'MTL')
                [temp_w,vargout1,vargout2]=Port_weight_calcu(Y,{est_r},strategy{stra_num},lambda_1,lambda_2);
                err=[err;{vargout1{1,1}}];
                omegainv=[omegainv;{vargout2{1,1}}];
                weight_MTL=[weight_MTL temp_w];
            else
                [temp_w]=Port_weight_calcu(Y,{est_r},strategy{stra_num},lambda_1,lambda_2);
                weight_normal=[weight_normal temp_w];
            end
        end  
%     num_loop=num_loop+1; 
    end 
    for ii=1:total_cic
        if mod(ii-1,interval)~=0
            continue
        end
        t=ii+train_size*(rolling_window+eva_window)-1;
        post_r=[post_r;{temp(t+1:t+eva_window(1)*day(1),:)}];
        
    end
%     load(['I:\工作台\2020-10-13-系统工程实践-MTL+投资组合\临时数据\interval_',num2str(interval),'\lam1和lam2均不固定\parfor_result_',fname{i1},'.mat'],...
%         'weight_MTL','weight_normal')
    % 分别存储不同策略的权重
    kk_1=1;
    kk_2=1;
    for num_stra=1:size(strategy,2)
        if strcmp(strategy{num_stra},'MTL')
            for j=1:size(MTL_substr,2)
                weight.(MTL_substr{j})=weight_MTL(:,j:2:end);
            end
        else
            weight.(strategy{num_stra})=weight_normal(:,kk_2:(size(strategy,2)-1):end);
            kk_2=kk_2+1;
        end
    end  
    %%%%%%%%%%%% 评估投资策略 %%%%%%%%%%
    [performance.(fname{i1}),perf_detail.(fname{i1}),total_r.(fname{i1})]=Port_evaluation(weight,post_r,rolling_names); %
%     clear post_r
end
     save(['H:\工作台\系统-MTRL-修改-2020-12-30\美股数据\投资结果\interval_',num2str(interval),'.mat'],...
         'performance','perf_detail','total_r')
toc
function [w,varargout]=Port_weight_calcu(Y,est_r,strategy_type,lambda_1,lambda_2)
% 计算权重
switch strategy_type
    case 'MTL'
        [w,err,omegainv]=MTL_strategies(Y,est_r,lambda_1,lambda_2,1e-07);
        varargout{1}={err};
        varargout{2}={omegainv};
    case 'GMV'
        opts = optimoptions('quadprog','Display','off');
        for i=1:size(est_r,2)
            V=cov(est_r{i}); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
            w(:,i)=quadprog((V+V')/2,[],[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),[],ones(size(V,1),1)./size(V,1),opts);
        end
    case 'MV'
        opts = optimoptions('quadprog','Display','off');
        for i=1:size(est_r,2)
            V=cov(est_r{i}); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
            mu=mean(est_r{i},1);
            w(:,i)=quadprog((V+V')/4,-mu',[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),[],ones(size(V,1),1)./size(V,1),opts);
        end
    case 'TZ' % Tu and Zhou (2011)
        gamma=1;
        opts = optimoptions('quadprog','Display','off');
        for i=1:size(est_r,2)
            T=size(est_r{i},1);
            V=cov(est_r{i}); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
             num_asset=size(V,1);
            mu=mean(est_r{i},1);
            mv_w=quadprog((V+V')/4,-mu',[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),[],ones(size(V,1),1)./size(V,1),opts); % mv
            eq_w=ones(size(V,1),1)./size(V,1); % EW
            theta=mu*inv(V)*mu';
            pi1=eq_w'*V*eq_w-(2/gamma)*eq_w'*mu'+(1/gamma^2)*theta^2;
            c=(T-2)*(T-num_asset-2)/((T-num_asset-1)*(T-num_asset-4));
            pi2=(1/gamma^2)*(c-1)*theta^2+(c/gamma^2)*(num_asset/T);
            delta=pi1/(pi1+pi2);
            w(:,i)=(1-delta).*eq_w+delta.*mv_w; % 组合权重
        end
        
    case 'EW'
        for i=1:size(est_r,2)
            num_asset=size(est_r{i},2); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
            w(:,i)=ones(num_asset,1)./num_asset;
        end
    case 'BS'
        gamma=1;
        opts = optimoptions('quadprog','Display','off');
        for i=1:size(est_r,2)
            T=size(est_r{i},1);
            V=cov(est_r{i}); % *(size(est_r{i},1)-1)/(size(est_r{i},1))
             num_asset=size(V,1);
            re=mean(est_r{i},1)';
        Y0=ones(1,num_asset)*(T-1)*inv(V)/(T-num_asset-2)*re/(ones(1,num_asset)*(T-1)*inv(V)/(T-num_asset-2)*ones(num_asset,1)); % 收益率的shrinkage target
        si=(num_asset+2)/((num_asset+2)+(re-Y0*ones(num_asset,1))'*T*(T-1)*inv(V)/(T-num_asset-2)*(re-Y0*ones(num_asset,1))); % shrinkage intensity
        mu=(1-si)*re+si*ones(num_asset,1)*Y0; % 收益率的shrinkage估计量
        lambda=(num_asset+2)/((re-Y0*ones(num_asset,1))'*(T-1)*inv(V)/(T-num_asset-2)*(re-Y0*ones(num_asset,1)));
        % 协方差的shrinkage估计量
        coVar=((T-1)*V/(T-num_asset-2))*(1+1/(T+lambda))+(lambda/(T*(T+1+lambda)))*ones(num_asset,1)*ones(1,num_asset)/(ones(1,num_asset)*(T-1)*inv(V)/(T-num_asset-2)*ones(num_asset,1));
        w=quadprog(gamma.*coVar,-mu,[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),ones(size(V,1),1),ones(size(V,1),1)./size(V,1),opts);
        end
end
end


%% result
%  返修增加：用美股的FF3因子组合作为稳健性结果
load('H:\工作台\美股数据\投资结果\interval_120.mat')
fnames=fieldnames(performance);
for i=1:size(fnames,1)
    % 提取7个策略的样本外实现的收益率
%     temp_total_r=structfun(@(x) x{2,3},perf_detail.(fnames{i}),'UniformOutput',false); % 每个评估区间的收益率均值
    temp_total_r=total_r.(fnames{i}); % 每个评估区间的总收益率
    re=[];
    ffnames=fieldnames(temp_total_r);
    for j=1:size(ffnames,1)
        re=[re;temp_total_r.(ffnames{j})];
    end
    % 计算Jobson-Korkie的Z值
    base=[1 2]'; % 基准策略
    target=[3 4]'; % 基准策略对应的目标策略
    mu_base=mean(re(base,:),2);
    std_base=std(re(base,:),0,2);
    mu_target=mean(re(target,:),2);
    std_target=std(re(target,:),0,2);
    for j=1:size(base,1)
        temp=cov(re(base(j),:),re(target(j),:));
        covar(j)=temp(1,2);
        rho=(2*std_target(j)^2*std_base(j)^2-2*std_base(j)*std_target(j)*covar(j)+...
            0.5*mu_target(j)^2*std_base(j)^2+0.5*mu_base(j)^2*std_target(j)^2-...
            mu_base(j)*mu_target(j)*covar(j)^2/(std_target(j)*std_base(j)))/size(re,2);
        Z(j)=(std_base(j)*mu_target(j)-std_target(j)*mu_base(j))/sqrt(rho);
        if Z(j)>=0
        pvalue(j)=1-normcdf(Z(1)); % 单边p值 H0:GMV-MTL_GMV=0
        else
        pvalue(j)=normcdf(Z(j)); % 单边p值 H0:GMV-MTL_GMV=0    
        end
    end
    % 表格
    temp_result=performance.(fnames{i});
    temp_new=table2array(temp_result)';
    result=array2table([temp_new(1,:);[Z nan(1,size(temp_new,2)-size(pvalue,2))];[pvalue nan(1,size(temp_new,2)-size(pvalue,2))];temp_new(2:end,:)]);
    result.Properties.VariableNames=cellfun(@(x) x(1:end-4),temp_result.Properties.RowNames,'UniformOutput',false);
    result.Properties.RowNames={'SR','Z-value','p','Std.','Avg.(r)','Turnover'};
    table2latex({result},fnames{i},'C:\Users\qianp\OneDrive\材料\论文相关\表格\robust_US_interval_120.tex')
end

%% portfolio strategy
function [weight,StrategyType] = PortfolioStrategy(reEst,coVar,T,varargin)
% coVar-协方差估计量 
% reEst-收益率均值估计量(num_asset*1)向量
% vargin{1,1} 为自定义各策略的组合权重 需要1 * num_strategies 
% 若vargin为空，那么默认为1/n
% T为样本数量
gamma=3;
opts = optimoptions('quadprog','Display','off');
weight=[];
weight = [weight GMV(coVar,opts,'no')];
weight = [weight MV(reEst,coVar,gamma,opts)];
weight = [weight TzCmb(reEst,coVar,gamma,T,opts)]; 
% temp=KZCmb(reEst,coVar,gamma,T);
% weight = [weight temp./sum(temp)]; 
% weight{1,4} = KZCmb(reEst,coVar,gamma,T); 
% weight{1,4} =weight{1,4}./sum(weight{1,4}); % KZ策略默认有无风险资产
weight = [weight BS(reEst,coVar,gamma,T,opts)];
weight = [weight ones(size(coVar,1),1)./size(coVar,1)]; % 等权重策略
% 混合权重
if ~ isempty( varargin ) % 是否输入了策略之间的自定义权重？
weight =[weight (varargin{1,1}*weight')'];
else
    weight =[weight (ones(1,size(weight,2))/size(weight,2)*weight')'];
end
StrategyType={'GMV','MV','TZ','BS','EW','MIXED'}; % 'KZ',
end
% 以下为投资组合策略
% 1、Global Minimum Variance Portfolio
function w = GMV(V,opts,SHORTSELL)
switch SHORTSELL
    case 'no'
        w=quadprog(V*1000000,[],[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),ones(size(V,1),1),ones(size(V,1),1)./size(V,1),opts);
        w((w<1e-8))=0;
        w((w>1-1e-8))=1; % 消除浮点误差
    case 'yes'
        w=quadprog(V*1000000,[],[],[],ones(size(V,1),1)',1,[],[],ones(size(V,1),1)./size(V,1),opts);
        % w((w<1e-8))=0;
        % w((w>1-1e-8))=1; % 消除浮点误差
end
end
% 2、Markowitz's mean-variance model
function w = MV(re,V,gamma,opts)
w=quadprog(gamma.*V*1000000,-re'*1000000,[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),ones(size(V,1),1),ones(size(V,1),1)./size(V,1),opts);
w((w<1e-8))=0;
w((w>1-1e-8))=1; % 消除浮点误差
end
% 3、Tu and Zhou (2011)'s combination of MV and Equally-weighted stragtegies
function w = TzCmb(re,V,gamma,T,opts) 
num_asset=size(V,1);
mv_w=quadprog(gamma.*V*1000000,-re'*1000000,[],[],ones(num_asset,1)',1,zeros(num_asset,1),ones(num_asset,1),ones(num_asset,1)./num_asset,opts); % MV
mv_w((mv_w<1e-8))=0;
mv_w((mv_w>1-1e-8))=1; % 消除浮点误差
eq_w=ones(size(V,1),1)./size(V,1); % EW
% Tu和Zhou文中的系数
theta=re'*inv(V)*re;
pi1=eq_w'*V*eq_w-(2/gamma)*eq_w'*re+(1/gamma^2)*theta^2;
c=(T-2)*(T-num_asset-2)/((T-num_asset-1)*(T-num_asset-4));
pi2=(1/gamma^2)*(c-1)*theta^2+(c/gamma^2)*(num_asset/T);
delta=pi1/(pi1+pi2);
w=(1-delta).*eq_w+delta.*mv_w; % 组合权重
end
% 4、Kan and Zhou (2007)'s three-fund optimal portfolio weights
function w=KZCmb(re,V,gamma,T) 
num_asset=size(V,1);
mu_g=(re'*inv(V)*ones(num_asset,1))./(ones(1,num_asset)*inv(V)*ones(num_asset,1));
psi2=(re-mu_g*ones(num_asset,1))'*inv(V)*(re-mu_g*ones(num_asset,1));
hat_psi2_a=((T-num_asset-1)*psi2*-num_asset+1)/T+(2*psi2^((num_asset-1)/2)*(1+psi2)^(-(T-2)/2))./(T*betainc(psi2/(psi2+1),(num_asset-1)/2,(T-num_asset+1)/2)*beta((num_asset-1)/2,(T-num_asset+1)/2));
c3=(T-num_asset-1)*(T-num_asset-4)/(T*(T-2));
w=(c3/gamma)*(hat_psi2_a/(hat_psi2_a+num_asset/T)*inv(V)*re+((num_asset/T)/(hat_psi2_a+(num_asset/T)))*mu_g*inv(V)*ones(num_asset,1));
w=w./sum(w); % 归一化为1 默认仅为风险资产权重
end
% 5、Jorion (1986)'s Bayes-Stein Shrinkage Porfolio Strategy
function w=BS(re,V,gamma,T,opts) % 公式参考自Kan and Zhou (2007)
num_asset=size(V,1);
Y0=ones(1,num_asset)*(T-1)*inv(V)/(T-num_asset-2)*re/(ones(1,num_asset)*(T-1)*inv(V)/(T-num_asset-2)*ones(num_asset,1)); % 收益率的shrinkage target
si=(num_asset+2)/((num_asset+2)+(re-Y0*ones(num_asset,1))'*T*(T-1)*inv(V)/(T-num_asset-2)*(re-Y0*ones(num_asset,1))); % shrinkage intensity 
mu=(1-si)*re+si*ones(num_asset,1)*Y0; % 收益率的shrinkage估计量 
lambda=(num_asset+2)/((re-Y0*ones(num_asset,1))'*(T-1)*inv(V)/(T-num_asset-2)*(re-Y0*ones(num_asset,1)));
coVar=((T-1)*V/(T-num_asset-2))*(1+1/(T+lambda))+(lambda/(T*(T+1+lambda)))*ones(num_asset,1)*ones(1,num_asset)/(ones(1,num_asset)*(T-1)*inv(V)/(T-num_asset-2)*ones(num_asset,1));
w=quadprog(gamma.*coVar*1000000,-mu*1000000,[],[],ones(size(V,1),1)',1,zeros(size(V,1),1),ones(size(V,1),1),ones(size(V,1),1)./size(V,1),opts);
w((w<1e-8))=0;
w((w>1-1e-8))=1; % 消除浮点误差
end

%% MTL_strategies
function [w,err,omegainv]=MTL_strategies(Y,est_r,lambda_1,lambda_2,tol)
mu=mean(est_r{1},1);
V=cov(est_r{1});
m=2;% 任务个数
n=size(V,2); % 求解向量的维度
%% 分两步求解优化问题
err=1;
% tol=1e-04;
omegainv=eye(m)/m; % Omega的初始值
k=1;
opts = optimoptions('quadprog','Display','off');
while err>=tol
    % 第一步：omegainv假设为常数
%     H=XX+lambda_1/2*eye(m*n)+lambda_2/2*kron(omegainv,eye(n)); % 回归形式
%     vec_w=quadprog((H+H')/2,XY*ones(m,1),[],[],[],[],zeros(n*m,1),[],zeros(n*m,1),opts); % 回归形式
    H=kron(eye(m),0.5*V)+lambda_1/2*eye(m*n)+lambda_2/2*kron(omegainv,eye(n)); % 直接使用二次型 
    A=[ones(n,1);mu'];
    if ~isreal(H)
        vec_w=ones(m*n,1);
   else
    vec_w=quadprog((H+H')/2,-A,[],[],[],[],zeros(n*m,1),[],zeros(n*m,1),opts); % 直接使用二次型
    end   
    % 第2步：已知w，求omegainv的解析解
    temp=(reshape(vec_w,n,m)'*reshape(vec_w,n,m))^(1/2)/trace((reshape(vec_w,n,m)'*reshape(vec_w,n,m))^(1/2));
    err(k)=sqrt(sum(sum((omegainv-temp).^2,1),2));
    k=k+1;
    omegainv=temp;
    if k>1000
       break 
    end
end
w=reshape(vec_w,n,m);
w=w./sum(w,1);
end



