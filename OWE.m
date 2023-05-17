
%% Data Processing
clear
addpath(genpath('D:'));
di=dir('I:\工作\A股交易数据\A股1分钟数据');
di(1:2)=[];

% =========数据中包含的股票代码=========
stklist=[];
for i=1 :size(di,1)
    file=fopen([di(i).folder,'\',di(i).name]);
    data=textscan(file,'%d %f %f %d %f\n','Delimiter',',');
    stklist=unique([stklist;unique(data{1,1})]);
    fclose(file)
end
save('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\hf_stklist.mat','stklist')

% ========交易状态===========
load('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\hf_stklist.mat','stklist')
trading_graph=zeros(size(di,1),size(stklist,1));
trading_date=cell(size(di,1),1);
h = waitbar(0,'Please wait...');
t1=tic;
for i=1 :size(di,1)
    i
    trading_date{i,1}=di(i).name([1:4 6:7 9:10]);
    file=fopen([di(i).folder,'\',di(i).name]);
    data=textscan(file,'%d %f %f %d %f\n','Delimiter',',');
    temp_stklist=unique(data{1,1});
    trading_graph(i,:)=ismember(stklist,temp_stklist)';
    fclose(file);
    str=['calculating IVOL...',num2str(i),' /',num2str(size(di,1)),' , time elapses：',num2str(toc(t1)),' s'];
    waitbar(i/size(di,1),h,str);
end
save('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\hf_trading_status.mat','trading_graph','trading_date','stklist')
%% main.m
clear
addpath(genpath('D:\工作台\2019-08-15 （修改）\修改\程序相关'));
load('J:\工作\2019-08-15 （修改）\修改\程序相关\数据\hf_re_1st_10stks.mat')
% 清洗数据 计算RCOV-P-N-M
log_error=[];
Realized_matrix.RCOV=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
Realized_matrix.P=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
Realized_matrix.N=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
Realized_matrix.M=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
Realized_matrix.M1=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
Realized_matrix.M2=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
dbstop if error
for i=1:size(hf_re,3)
    temp=hf_re(:,:,i);
    temp(isnan(temp))=0;
    temp(isinf(temp))=0;
    if sum(sum(temp>0.1))>=1 || sum(sum(temp<-0.1))>=1
        log_error=[log_error;i];
    end
    temp=log(1+temp);
    temp_Pos=(abs(temp)+temp)/2;
    temp_Neg=(temp-abs(temp))/2;
    Realized_matrix.RCOV(:,:,i)=temp'*temp;
    Realized_matrix.P(:,:,i)=temp_Pos'*temp_Pos;
    Realized_matrix.N(:,:,i)=temp_Neg'*temp_Neg;
    Realized_matrix.M1(:,:,i)=temp_Neg'*temp_Pos;
    Realized_matrix.M2(:,:,i)=temp_Pos'*temp_Neg;
    Realized_matrix.M(:,:,i)=Realized_matrix.M1(:,:,i)+Realized_matrix.M2(:,:,i);
end
save('J:\工作\2019-08-15 （修改）\修改\程序相关\数据\hf_matrix_1st_10stks.mat','Realized_matrix','daily_re_date')
% HAR-RV
name={'RCOV','P','N'}; % 选择要进行预测的矩阵
% =========cholesky + vectorize=================
loc_vec=[]; % 上三角矩阵的索引
k=1;
for i=1:size(hf_re,2)
    loc_vec=[loc_vec;[1:k]'+size(hf_re,2)*(i-1)];
    k=k+1;
end
log_notpd=[];
X=struct;
for i=1:size(name,2)
    temp=getfield(Realized_matrix,name{i});
    temp_matrix=nan(size(hf_re,2)*(1+size(hf_re,2))/2,size(temp,3));
    for j=1:size(temp,3)
        % 正定性 将元素全为0的对应证券的方差用过去10日的日已实现方差代替
        if rank(temp(:,:,j))<size(hf_re,2)
            log_notpd=[log_notpd;i j];
            temp_loc=find(diag(temp(:,:,j))==0);
            temp(temp_loc,temp_loc,j)=mean(temp(temp_loc,temp_loc,j-10:j-1));
            % rank(temp(:,:,j))
        end
        temp_chol=chol(temp(:,:,j));
        temp_matrix(:,j)=temp_chol(loc_vec);
    end
    X=setfield(X,name{i},temp_matrix);
end
save('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\vech_chol_for_HAR_RV.mat','X','loc_vec')
%
% % =========RCOV P N 仅基于HAR-RV的样本外表现(基于fronbenius rmse)=================
fre={'day','week','biweek','month'}; % 四种不同期限的预测
fre_size=[1 5 10 20]; % 对应的长度
HAR_result=struct;
window_size=240;
for t=1:size(name,2)
    t
    temp=getfield(X,name{t});
    for i=1:size(fre,2)
        k=1;
        forecast_date=cell(size(stk_date,1)-fre_size(i)+1-window_size+1,1);
        FORE=nan(size(hf_re,2),size(hf_re,2),size(stk_date,1)-fre_size(i)-window_size+1);
        REAL=nan(size(hf_re,2),size(hf_re,2),size(stk_date,1)-fre_size(i)-window_size+1);
        rmse=nan(size(stk_date,1)-fre_size(i)-240+1,1);
        for j=window_size:size(stk_date,1)-fre_size(i) % 第一个预测 留一年作为初始训练集长度 此后将使用window_size的历史数据作为训练集 最后留出fre_size天作为最后一个预测区间
            forecast_date{k}=stk_date(j); %划分历史和未来的日期（下一日为未来）
            input=temp(:,1:j);
            % vech(chol(cov))预测值
            [beta,se,x_hat,~,t_stats,p_value]=HAR_RV_est(input,fre{i});
            [X_hat]=HAR_RV_fore(input,fre{i},beta);
            % chol(cov)预测值
            temp_fre=zeros(size(hf_re,2),size(hf_re,2));
            temp_fre(loc_vec)=X_hat;
            % cov预测值
            FORE(:,:,k)=temp_fre'*temp_fre;
            % 真实值
            temp_fre=zeros(size(hf_re,2),size(hf_re,2));
            temp_fre(loc_vec)=movmean(temp(:,j+1:j+fre_size(i)),fre_size(i),2,'Endpoints','discard'); %
            REAL(:,:,k)=temp_fre'*temp_fre;
            % RMSE
            rmse(k)=sum(sum((FORE(:,:,k)-REAL(:,:,k)).^2,1),2);
            k=k+1;
        end
        HAR_result=setfield(HAR_result,{1},[name{t},'_',fre{i}],FORE); % 字段第1行为预测协方差值
        HAR_result=setfield(HAR_result,{2},[name{t},'_',fre{i}],REAL); % 字段第2行为协方差真实值
        HAR_result=setfield(HAR_result,{3},[name{t},'_',fre{i}],rmse); % 字段第3行为对应rmse
        HAR_result=setfield(HAR_result,{4},[name{t},'_',fre{i}],forecast_date); % 字段第4行为日期
    end
end
stats=fieldnames(HAR_result);
stats(:,2)=num2cell(structfun(@(x) mean(x(~isnan(x))),HAR_result(3),'UniformOutput',true)'); % mean
stats(:,3)=num2cell(structfun(@(x) std(x(~isnan(x))),HAR_result(3),'UniformOutput',true)');  % std
save('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\result\HAR_RV.mat','HAR_result')

%% ERROR_calculator
function [ERROR_OWE]=ERROR_calculator(X,beta,fre) % 默认最后一个样本为最新
switch fre
    case 'day'
        X_tp1=reshape(X(:,21:end)',size(X(:,21:end),1)*size(X(:,21:end),2),1); % X_t plus 1
        X_t=reshape(X(:,20:end-1)',size(X(:,20:end-1),1)*size(X(:,20:end-1),2),1);  % X_t day
        temp_X=movmean(X(:,16:end-1),5,2,'Endpoints','discard'); % X_t week
        X_tw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,11:end-1),10,2,'Endpoints','discard'); % X_t bi-week
        X_tbw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,1:end-1),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        % [beta,bint,r,rint,stats] = regress(X_tp1,[ones(size(X_tp1,1),1) X_t X_tw X_tbw X_tm]);
        x_hat=reshape([ones(size(X_tp1,1),1) X_t X_tw X_tbw X_tm]*beta,size(X(:,21:end),2),size(X(:,21:end),1))'; % 还原
        y=reshape(X_tp1,size(X(:,21:end),2),size(X(:,21:end),1))';
    case 'week'
        temp_Y=movmean(X(:,21:end),5,2,'Endpoints','discard'); % X_t plus 1 week
        X_tp1=reshape(temp_Y',size(temp_Y,1)*size(temp_Y,2),1); % X_t plus 1
        temp_X=movmean(X(:,16:end-5),5,2,'Endpoints','discard'); % X_t week
        X_tw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,11:end-5),10,2,'Endpoints','discard'); % X_t bi-week
        X_tbw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,1:end-5),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        x_hat=reshape([ones(size(X_tp1,1),1) X_tw X_tbw X_tm]*beta,size(temp_Y,2),size(temp_Y,1))'; % 还原
        y=reshape(X_tp1,size(temp_Y,2),size(temp_Y,1))';
    case 'biweek'
        temp_Y=movmean(X(:,21:end),10,2,'Endpoints','discard'); % X_t plus 1 week
        X_tp1=reshape(temp_Y',size(temp_Y,1)*size(temp_Y,2),1); % X_t plus 1
        temp_X=movmean(X(:,11:end-10),10,2,'Endpoints','discard'); % X_t bi-week
        X_tbw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,1:end-10),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        x_hat=reshape([ones(size(X_tp1,1),1) X_tbw X_tm]*beta,size(temp_Y,2),size(temp_Y,1))'; % 还原
        y=reshape(X_tp1,size(temp_Y,2),size(temp_Y,1))';
    case 'month'
        temp_Y=movmean(X(:,21:end),20,2,'Endpoints','discard'); % X_t plus 1 week
        X_tp1=reshape(temp_Y',size(temp_Y,1)*size(temp_Y,2),1); % X_t plus 1
        temp_X=movmean(X(:,1:end-20),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        x_hat=reshape([ones(size(X_tp1,1),1) X_tm]*beta,size(temp_Y,2),size(temp_Y,1))'; % 还原
        y=reshape(X_tp1,size(temp_Y,2),size(temp_Y,1))';
end
ERROR_OWE=sum((y-x_hat).^2,1); % OWE在最新滚动样本上vech_chol_realizedCOV的mse
end

%% func_riskparity
function [f,g] = func_riskparity(x,V)
f=x'*V*x-sum(log(x));
if nargout > 1 %gradient required
    g=2*V*x-1./x;
end
end

%%  generate_realized_matrix
function [Realized_Matrix] = generate_realized_matrix(fre,hf_re,rcov_type)
% 生成不同类型的RCOV矩阵 RCOV P N
k=1;
log_error=[];
Realized_Matrix=nan(size(hf_re,2),size(hf_re,2),size([1:fre:size(hf_re,3)],2));
for i=1:fre:size(hf_re,3)
    temp=hf_re(:,:,i);
    temp(isnan(temp))=0;
    temp(isinf(temp))=0;
    if sum(sum(temp>0.1))>=1 || sum(sum(temp<-0.1))>=1
        log_error=[log_error;i];
    end
    temp=log(1+temp);
    switch rcov_type
        case 'RCOV'
            Realized_Matrix(:,:,k)=temp'*temp;
        case 'P'
            temp_Pos=(abs(temp)+temp)/2;
            Realized_Matrix(:,:,k)=temp_Pos'*temp_Pos;
        case 'N'
            temp_Neg=(temp-abs(temp))/2;
            Realized_Matrix(:,:,k)=temp_Neg'*temp_Neg;
    end
    k=k+1;
end
end

%% HAR_RV_est
function [beta,se,x_hat,y,t_stats,p_value]=HAR_RV_est(X,fre)
% X 的格式为 m*T m为vec后的元素个数  T为样本长度 从旧到新
% fre为频率 1（day） 5（week） 10（bi-week） 20(month)
switch fre
    case 'day'
        X_tp1=reshape(X(:,21:end)',size(X(:,21:end),1)*size(X(:,21:end),2),1); % X_t plus 1
        X_t=reshape(X(:,20:end-1)',size(X(:,20:end-1),1)*size(X(:,20:end-1),2),1);  % X_t day
        temp_X=movmean(X(:,16:end-1),5,2,'Endpoints','discard'); % X_t week
        X_tw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,11:end-1),10,2,'Endpoints','discard'); % X_t bi-week
        X_tbw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,1:end-1),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        % [beta,bint,r,rint,stats] = regress(X_tp1,[ones(size(X_tp1,1),1) X_t X_tw X_tbw X_tm]);
        [~,se,beta] = hac([X_t X_tw X_tbw X_tm],X_tp1,'type','HC','display','off');
        t_stats=beta./se;
        p_value=(1-normcdf(t_stats,0,1));
        x_hat=reshape([ones(size(X_tp1,1),1) X_t X_tw X_tbw X_tm]*beta,size(X(:,21:end),2),size(X(:,21:end),1))'; % 还原
        y=reshape(X_tp1,size(X(:,21:end),2),size(X(:,21:end),1))';
    case 'week'
        temp_Y=movmean(X(:,21:end),5,2,'Endpoints','discard'); % X_t plus 1 week
        X_tp1=reshape(temp_Y',size(temp_Y,1)*size(temp_Y,2),1); % X_t plus 1
        temp_X=movmean(X(:,16:end-5),5,2,'Endpoints','discard'); % X_t week
        X_tw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,11:end-5),10,2,'Endpoints','discard'); % X_t bi-week
        X_tbw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,1:end-5),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        % [beta,bint,r,rint,stats] = regress(X_tp1,[ones(size(X_tp1,1),1) X_t X_tw X_tbw X_tm]);
        [~,se,beta] = hac([X_tw X_tbw X_tm],X_tp1,'type','HC','display','off');
        t_stats=beta./se;
        p_value=(1-normcdf(t_stats,0,1));
        x_hat=reshape([ones(size(X_tp1,1),1) X_tw X_tbw X_tm]*beta,size(temp_Y,2),size(temp_Y,1))'; % 还原
        y=temp_Y;
    case 'biweek'
        temp_Y=movmean(X(:,21:end),10,2,'Endpoints','discard'); % X_t plus 1 week
        X_tp1=reshape(temp_Y',size(temp_Y,1)*size(temp_Y,2),1); % X_t plus 1
        temp_X=movmean(X(:,11:end-10),10,2,'Endpoints','discard'); % X_t bi-week
        X_tbw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,1:end-10),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        % [beta,bint,r,rint,stats] = regress(X_tp1,[ones(size(X_tp1,1),1) X_t X_tw X_tbw X_tm]);
        [~,se,beta] = hac([X_tbw X_tm],X_tp1,'type','HC','display','off');
        t_stats=beta./se;
        p_value=(1-normcdf(t_stats,0,1));
        x_hat=reshape([ones(size(X_tp1,1),1) X_tbw X_tm]*beta,size(temp_Y,2),size(temp_Y,1))'; % 还原
        y=temp_Y;
    case 'month'
        temp_Y=movmean(X(:,21:end),20,2,'Endpoints','discard'); % X_t plus 1 week
        X_tp1=reshape(temp_Y',size(temp_Y,1)*size(temp_Y,2),1); % X_t plus 1
        temp_X=movmean(X(:,1:end-20),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        % [beta,bint,r,rint,stats] = regress(X_tp1,[ones(size(X_tp1,1),1) X_t X_tw X_tbw X_tm]);
        [~,se,beta] = hac([X_tm],X_tp1,'type','HC','display','off');
        t_stats=beta./se;
        p_value=(1-normcdf(t_stats,0,1));
        x_hat=reshape([ones(size(X_tp1,1),1) X_tm]*beta,size(temp_Y,2),size(temp_Y,1))'; % 还原
        y=temp_Y;
end
end

%% HAR_RV_fore.m
function [X_hat]=HAR_RV_fore(X,fre,beta)
% 对应fre的估计的系数（从HAR_RV_est估计得到）
% forecast
switch fre
    case 'day'
        X_t=X(:,end);  % X_t da
        temp_X=movmean(X(:,end-4:end),5,2,'Endpoints','discard'); % X_t week
        X_tw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,end-9:end),10,2,'Endpoints','discard'); % X_t bi-week
        X_tbw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,end-19:end),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        X_hat=([ones(size(X_t,1),1) X_t X_tw X_tbw X_tm]*beta);
    case 'week'
        temp_X=movmean(X(:,end-4:end),5,2,'Endpoints','discard'); % X_t week
        X_tw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,end-9:end),10,2,'Endpoints','discard'); % X_t bi-week
        X_tbw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,end-19:end),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        X_hat=([ones(size(X_tw,1),1) X_tw X_tbw X_tm]*beta);
    case 'biweek'
        temp_X=movmean(X(:,end-9:end),10,2,'Endpoints','discard'); % X_t bi-week
        X_tbw=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        temp_X=movmean(X(:,end-19:end),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        X_hat=([ones(size(X_tbw,1),1) X_tbw X_tm]*beta);
    case 'month'
        temp_X=movmean(X(:,end-19:end),20,2,'Endpoints','discard'); % X_t month
        X_tm=reshape(temp_X',size(temp_X,1)*size(temp_X,2),1);
        X_hat=([ones(size(X_tm,1),1) X_tm]*beta);
end
end

%% HAR_RV_OWE.m
function [mse,result,date_update,predict_result,real_result]=HAR_RV_OWE(X,fre,name, fre_size,VALUE_theta,VALUE_alpha,loc_vec,stk_date,hf_re)
%  fre={'day','week','biweek','month'}; % 四种不同期限的预测
% fre_size=[1 5 10 20]; % 对应的长度
% fre='biweek';
% fre_size=10;
% name='N';
% fre_set={'day',1;'week',5;'biweek',10;'month',20};
realized=getfield(X,name);
% 参数
% 第一个模型参数 (如果改了 那么speedup不可用)
window_train=100; % in-sample
window_valid=fre_size; % out-of-sample
% 后续在线组合的参数
sample_size=220;% sample size  220
% VALUE_theta=8e-05; % 用于判断预测正确还是错误的阈值
% VALUE_alpha=8e-05;% 用于决定是否加入一个新训练的CV模型的阈值
timer=1;
% 需要遍历的参数集
for i=1:size(VALUE_theta,2)
    for j=1:size(VALUE_alpha,2)
        config_set(timer,:)={VALUE_theta(i),VALUE_alpha(j)};
        timer=timer+1;
    end
end
k=0.995; % 折现因子 越早的样本越不重要
B=5; % OWE中最大的HAR-RV模型数
% 与HAR-RV做比较
% load('HAR_RV.mat');
% date_harRV=HAR_result(4).RCOV_day{1,1};
% date_harRV={'20121122'};
% loc_date_start=find(ismember(stk_date,date_harRV)==1); % 样本外日期开始的位置
% h = waitbar(0,'Please wait...');
% parfor_progress(size(config_set,1)); % Initialize
for q=1:size(config_set,1)
    theta=config_set{q,1};
    alpha=config_set{q,2};
    % warning('off','all')
    % warning
    % 训练
    % =================在线加权集成组合OEW=================
    [log_num_model,OWE_predict,OWE_pre_coef,date_update,TOTAL_ERROR,log_error_owe]=OWE_circle(realized,k,B,alpha,theta,window_train,window_valid,sample_size,fre,fre_size,stk_date);
    date_update(end)=[]; % 与 OWE_predict 对齐
    % 模型的输出结果 :  log_num_model OWE_predict OWE_pre_coef date_update
    % 将vech_chol_realizedCOV的OWE预测结果转化为matrix
    predict_result=[];
    real_result=[];
    mse=[];
    for i=1:size(OWE_predict,2)
        temp_matrix=zeros(size(hf_re,2),size(hf_re,2));
        temp_matrix(loc_vec)=OWE_predict(:,i);
        predict_result(:,:,i)=temp_matrix'*temp_matrix; % 模型输出预测值
        % 真实值
        temp_fre=zeros(size(hf_re,2),size(hf_re,2));
        loc_date=find(ismember(stk_date,date_update(i))==1)+1; % 样本外日期开始的位置
        temp_fre(loc_vec)=movmean(realized(:,loc_date:loc_date+fre_size-1),fre_size,2,'Endpoints','discard'); %
        real_result(:,:,i)=temp_fre'*temp_fre;
        % mse
        mse(i)=sum(sum((real_result(:,:,i)-predict_result(:,:,i)).^2,1),2);
        % 投资组合策略结果
        if isnan(mse(i))
            i
        end
    end
    % MSE(q)=mean(mse(find(ismember(date_update,date_harRV)==1):end));
    % STD(q)=std(mse(find(ismember(date_update,date_harRV)==1):end));
    MSE(q)=mean(mse);
    STD(q)=std(mse);
    %     pause(rand); % Replace with real code
    %     parfor_progress; % Count
end
% dbclear if error
% mean(HAR_result(3).RCOV_month(20:20:end))
% save('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\configuration\N_month.mat')
% save('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\configuration\P_month.mat')
result(1,1)=MSE;
result(1,2)=STD;
% eval(['OWE_stats(g,3)=mean(HAR_result(3).N_',fre,'(fre_size:fre_size:end));']);
% eval(['OWE_stats(g,4)=std(HAR_result(3).N_',fre,'(fre_size:fre_size:end));']);
end

%% 非对称检验 每日选取500组进行检验
addpath(genpath('J:\工作\2019-08-15 （修改）\修改\程序相关'));
% 高频数据处理 2011129到20181129 每日选择有交易的股票进行非对称检验
di=dir('I:\工作\A股交易数据\A股1分钟数据');
di(1:2)=[];di(end)=[];
load('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\hf_trading_status.mat','trading_graph','trading_date','stklist')
loc_start=2981; % 从20111129开始
loc_end=4684; % 到20181129结束
load('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\500_random_pick.mat','stk_selected','loc_start','loc_end')
reject_rate=[];
parfor_progress(loc_end-loc_start+1); % Initialize
parfor i=loc_start:loc_end
    
    %     stk_date{k,1}=di(i).name([1:4 6:7 9:10]);
    file=fopen([di(i).folder,'\',di(i).name]);
    data=textscan(file,'%d %f %f %d %f\n','Delimiter',',');
    fclose(file);
    count1=0;
    count2=0;
    count3=0;
    gamma=[];
    temp_P=nan(500,3);
    for j=1:500
        temp_loc=stk_selected(j,:,i);
        hf_re=[data{1,5}(ismember(data{1,1},stklist(temp_loc(1))),:) data{1,5}(ismember(data{1,1},stklist(temp_loc(2))),:)];
        hf_re(isinf(hf_re))=0;
        hf_re(isnan(hf_re))=0;
        hf_re=log(1+hf_re);
        %         hf_re=movmean(hf_re,5,'Endpoints','discard');
        RCOV=hf_re'*hf_re;
        P=((hf_re+abs(hf_re))/2)'*((hf_re+abs(hf_re))/2);
        N=((hf_re-abs(hf_re))/2)'*((hf_re-abs(hf_re))/2);
        r_p1=(hf_re(:,1)+abs(hf_re(:,1)))/2;
        r_n1=(hf_re(:,1)-abs(hf_re(:,1)))/2; %正负收益
        r_p2=(hf_re(:,2)+abs(hf_re(:,2)))/2;
        r_n2=(hf_re(:,2)-abs(hf_re(:,2)))/2;
        gamma(j)=sqrt(240)*(P(1,2)-N(1,2))/((240*((r_p1.^2)'*(r_p2.^2)+(r_n1.^2)'*(r_n2.^2)))-(P(1,2)-N(1,2))^2);
        count1=count1+(gamma(j)>norminv(0.99,0,1)); % 若P>N 则+1 % gamma(j)<norminv(0.005,0,1) ||
        count2=count2+(gamma(j)>norminv(0.95,0,1));
        count3=count3+(gamma(j)>norminv(0.90,0,1));
        temp_P(j,:)=[RCOV(1,2) P(1,2) N(1,2)];
    end
    non_diagnol(i,:)=mean(temp_P,1);
    P_plus_N_99(i)=count1/500; %计算每天的拒绝比例
    P_plus_N_95(i)=count2/500; %计算每天的拒绝比例
    P_plus_N_90(i)=count3/500; %计算每天的拒绝比例
    pause(rand); % Replace with real code
    parfor_progress; % Count
end
save('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\result\P_N.mat','P_plus_N_99','P_plus_N_95','P_plus_N_90','non_diagnol')
% 表
% RCOV P N 的描述性统计
load('J:\工作\2019-08-15 （修改）\修改\程序相关\数据\hf_matrix_1st_10stks.mat');
label={'RCOV';'P';'N'};
for i=1:size(label,1)
    temp=Realized_matrix.(label{i});
    for j=1:size(temp,3)
        tempp=temp(:,:,j);
        diagonal(:,j)=tempp(logical(eye(10))); % 主对角元素
        nondiag(:,j)=tempp(logical(tril(ones(10,10),-1)));
    end
    stats_diag(i,:)=[mean(mean(diagonal,2)) mean(prctile(diagonal,25,2)) mean(median(diagonal,2)) ...
        mean(prctile(diagonal,75,2))  mean(var(diagonal,0,2)) mean(skewness(diagonal,0,2)) mean(kurtosis(diagonal,0,2))];
    stats_nondiag(i,:)=[mean(mean(nondiag,2)) mean(prctile(nondiag,25,2)) mean(median(nondiag,2))...
        mean(prctile(nondiag,75,2))  mean(var(nondiag,0,2)) mean(skewness(nondiag,0,2)) mean(kurtosis(nondiag,0,2))];
end
temp_table=[stats_diag; stats_nondiag];
% 输出latex表格
rownames={'RCOV','P','N','RCOV_1','P_1','N_1'}; % ,'subs-4','subs-5'
colnames={'均值','25分位数','中位数','75分位数','方差','偏度','峰度'};
a=table(temp_table(:,1),temp_table(:,2),temp_table(:,3),temp_table(:,4),temp_table(:,5),temp_table(:,6),temp_table(:,7),'VariableNames',colnames');
a.Properties.RowNames = rownames;
table2latex({a},'协方差阵','J:\工作\2019-08-15 （修改）\修改\程序相关\表格\table_summary_stats.tex');
% 画图
load('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\500_random_pick.mat','stk_selected','loc_start','loc_end')
load('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\result\P_N.mat','P_plus_N_99','P_plus_N_95','P_plus_N_90','non_diagnol')
%画图1 已实现半协方差的非对称性检验
a=figure('units','normalized','position',[0.1,0.1,0.6,0.2])
date=cellfun(@(x) datenum(x,'yyyymmdd'),trading_date,'UniformOutput',true);
set(gcf,'color','white')
% subplot(2,1,1)
b1=bar(date(loc_start:loc_end),P_plus_N_99(loc_start:end),'FaceColor',[0.3 0.3 0.3])
datetick('x','yyyy/mm/dd')
xlim([date(loc_start) date(loc_end)])
ylabel('拒绝比例','Interpreter','latex')
set(gca,'Ticklength',[0.01 0],'Fontsize',12)
legend(b1,'$$P\leq N$$','Interpreter','latex')
ylim([0 1.1])
% 图2 RCOV(1,2) P(1,2) N(1,2) 的时变图
a=figure('units','normalized','position',[0.1,0.1,0.5,0.6])
date=cellfun(@(x) datenum(x,'yyyymmdd'),trading_date,'UniformOutput',true);
set(gcf,'color','white')
subplot(3,1,1)
p1=plot(date(loc_start:loc_end),non_diagnol(loc_start:end,1),'-k');
ylim([-1e-04 4e-03])
title('$$RCOV^{(m)}$$','Interpreter','latex')
datetick('x','yyyy/mm/dd')
xlim([date(loc_start) date(loc_end)])
set(gca,'Ticklength',[0.005,0],'Fontsize',12)
% legend(p1,'$$RCOV^{(m)}$$','Interpreter','latex')
subplot(3,1,2)
p2=plot(date(loc_start:loc_end),non_diagnol(loc_start:end,2),'-k');
ylim([-1e-04 4e-03])
title('$$P^{(m)}$$','Interpreter','latex')
datetick('x','yyyy/mm/dd')
xlim([date(loc_start) date(loc_end)])
set(gca,'Ticklength',[0.005,0],'Fontsize',12)
% legend(p2,'$$P^{(m)}$$','Interpreter','latex')
subplot(3,1,3)
p3=plot(date(loc_start:loc_end),non_diagnol(loc_start:end,3),'-k');
ylim([-1e-04 4e-03])
title('$$N^{(m)}$$','Interpreter','latex')
datetick('x','yyyy/mm/dd')
xlim([date(loc_start) date(loc_end)])
set(gca,'Ticklength',[0.005,0],'Fontsize',12)
% legend(p3,'$$N^{(m)}$$','Interpreter','latex')
% =============图3 和 图4================
%  图3
load('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\hf_re_1st_10stks.mat')
daily_re_date=cellstr(num2str(daily_re_date) );
for i=1:size(hf_re,3)  % 887 对应20150724 R_n-R_rcov ERC 最大
    i
    temp_hf_re=hf_re(:,:,i);
    temp_hf_re(isinf(temp_hf_re))=0;
    temp_hf_re(isnan(temp_hf_re))=0;
    temp_hf_re=log(temp_hf_re+1);
    RSCOV(:,:,1)=temp_hf_re'*temp_hf_re;
    RSCOV(:,:,2)=(abs(temp_hf_re)+temp_hf_re)'*(abs(temp_hf_re)+temp_hf_re);
    RSCOV(:,:,3)=(temp_hf_re-abs(temp_hf_re))'*(temp_hf_re-abs(temp_hf_re));
    for t=1:3
        covar=RSCOV(:,:,t);
        % ERC
        opts = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'HessUpdate','steepdesc','Display','off');
        [temp_w, ~, ~, ~] = fminunc(@(x) func_riskparity(x,covar),ones(size(covar,1),1)./size(covar,1),opts);
        w(:,1,t)=temp_w./sum(temp_w);
        % GMV
        opts = optimoptions('quadprog','Display','off');
        w(:,2,t)=quadprog(covar,[],[],[],ones(size(covar,1),1)',1,zeros(size(covar,1),1),ones(size(covar,1),1),ones(size(covar,1),1)./size(covar,1),opts);
    end
    loc=find(ismember(daily_re_date,stk_date{i})==1);
    ERC_re(i,1)=daily_re(loc,:)*w(:,1,1);ERC_re(i,2)=daily_re(loc,:)*w(:,1,2);ERC_re(i,3)=daily_re(loc,:)*w(:,1,3);
    GMV_re(i,1)=daily_re(loc,:)*w(:,2,1);GMV_re(i,2)=daily_re(loc,:)*w(:,2,2);GMV_re(i,3)=daily_re(loc,:)*w(:,2,3);
    %     DAILY_RE=daily_re(loc,:);
    %     save('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\result\ERC_20150724.mat','w','temp_hf_re','DAILY_RE')
end
ERC_re(isnan(ERC_re))=0;
GMV_re(isnan(GMV_re))=0;
date=cellfun(@(x) datenum(x,'yyyymmdd'),stk_date,'UniformOutput',true);
a=figure('units','normalized','position',[0.1,0.1,0.6,0.2])
set(gcf,'color','white')
subplot(1,2,1)
p1=plot(date,ERC_re(:,3)-ERC_re(:,1),'-k')
%  legend(p1,'$$R_N-R_{RCOV}$$','Interpreter','latex')
datetick('x','yyyy/mm/dd')
xlim([date(1) date(end)])
set(gca,'FontSize',12)
title('ERC')
subplot(1,2,2)
p2=plot(date,GMV_re(:,3)-GMV_re(:,1),'-k')
%   legend(p2,'$$R_N-R_{RCOV}$$','Interpreter','latex')
datetick('x','yyyy/mm/dd')
xlim([date(1) date(end)])
set(gca,'FontSize',12)
title('GMV')
% 图
load('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\result\ERC_20150724.mat')
load('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\hf_re_1st_10stks.mat')
weight_ERC=[w(:,1,1) w(:,1,2) w(:,1,3)];
weight_GMV=[w(:,2,1) w(:,2,2) w(:,2,3)];
b=bar(weight_ERC);
legend(b,{'RCOV','P','N'})
% 调参结果
load('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\configuration\N_month.mat','MSE','config_set')
a=figure('units','normalized','position',[0.1,0.1,0.4,0.3])
set(gcf,'color','white')
% 固定alpha
subplot(1,2,1)
p1=plot([1e-05:1e-05:1e-03],MSE(1:100:end),'--k')
hold on
p2=plot([1e-05:1e-05:1e-03],MSE(20:100:end),'-k')
hold on
p3=plot([1e-05:1e-05:1e-03],MSE(50:100:end),'-.k')
hold on
p4=plot([1e-05:1e-05:1e-03],MSE(90:100:end),'-*k')
xlabel('$$\theta$$','Interpreter','latex')
legend([p1,p2,p3,p4],{'$$\alpha=1e-05$$','$$\alpha=2e-04$$','$$\alpha=5e-04$$','$$\alpha=9e-04$$'},'Interpreter','latex','Box','off')
set(gca,'Fontsize',12)
% 固定theta
subplot(1,2,2)
p1=plot([1e-05:1e-05:1e-03],MSE(1:100),'--k')
hold on
p2=plot([1e-05:1e-05:1e-03],MSE(1901:2000),'-k')
hold on
p3=plot([1e-05:1e-05:1e-03],MSE(4901:5000),'-.k')
hold on
p4=plot([1e-05:1e-05:1e-03],MSE(8901:9000),'-*k')
xlabel('$$\alpha$$','Interpreter','latex')
legend([p1,p2,p3,p4],{'$$\theta=1e-05$$','$$\theta=2e-04$$','$$\theta=5e-04$$','$$\theta=9e-04$$'},'Interpreter','latex','Box','off','Location','southeast')
set(gca,'Fontsize',12)

%% main3
% 不使用预测方法(HAR-RV) 直接基于RCOV P N 等 进行样本外投资
clear
load('D:\工作台\2019-08-15 （修改）\修改\程序相关\数据\hf_re_1st_10stks.mat')
daily_re_date=cellstr(num2str(daily_re_date));
% 与HAR-RV以及OWE-HAR-RV对齐
name={'RCOV','P','N'};
fre_set={'day',1}; % ;'week',5;'biweek',10;'month',20
k=1;
title={};
log_error=[];
for i=1:size(name,2)
    for j=1:size(fre_set,1)
        temp_hf_re=hf_re(:,:,240+fre_set{j,2}-1:end-2);
        temp_stk_date=stk_date(240+fre_set{j,2}-1:end-2);
        [i,j]
        % 生成RCOV矩阵
        switch name{i}
            case 'RCOV'
                Realized_Matrix=generate_realized_matrix(fre_set{j,2},temp_hf_re,name{i});
            case 'P'
                Realized_Matrix=generate_realized_matrix(fre_set{j,2},temp_hf_re,name{i});
            case 'N'
                Realized_Matrix=generate_realized_matrix(fre_set{j,2},temp_hf_re,name{i});
        end
        date_update=temp_stk_date(1:fre_set{j,2}:end);
        % 3种投资策略表现
        [temp_result,w]=Portfolio_evualtion(daily_re,date_update,daily_re_date,fre_set{j,2},Realized_Matrix,'re'); % temp_result 为三种策略的表现 w 为权重
        if sum(sum(isnan(temp_result)))>0
            break
        end
        result(k,1:2)=[mean(temp_result(:,1)) min(temp_result(:,1))];
        result(k,3:4)=[mean(temp_result(:,2)) min(temp_result(:,2))];
        result(k,5:6)=[mean(temp_result(:,3)) min(temp_result(:,3))];
        title(k,:)={name(i) fre_set(j,1) fre_set{j,2}};
        find(temp_result(:,2)==min(temp_result(:,2)))
        k=k+1;
    end
end
% [result(1:4:end,:);result(2:4:end,:);result(3:4:end,:);result(4:4:end,:)]'; % 表1的格式

%% OWE_circle
function [log_num_model,OWE_predict,OWE_pre_coef,date_update,TOTAL_ERROR,log_error_owe]=OWE_circle(realized,k,B,alpha,theta,window_train,window_valid,sample_size,fre,fre_size,stk_date)
date_update={};
num_update=0;
num_model=0;
penalty=ones(sample_size,1)./sample_size;
window_length=0;
% OWE
log_num_model=[];
total_error=[];
for i=1:size(stk_date,1)-fre_size % 最后留出fre_size天用作评估
    % 3(a)
    if i>sample_size && i>sample_size+20+window_valid-1 % 更新OWE(由于初始模型训练需要【1:sample_size+window_train+window_valid-1】的样本，因此滚动从sample_size+window_train+window_valid开始
        window_length=window_length+1;
        if window_length==window_valid % 填满一个window_valid 当做增加一个样本 更新一次模型
            % 3(c)整体OWE模型输出加入了新样本后的混合权重
            model_predict=nan(size(realized,1),num_model);
            model_predict_coed=[];
            for j=1:num_model
                model_predict(:,j)=model_ind_yhat{j}(:,end);
                model_predict_coed(:,j)=model_ind_coef{j}(:,end);
            end
            % error 为0时 防止出现nan
            if sum(total_error==0)>=1
                i
                total_error(total_error==0)=0.000000001;
            end
            %             realized(:,i-sample_size+1:i+20+window_valid-1);% 此阶段的vech_chol_realizedCOV 20是因为每个样本长度总是至少跨越21天
            OWE_predict(:,num_update)=sum(model_predict.*repmat(log(1./total_error),size(model_predict,1),1),2)/sum(log(1./total_error)); % OWE模型输出的预测值vech_chol_realizedCOV
            OWE_pre_coef(:,num_update)=sum(model_predict_coed.*repmat(log(1./total_error),size(model_predict_coed,1),1),2)/sum(log(1./total_error)); % OWE模型输出的预测值vech_chol_realizedCOV
            %             OWE_predict(:,num_update)=mean(model_predict,2);
            %             OWE_pre_coef(:,num_update)=mean(model_predict_coed,2);
            num_update=num_update+1;
            date_update{num_update}=stk_date{i}; % 更新模型的日期(下一日相对当前模型来说，即划分为out-of-sampe)
            % 3(d) owe 在最新的sample_size个样本集上的误差(对应原文中的ARE误差 这里基于frobenius的MSE)
            temp_vechol=realized(:,i-(sample_size+20+window_valid-2):i);% 此阶段的vech_chol_realizedCOV
            [ERROR_OWE]=ERROR_calculator(temp_vechol,OWE_pre_coef(:,num_update-1),fre);
            log_error_owe(num_update)=mean(ERROR_OWE);
            % 3(e)
            total_samples=sum(abs(ERROR_OWE)>theta);
            %3(f)
            if total_samples==0
                upFactor=1;
                downFactor=1;
            else
                upFactor=sample_size/total_samples;
                downFactor=1/upFactor;
            end
            % 3(g)
            penalty=ones(sample_size,1)./sample_size;
            penalty(ERROR_OWE>theta)=penalty(ERROR_OWE>theta).*upFactor;
            penalty(ERROR_OWE<=theta)=penalty(ERROR_OWE<=theta).*downFactor;
            penalty=penalty./sum(penalty);
            TOTAL_ERROR(num_update)=ERROR_OWE(end); % OWE模型的总体error
            % 3(h)
            if sum((temp_vechol(:,end)-OWE_predict(:,num_update-1)).^2)>alpha
                % 加入一个新模型
                num_model=num_model+1;
                new_model=1; % 当次更新加入了新模型
                temp_vechol=realized(:,i-(sample_size+20+window_valid-2):i);% 此阶段的vech_chol_realizedCOV 20是因为每个样本长度总是至少跨越21天
                
                [beta,~,x_hat,y,~,~]=HAR_RV_est(temp_vechol,fre); % 估计HAR-RV模型
                [X_hat]=HAR_RV_fore(temp_vechol,fre,beta); % 预测
                model_ind_yhat{num_model}(:,num_update)=X_hat;  % 模型预测值
                model_ind_coef{num_model}(:,num_update)=beta;   % 模型系数
            end
            % 3（i）
            for j=1:num_model
                %             temp_vechol 用上面的
                [temp_error]=ERROR_calculator(temp_vechol,model_ind_coef{j}(:,end),fre);
                
                % 计算各HAR-RV模型的mse
                temp_error(abs(temp_error)<=theta)=0.0000000001;
                temp_error(abs(temp_error)>theta)=1;
                error_model(j,num_update)=temp_error*penalty; % 第j个模型在第num_update个窗口的误差 \epsilon_1^{\tau_j}
            end
            % 3(j)
            discount_k_matrix=k.^repmat([num_update-1:-1:0],num_model,1); % 每个模型每个ARE对应的折现因子
            temp_total_error=error_model.*discount_k_matrix;
            discount_k_matrix(isnan(temp_total_error))=0;
            temp_total_error(isnan(temp_total_error))=0;
            for j=1:num_model
                total_error(j)=sum(temp_total_error(j,:),2)./(sum(discount_k_matrix(j,~temp_total_error(j,:)==0),2)); % 各模型的总ARE
            end
            log_num_model=[log_num_model;num_model];
            % 3(k)
            if num_model>B
                %                 log_delete=[log_delete;1];
                if sum(total_error(1:end-1)>0)==0
                    delete_model=find(total_error(1:end-1)==max(total_error(1:end-1))); % 要删除的模型,不删除在本次刚加入的新模
                    delete_model=delete_model(1);
                else
                    delete_model=find(total_error(1:end-1)==max(total_error(1:end-1))); % 要删除的模型,不删除在本次刚加入的新模
                end
                if size(delete_model,2)>1
                    delete_model=delete_model(1);
                end
                error_model(delete_model,:)=[];
                model_ind_yhat(delete_model)=[];
                model_ind_coef(delete_model)=[];
                total_error(delete_model)=[];
                num_model=num_model-1;
                new_model=0;
            end
            window_length=0;
        else
            continue
        end
    else
        % 3(b)
        if i==sample_size % 加入第一个模型
            %             TEMP=zeros(size(hf_re,1),size(hf_re,1));
            temp_vechol=realized(:,i-sample_size+1:i+20+window_valid-1);% 此阶段的vech_chol_realizedCOV 20是因为每个样本长度总是至少跨越21天
            [beta,~,x_hat,y,~,~]=HAR_RV_est(temp_vechol,fre); % 估计HAR-RV模型
            [X_hat]=HAR_RV_fore(temp_vechol,fre,beta); % 预测
            temp_error=sum((y-x_hat).^2,1); % 基于vech_chol_realizedCOV的mse
            num_model=num_model+1;
            % 计算第一个模型的ARE:基于第一批样本(样本内的)
            temp_error(abs(temp_error)<=theta)=0;
            error_model(num_model,1)=temp_error*penalty; % 第一个模型在第一个窗口的误差 \epsilon_1^{\tau_j}
            % 各模型整体的误差（此处等于第一个模型在第一个窗口的误差）
            discount_k_matrix=k.^repmat([num_update:-1:0],num_model,1); % 每个模型每个ARE对应的折现因子
            temp_total_error=error_model.*discount_k_matrix;
            discount_k_matrix(isnan(temp_total_error))=0;
            temp_total_error(isnan(temp_total_error))=0;
            total_error(num_model,1)=sum(temp_total_error,2)./(sum(discount_k_matrix,2)); % 各模型的总ARE
            num_update=num_update+1;
            date_update{num_update}=stk_date{i+20+window_valid-1};% 更新模型的日期(下一日相对当前模型来说，即划分为out-of-sampe)
            model_ind_yhat{num_model}=X_hat;  % 模型预测值
            model_ind_coef{num_model}=beta;   % 模型系数
        else
            continue
        end
        log_num_model(num_update)=num_model;
    end
    %     str=['q:',num2str(q),' /',num2str(size(VALUE_alpha,2)),'; ',num2str(i),' /',num2str(size(stk_date,1)-fre_size ),' , time elapses：',num2str(toc(t1)),' s'];
    %     waitbar(i/(size(stk_date,1)-fre_size ),h,str);
end
end

%%  做稳健性检验
% addpath(genpath('J:\工作\2019-08-15 （修改）\修改\程序相关')); 
%从原始数据得到高频的hf_re
di=dir('J:\工作\A股交易数据\A股1分钟数据');
di(1:2)=[];
load('hf_stk_selected.mat','stk_selected','date_start','date_end')
stk_selected=stk_selected(1:43);
h = waitbar(0,'Please wait...');
t1=tic;
% 第1组股票
stk_date=cell(date_end-date_start+1,1);
k=1;
hf_re=nan(240,size(stk_selected,1),date_end-date_start+1);
for i=date_start:date_end
    stk_date{k,1}=di(i).name([1:4 6:7 9:10]);
    file=fopen([di(i).folder,'\',di(i).name]);
    data=textscan(file,'%d %f %f %d %f\n','Delimiter',',');
    fclose(file);
    loc_selected=ismember(data{1,1},stk_selected); % 选中股票在当天数据中的位置
    hf_re(:,:,k)=reshape(data{1,5}(loc_selected,:),240,size(data{1,1}(loc_selected),1)/240);
    if size(data{1,1}(loc_selected),1)/240~=size(stk_selected,1) % 是否都包含240个区间？
    log_error(k)=i;
    end
    k=k+1;
    str=['calculating IVOL...',num2str(k),' /',num2str(date_end-date_start+1),' , time elapses：',num2str(toc(t1)),' s'];
    waitbar(k/(date_end-date_start+1),h,str);
end
% 补充对应的日收益率数据
load('aligned_daily_re.mat','daily_re','stklist','datelist_csiindex')
stklist=cellstr(stklist);
stklist=cellfun(@(x) str2num(x(1:6)),stklist,'UniformOutput',true);
loc_stk=ismember(stklist,stk_selected); % 匹配
loc_date=ismember(datelist_csiindex,cellfun(@(x) str2num(x),stk_date,'UniformOutput',true));
daily_re=daily_re(:,loc_stk);
daily_re_date=datelist_csiindex;
save('J:\工作\2019-08-15 （修改）\修改\程序相关\数据\robust_hf_re_43stks.mat','hf_re','stk_date','stk_selected','daily_re','daily_re_date')
% 清洗数据 计算RCOV-P-N-M
load('J:\工作\2019-08-15 （修改）\修改\程序相关\数据\robust_hf_re_43stks.mat')
log_error=[];
Realized_matrix.RCOV=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
Realized_matrix.P=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
Realized_matrix.N=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
Realized_matrix.M=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
Realized_matrix.M1=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
Realized_matrix.M2=nan(size(stk_selected,1),size(stk_selected,1),size(hf_re,3));
dbstop if error
for i=1:size(hf_re,3)
    temp=hf_re(:,:,i);
    temp(isnan(temp))=0;
    temp(isinf(temp))=0;
    if sum(sum(temp>0.1))>=1 || sum(sum(temp<-0.1))>=1
        log_error=[log_error;i];  
    end
    temp=log(1+temp);
    temp_Pos=(abs(temp)+temp)/2;
    temp_Neg=(temp-abs(temp))/2;
    Realized_matrix.RCOV(:,:,i)=temp'*temp;    
    Realized_matrix.P(:,:,i)=temp_Pos'*temp_Pos;   
    Realized_matrix.N(:,:,i)=temp_Neg'*temp_Neg;    
    Realized_matrix.M1(:,:,i)=temp_Neg'*temp_Pos;
    Realized_matrix.M2(:,:,i)=temp_Pos'*temp_Neg;
    Realized_matrix.M(:,:,i)=Realized_matrix.M1(:,:,i)+Realized_matrix.M2(:,:,i);
end
save('J:\工作\2019-08-15 （修改）\修改\程序相关\数据\robust_hf_matrix_43_stks.mat','Realized_matrix','hf_re')
%%%%%%%%% HAR-RV %%%%%%%%%%
load('J:\工作\2019-08-15 （修改）\修改\程序相关\数据\robust_hf_matrix_43_stks.mat')
name={'RCOV','P','N'}; % 选择要进行预测的矩阵
 % =========cholesky + vectorize=================
 loc_vec=[]; % 上三角矩阵的索引
 k=1;
 for i=1:size(hf_re,2)
     
      loc_vec=[loc_vec;[1:k]'+size(hf_re,2)*(i-1)];
      k=k+1;
     
 end
log_notpd=[]; 
X=struct;
for i=1:size(name,2)

 temp=getfield(Realized_matrix,name{i});

 temp_matrix=nan(size(hf_re,2)*(1+size(hf_re,2))/2,size(temp,3));
 for j=1:size(temp,3)
     % 正定性 将元素全为0的对应证券的方差用过去10日的日已实现方差代替
     if rank(temp(:,:,j))<size(hf_re,2)
        log_notpd=[log_notpd;i j];
        temp_loc=find(diag(temp(:,:,j))==0);
        temp(temp_loc,temp_loc,j)=mean(temp(temp_loc,temp_loc,j-10:j-1),3);
     end
      if rank(temp(:,:,j))<size(hf_re,2)
        temp(:,:,j)=(temp(:,:,j-1)+temp(:,:,j+1))/2;
     end         
     temp_chol=chol(temp(:,:,j));
      temp_matrix(:,j)=temp_chol(loc_vec);
 end
  X=setfield(X,name{i},temp_matrix);
end
save('J:\工作\2019-08-15 （修改）\修改\程序相关\数据\robustCheck_vechChol_HAR_RV_43stks.mat','X','loc_vec')
% 生成结果
% =========RCOV P N 仅基于HAR-RV的样本外表现(基于fronbenius rmse)=================
load('J:\工作\2019-08-15 （修改）\修改\程序相关\数据\robustCheck_vechChol_HAR_RV_43stks.mat')
load('J:\工作\2019-08-15 （修改）\修改\程序相关\数据\robust_hf_re_43stks.mat')
fre={'month'}; % 不同期限的预测
name={'RCOV','P','N'}; % 选择要进行预测的矩阵
fre_size=[20]; % 对应的长度
HAR_result=struct;
window_size=240;
for t=1:size(name,2) 
    temp=getfield(X,name{t});
    h = waitbar(0,'Please wait...');
    t1=tic;
    for i=1:size(fre,2)
        k=1;
        forecast_date=cell(size(stk_date,1)-fre_size(i)+1-window_size+1,1);
        FORE=nan(size(hf_re,2),size(hf_re,2),size(stk_date,1)-fre_size(i)-window_size+1);
        REAL=nan(size(hf_re,2),size(hf_re,2),size(stk_date,1)-fre_size(i)-window_size+1);
        rmse=nan(size(stk_date,1)-fre_size(i)-240+1,1);
        for j=window_size:size(stk_date,1)-fre_size(i) % 第一个预测 留一年作为初始训练集长度 此后将使用window_size的历史数据作为训练集 最后留出fre_size天作为最后一个预测区间
            str=['t=',num2str(t),'; 处理进度：',num2str((k/(size(stk_date,1)-fre_size(i)-window_size+1))*100),' %, time elapses：',num2str(toc(t1)),' s'];
            waitbar(k/(size(stk_date,1)-fre_size(i)-window_size+1),h,str);
            forecast_date{k}=stk_date(j); %划分历史和未来的日期（下一日为未来）
            input=temp(:,1:j);
            % vech(chol(cov))预测值
            [beta,se,x_hat,~,t_stats,p_value]=HAR_RV_est(input,fre{i});
            [X_hat]=HAR_RV_fore(input,fre{i},beta);
            % chol(cov)预测值
            temp_fre=zeros(size(hf_re,2),size(hf_re,2));
            temp_fre(loc_vec)=X_hat;
            % cov预测值
            FORE(:,:,k)=temp_fre'*temp_fre;
            % 真实值
            temp_fre=zeros(size(hf_re,2),size(hf_re,2));
            temp_fre(loc_vec)=movmean(temp(:,j+1:j+fre_size(i)),fre_size(i),2,'Endpoints','discard'); % 
            REAL(:,:,k)=temp_fre'*temp_fre;
            % RMSE
            rmse(k)=sum(sum((FORE(:,:,k)-REAL(:,:,k)).^2,1),2);
            k=k+1;
        end
        HAR_result=setfield(HAR_result,{1},[name{t},'_',fre{i}],FORE); % 字段第1行为预测协方差值
        HAR_result=setfield(HAR_result,{2},[name{t},'_',fre{i}],REAL); % 字段第2行为协方差真实值
        HAR_result=setfield(HAR_result,{3},[name{t},'_',fre{i}],rmse); % 字段第3行为对应rmse
        HAR_result=setfield(HAR_result,{4},[name{t},'_',fre{i}],forecast_date); % 字段第4行为日期
    end 
end
stats=fieldnames(HAR_result);
stats(:,2)=num2cell(structfun(@(x) mean(x(~isnan(x))),HAR_result(3),'UniformOutput',true)'); % mean
stats(:,3)=num2cell(structfun(@(x) std(x(~isnan(x))),HAR_result(3),'UniformOutput',true)');  % std
% 组合投资
load('J:\工作\2019-08-15 （修改）\修改\程序相关\数据\result\robust_30stks_HAR_RV.mat');
load('robust_hf_re_30stks.mat')
daily_re_date=cellstr(num2str(daily_re_date));
name={'RCOV','P','N'}; %
fre_set={'month',20};
k=1;
title={};
for i=1:size(name,2)
    for j=1:size(fre_set,1)
        [i,j]
        eval(['result(k,1)=mean(HAR_result(3).',name{i},'_',fre_set{j,1},'(fre_set{j,2}:fre_set{j,2}:end));'])
        eval(['result(k,2)=std(HAR_result(3).',name{i},'_',fre_set{j,1},'(fre_set{j,2}:fre_set{j,2}:end));'])
        eval(['predict_result=HAR_result(1).',name{i},'_',fre_set{j,1},'(:,:,1:fre_set{j,2}:end-1);'])
        eval(['date_update=HAR_result(4).',name{i},'_',fre_set{j,1},'(1:fre_set{j,2}:end-2);'])
        % 3种投资策略表现
        [temp_result,w]=Portfolio_evualtion(daily_re,date_update,daily_re_date,fre_set{j,2},predict_result,'re'); % temp_result 为三种策略的表现 w 为权重
        if sum(sum(isnan(temp_result)))>0
            break
        end
        result(k,3:4)=[mean(temp_result(:,1)) min(temp_result(:,1))];
        result(k,5:6)=[mean(temp_result(:,2)) min(temp_result(:,2))];
        result(k,7:8)=[mean(temp_result(:,3)) min(temp_result(:,3))];
        title(k,:)={name(i) fre_set(j,1) fre_set{j,2}};
        % 交易成本
        cost(k,:)=[0 0];
        for p=1:size(w,3)-1
        cost(k,:)=cost(k,:)+sum(abs(w(:,1:2,p+1)-w(:,1:2,p)),1);
        end
        k=k+1;
    end
end
title_row=title;
title_col={'MSE','std(MSE)','ERC_mean(Re)','ERC_min(Re)','GMV_mean(Re)','GMV_min(Re)','1/N_mean(Re)','1/N_min(Re)'};
result=[[{''};title(:,1)] [title_col;num2cell(result)]];
% 表格
load('robust_result_43stks_HAR_RV.mat')
a=[result(2:end,2:end)];
load('robust_result_30stks_HAR_RV.mat')
a=[a;result(2:end,2:end)];
% 输出latex表格
rownames={'RCOV','P','N','RCOV_1','P_1','N_1'}; % ,'subs-4','subs-5'
colnames={'MSE','std(MSE)','ERC_mean(Re)','ERC_min(Re)','GMV_mean(Re)','GMV_min(Re)','1/N_mean(Re)','1/N_min(Re)'};
a=table(a(:,1),a(:,2),a(:,3),a(:,4),a(:,5),a(:,6),a(:,7),a(:,8),'VariableNames',colnames');
a.Properties.RowNames = rownames;
table2latex({a},'协方差阵','J:\工作\2019-08-15 （修改）\修改\程序相关\表格\table_robustCheck.tex');


