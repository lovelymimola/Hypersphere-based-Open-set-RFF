clc; close all;clear all;

addpath('./lib')
%% initialization
dataset_name = 'ADS-B 8';
view_error_histogram=1;
hist_bins=500;
worked_flag=1;
prior0=0.5;

%% loading data
load('./data/test_out_prob_open.mat');  %测试集的分类得分，可以是logits
load('./data/intra_c_dis.mat'); %训练集的类内得分，用于基于极值理论的尾部建模
load('./data/max_c_dis.mat');  %测试集的已知未知得分，可以是特征与原型的最小欧式距离
load('./data/label.mat');  %测试集的真实标签

label=test_am_labels';
[r,c] = find(label>=8)
label(r)=-1
mlosr_mse_train = -intra_c_dis';
mlosr_mse=-max_c_dis';

xlswrite('./cm/label.xlsx', label)

pred_evt=zeros(length(mlosr_mse),1);
p_mlosr_evt=zeros(length(mlosr_mse),1);

acc_mlosr_evt=[];
fm_mlosr_evt=[];
tp_mlosr_evt=zeros(1,1);
tn_mlosr_evt=zeros(1,1);
fp_mlosr_evt=zeros(1,1);
fn_mlosr_evt=zeros(1,1);

kwn = mlosr_mse(label~=-1);
unk = mlosr_mse(label==-1);

if(view_error_histogram)
    figure;
    histogram(-unk,'Normalization','probability');
    hold on;
    histogram(-kwn,'Normalization','probability');
    legend('Unknown','Known');
end

%% training GPD
kwn = mlosr_mse_train
val_match = sort(kwn,'descend');
tailsize_match = 17;

gpd_para_match = gpfit(val_match(1:tailsize_match)-val_match(tailsize_match)+eps);

%% estimating the threshold
thr_prob_evt = 0.5;

prob_vals = 1-gpcdf(mlosr_mse_train-val_match(tailsize_match)+eps, gpd_para_match(1), gpd_para_match(2),0);
prob_vals = prob_vals/max(prob_vals);
id = find(prob_vals<=thr_prob_evt);
thr_rec_evt = mlosr_mse_train(id(1));


%% testing on data
for i=1:length(mlosr_mse)
    
    % get softmax probs
    s_mlosr = test_out_prob_open(i,:);
    [mlosr_score, mlosr_id] = max(s_mlosr);
    
    % get mlosr match score for evt
    p_match = (1-gpcdf(mlosr_mse(i)-val_match(tailsize_match)+eps, gpd_para_match(1),gpd_para_match(2),0));
    p_mlosr_evt(i) = p_match;
    
    % lcos predictions with evt
    if(p_mlosr_evt(i) >= thr_prob_evt)
         pred_evt(i) = mlosr_id-1;
    else
        pred_evt(i) = -1;
    end

end
xlswrite('./cm/pred_label.xlsx', pred_evt)

%% get error stats
for i=1:length(mlosr_mse)
    % tp, tn, fp, fn for lcos evt
    [tn_mlosr_evt, tp_mlosr_evt,...
        fn_mlosr_evt, fp_mlosr_evt] = getTFNP(label(i), pred_evt(i), tn_mlosr_evt,...
                               tp_mlosr_evt, fn_mlosr_evt, fp_mlosr_evt);
    
end


%% get fmeasure and accuracy
[fm_mlosr_evt, tpr, fpr] = getPRF(tp_mlosr_evt, fp_mlosr_evt, fn_mlosr_evt, tn_mlosr_evt);
acc_mlosr_evt=sum(pred_evt==label)/length(mlosr_mse);

%% display results
disp(['ACC is (ours)  : ' num2str(acc_mlosr_evt)])
disp(['F-measure is (ours)  : ' num2str(fm_mlosr_evt)])
disp(['TPR is (ours)  : ' num2str(tpr)])
disp(['FPR is (ours)  : ' num2str(fpr)])

