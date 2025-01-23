close all
clear all
load('cosface_m02_r8_classes8.mat')

s = 8;
am_embeds_norm = s * am_embeds_norm;
% 原型分析
am_embeds_mean = ones(8,512);
for i  = 0:7
    [r,c] = find(am_labels == i);
    am_embeds_mean(i+1,:) = mean(am_embeds_norm(c,:));
end

%类间类内距离分析
e_dis = zeros(size(am_embeds_norm,1),size(am_embeds_mean,1));
c_dis = zeros(size(am_embeds_norm,1),size(am_embeds_mean,1));
intra_e_dis = zeros(size(am_embeds_norm,1),1);
inter_e_dis = zeros(size(am_embeds_norm,1),7);
intra_c_dis = zeros(size(am_embeds_norm,1),1);
inter_c_dis = zeros(size(am_embeds_norm,1),7);
for i = 1:size(am_embeds_norm, 1)
    for j = 1:size(am_embeds_mean,1)
        e_dis(i,j) = sqrt(sum((am_embeds_norm(i,:)-am_embeds_mean(j,:)).^2));
        c_dis(i,j) = dot(am_embeds_norm(i,:),am_embeds_mean(j,:))/(norm(am_embeds_norm(i,:))*norm(am_embeds_mean(j,:)));
    end
    intra_e_dis(i) = e_dis(i,am_labels(:,i)+1);
    intra_c_dis(i) = c_dis(i,am_labels(:,i)+1);
    e_dis_i = e_dis(i,:);
    c_dis_i = c_dis(i,:);
    inter_e_dis(i,:) = e_dis_i(~ismember(e_dis_i, intra_e_dis(i)));
    inter_c_dis(i,:) = c_dis_i(~ismember(c_dis_i, intra_c_dis(i)));
end
save('intra_c_dis.mat','intra_c_dis')

figure(1)
h1 = histogram(intra_e_dis,'Normalization','probability')
hold on
h2 = histogram(inter_e_dis,'Normalization','probability')
xlabel('Distance')
ylabel('Probability')
legend('intra-distance','inter-distance')
title('欧式距离')

figure(2)
h1 = histogram(intra_c_dis,'Normalization','probability')
hold on
h2 = histogram(inter_c_dis,'Normalization','probability')
xlabel('Distance')
ylabel('Probability')
legend('intra-distance','inter-distance')
title('余弦相似度')

figure(3)
h1 = histfit(-intra_c_dis,40,'kernel')
xlabel('Distance')
ylabel('Probability')
legend('intra-distance','intra-distance')
title('余弦相似度-类内距离分析')

x1 = h1(2).XData;
y1 = h1(2).YData;

figure(4)
intra_c_dis = sort(-intra_c_dis,'ascend');
[h_cdf] = cdfplot(intra_c_dis);

x2 = h_cdf.XData;
y2 = h_cdf.YData;
[r_y,c_y] = find(y2>=0.99); %搜寻CDF值为0.99时的尾部长度
[r_x,c_x] = find(intra_c_dis>=x2(c_y(1))); %c_x的个数为尾部长度

figure(5)
[AX, H2, H1] = plotyy(x2,y2,x1,y1)
hold on
stem(x2(c_y(1)),y2(c_y(1)));
set(get(AX(2),'Ylabel'),'String','Frequency','FontSize',16,'FontName','Times New Roman') 
set(get(AX(1),'Ylabel'),'String','Cumulative Distribution','FontSize',16,'FontName','Times New Roman') 
xlabel('Distance') 
set(H1,'LineStyle','--')
set(H2,'LineStyle',':')

%测试集的开集打分(欧式距离、余弦相似度)
load('test_cosface_m02_r8_classes8+2.mat')
test_am_embeds_norm = s * test_am_embeds_norm;
e_dis = zeros(size(test_am_embeds_norm,1),size(am_embeds_mean,1));
c_dis = zeros(size(test_am_embeds_norm,1),size(am_embeds_mean,1));
min_e_dis = zeros(size(test_am_embeds_norm,1),1);
max_c_dis = zeros(size(test_am_embeds_norm,1),1);

for i = 1:size(test_am_embeds_norm, 1)
    for j = 1:size(am_embeds_mean,1)
        e_dis(i,j) = sqrt(sum((test_am_embeds_norm(i,:)-am_embeds_mean(j,:)).^2));
        c_dis(i,j) = dot(test_am_embeds_norm(i,:),am_embeds_mean(j,:))/(norm(test_am_embeds_norm(i,:))*norm(am_embeds_mean(j,:)));
    end
    min_e_dis(i) = min(e_dis(i,:));
    max_c_dis(i) = max(c_dis(i,:));
end
save('test_out_prob_open.mat','test_out_prob_open')
save('max_c_dis.mat','max_c_dis')
save('label.mat','test_am_labels')

[r,c] = find(test_am_labels>=8)
test_am_labels(c)=-1
e_dis_kwn = min_e_dis(test_am_labels~=-1);
e_dis_unk = min_e_dis(test_am_labels==-1);
c_dis_kwn = max_c_dis(test_am_labels~=-1);
c_dis_unk = max_c_dis(test_am_labels==-1);

figure(6)
h1 = histogram(e_dis_kwn,'Normalization','probability')
hold on
h2 = histogram(e_dis_unk,'Normalization','probability')
xlabel('Distance')
ylabel('Probability')
legend('known','unknown')
title('欧式距离')

figure(7)
h1 = histogram(c_dis_kwn,'Normalization','probability')
hold on
h2 = histogram(c_dis_unk,'Normalization','probability')
xlabel('Distance')
ylabel('Probability')
legend('known','unknown')
title('余弦相似度')
