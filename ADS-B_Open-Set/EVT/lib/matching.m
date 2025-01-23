close all
clear all
load('cosface_m04_r64_classes10.mat')

s = 64;
am_embeds_norm = s * am_embeds_norm;
% 原型分析
am_embeds_mean = ones(10,512);
for i  = 0:9
    [r,c] = find(am_labels == i);
    am_embeds_mean(i+1,:) = mean(am_embeds_norm(c,:));
end
save('centroid_vectors_m04.mat','am_embeds_mean')

%类间类内距离分析
e_dis = zeros(size(am_embeds_norm,1),size(am_embeds_mean,1));
c_dis = zeros(size(am_embeds_norm,1),size(am_embeds_mean,1));
intra_dis = zeros(size(am_embeds_norm,1),1);
inter_dis = zeros(size(am_embeds_norm,1),9);
for i = 1:size(am_embeds_norm, 1)
    for j = 1:size(am_embeds_mean,1)
        e_dis(i,j) = sqrt(sum((am_embeds_norm(i,:)-am_embeds_mean(j,:)).^2));
        c_dis(i,j) = sum(sum(am_embeds_norm(i,:).*am_embeds_mean(j,:)))/((sqrt(sum(sum(am_embeds_norm(i,:).^2))))*(sqrt(sum(sum(am_embeds_mean(j,:).^2)))));
    end
    intra_e_dis(i) = e_dis(i,am_labels(:,i)+1);
    intra_c_dis(i) = c_dis(i,am_labels(:,i)+1);
    e_dis_i = e_dis(i,:);
    c_dis_i = c_dis(i,:);
    inter_e_dis(i,:) = e_dis_i(~ismember(e_dis_i, intra_e_dis(i)));
    inter_c_dis(i,:) = c_dis_i(~ismember(c_dis_i, intra_c_dis(i)));
end
save('intra_e_dis.mat','intra_e_dis')
save('intra_c_dis.mat','intra_c_dis')

figure(1)
h1 = histogram(intra_e_dis,'Normalization','probability')
hold on
h2 = histogram(inter_e_dis,'Normalization','probability')
xlabel('Distance')
ylabel('Density')
legend('intra-distance','inter-distance')

figure(2)
h1 = histogram(intra_c_dis,'Normalization','probability')
hold on
h2 = histogram(inter_c_dis,'Normalization','probability')
xlabel('Distance')
ylabel('Density')
legend('intra-distance','inter-distance')

figure(5)
h1 = histfit(-intra_c_dis,40,'kernel')
xlabel('Distance')
ylabel('Frequency')
legend('intra-distance','intra-distance')

x1 = h1(2).XData;
y1 = h1(2).YData;

figure(6)
intra_c_dis = sort(-intra_c_dis,'ascend');
[h_cdf] = cdfplot(intra_c_dis);

x2 = h_cdf.XData;
y2 = h_cdf.YData;
[r_y,c_y] = find(y2>=0.98);
[r_x,c_x] = find(intra_c_dis>=x2(c_y(1)));

figure(7)
[AX, H2, H1] = plotyy(x2,y2,x1,y1)
hold on
stem(x2(c_y(1)),y2(c_y(1)));
set(get(AX(2),'Ylabel'),'String','Frequency','FontSize',16,'FontName','Times New Roman') 
set(get(AX(1),'Ylabel'),'String','Cumulative Distribution','FontSize',16,'FontName','Times New Roman') 
xlabel('Distance') 
set(H1,'LineStyle','--')
set(H2,'LineStyle',':')

%测试集的开集打分(欧式距离、余弦相似度)
load('test_cosface_m04_r64_classes10+2.mat')
test_am_embeds_norm = s * test_am_embeds_norm;
e_dis = zeros(size(test_am_embeds_norm,1),size(am_embeds_mean,1));
c_dis = zeros(size(test_am_embeds_norm,1),size(am_embeds_mean,1));
min_e_dis = zeros(size(test_am_embeds_norm,1),1);
max_c_dis = zeros(size(test_am_embeds_norm,1),1);
pred_label = zeros(size(test_am_embeds_norm, 1),1);
correct = 0;
[r,c] = find(test_am_labels>=10)
test_am_labels(c)=-1
for i = 1:size(test_am_embeds_norm, 1)
    for j = 1:size(am_embeds_mean,1)
        e_dis(i,j) = sqrt(sum((test_am_embeds_norm(i,:)-am_embeds_mean(j,:)).^2));
        c_dis(i,j) = sum(sum(test_am_embeds_norm(i,:).*am_embeds_mean(j,:)))/((sqrt(sum(sum(test_am_embeds_norm(i,:).^2))))*(sqrt(sum(sum(am_embeds_mean(j,:).^2)))));
    end
    min_e_dis(i) = min(e_dis(i,:));
    max_c_dis(i) = max(c_dis(i,:));
    if max_c_dis(i) >= 0.5
        [r, pred_label(i)] = find(c_dis(i,:)==max_c_dis(i));
        pred_label(i) = pred_label(i) - 1;
    else
        pred_label(i) = -1;
    end
    if pred_label(i)==test_am_labels(i)
        correct = correct + 1;
    end
end
acc = correct/size(test_am_embeds_norm, 1);

e_dis_kwn = min_e_dis(test_am_labels~=-1);
e_dis_unk = min_e_dis(test_am_labels==-1);
c_dis_kwn = max_c_dis(test_am_labels~=-1);
c_dis_unk = max_c_dis(test_am_labels==-1);

figure(3)
h1 = histogram(e_dis_kwn,'Normalization','probability')
hold on
h2 = histogram(e_dis_unk,'Normalization','probability')
xlabel('Distance')
ylabel('probability')
legend('known','unknown')

figure(4)
h1 = histogram(c_dis_kwn,'Normalization','probability')
hold on
h2 = histogram(c_dis_unk,'Normalization','probability')
xlabel('Distance')
ylabel('probability')
legend('known','unknown')

