clear all
close all

real_label_data = xlsread("label.xlsx");
pred_label_data = xlsread("pred_label.xlsx");
real_label = real_label_data(1:size(real_label_data,1),1)+1;
pred_label = pred_label_data(1:size(pred_label_data,1),1)+1;
[r,c] = find(real_label==0);
real_label(r)=9;
[r,c] = find(pred_label==0);
pred_label(r)=9;
MC = zeros(9,9);
for k = 1:length(pred_label)
    MC(real_label(k),pred_label(k)) = MC(real_label(k),pred_label(k)) + 1;
end
%filename_MC = strcat('MAT(DQ)_CNN_15label_MC.xlsx');
%xlswrite(filename_MC, MC)

label = {'0','1','2','3','4','5','6','7','ukn'};
maxcolor = [52,131,2 602]; % 最大值颜色
mincolor = [255,255,255]; % 最小值颜色

m = size(MC,1);
imagesc(1:m,1:m,MC)
xticks(1:m)
xlabel('Predict category','fontsize',16)
xticklabels(label)
yticks(1:m)
ylabel('True category','fontsize',16)
yticklabels(label)

%构造渐变色
mymap = [linspace(mincolor(1)/255,maxcolor(1)/255,64)',...
         linspace(mincolor(2)/255,maxcolor(2)/255,64)',...
         linspace(mincolor(3)/255,maxcolor(3)/255,64)'];

colormap(mymap)
colorbar()

% 色块填充数字
for i = 1:m
    for j = 1:m
        text(i,j,num2str(MC(j,i)),...
            'horizontalAlignment','center',...
            'verticalAlignment','middle',...
            'fontname','Times New Roman',...
            'fontsize',16);
    end
end

% 图像坐标轴等宽
ax = gca;
ax.FontName = 'Times New Roman';
set(gca,'box','on','xlim',[0.5,m+0.5],'ylim',[0.5,m+0.5]);
axis square

% 保存
saveas(gca,'m.png');