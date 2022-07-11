% mat =importdata("LA_EER.txt");
% data = mat.data;
% textdata = mat.textdata;
% index_x={"LA-C1","LA-C2","LA-C3","LA-C4","LA-C5","LA-C6","LA-C7"};
% index_y={"A07","A08","A09","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"};
% index_y={"A09","A13","A16","A12","A07","A14","A08","A15","A11","A19","A10","A17","A18"};
% index_y=flip(index_y);
% b = bar3(sort(data,"descend"));
% % for k = length(b):-1:1
% %     zdata = b(k).ZData;
% %     b(k).CData = zdata;
% %     b(k).FaceColor = 'interp';
% % end
% xlabel("Channels",'FontName','Times New Roman');
% ylabel("Attacks",'FontName','Times New Roman');
% zlabel("EER(%)",'FontName','Times New Roman');
% set(gca,'FontName','Times New Roman');
% set(gcf,'Position',[0,0,2000,2000]);
% ax=gca;
% ax.XTickLabel=index_x;
% ax.YTickLabel=index_y;
% 

mat =importdata("LA_min_tDCF.txt");
data = mat.data;
textdata = mat.textdata;
index_x={"LA-C1","LA-C7","LA-C4","LA-C2","LA-C5","LA-C6","LA-C3"};
index_x={"LA-C1","LA-C2","LA-C3","LA-C4","LA-C5","LA-C6","LA-C7"};
index_y={"A16","A14","A07","A12","A13","A15","A11","A08","A10","A19","A09","A18","A17"};
index_y=flip(index_y);
data = sort(data,"descend")
b = bar3(data);
% for k = length(b):-1:1
%     zdata = b(k).ZData;
%     b(k).CData = zdata;
%     b(k).FaceColor = 'interp';
% end
xlabel("Channels",'FontName','Times New Roman');
ylabel("Attacks",'FontName','Times New Roman');
zlabel("min-tDCF",'FontName','Times New Roman');
set(gca,'FontName','Times New Roman');
set(gcf,'Position',[0,0,2000,2000]);
ax=gca;
ax.XTickLabel=index_x;
ax.YTickLabel=index_y;


% mat =importdata("DF-EER.txt");
% data = mat.data;
% textdata = mat.textdata;
% index_x={"LA-C6","LA-C7","LA-C8","LA-C3","LA-C1","LA-C2","LA-C9","LA-C5","LA-C4"};
% index_x={"LA-C1","LA-C2","LA-C3","LA-C4","LA-C5","LA-C6","LA-C7","LA-C8","LA-C9"};
% index_y={"A02","A05","A01","A03","A04"};
% index_y=flip(index_y);
% b = bar3(sort(data,"descend"));
% % for k = length(b):-1:1
% %     zdata = b(k).ZData;
% %     b(k).CData = zdata;
% %     b(k).FaceColor = 'interp';
% % end
% xlabel("Channels",'FontName','Times New Roman');
% ylabel("Attacks",'FontName','Times New Roman');
% zlabel("EER(%)",'FontName','Times New Roman');
% set(gca,'FontName','Times New Roman');
% set(gcf,'Position',[0,0,2000,2000]);
% ax=gca;
% ax.XTickLabel=index_x;
% ax.YTickLabel=index_y;