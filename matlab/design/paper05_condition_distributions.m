function f=paper05_condition_distributions(D)
% Each row is one country; numeric values and colours share the same exact data.
C=edfig.colours();Z=D.context_dist;f=edfig.new(5.8);
map=interp1([0 .5 1],[C.blue;1 1 1;C.red],linspace(0,1,257));
names=Z.countries;
for i=1:numel(names)
 names{i}=strrep(names{i},'Democratic Republic of the Congo','DR Congo');
 names{i}=strrep(names{i},'Trinidad and Tobago','Trinidad & Tobago');
 names{i}=strrep(names{i},'People''s Republic of China','China');
end
outcomes={'wrong','correct'};heads={'Designated-entity change','Reference-answer change'};
for j=1:2
 a=edfig.ax(f,[.245+(j-1)*.39 .20 .34 .69],char('a'+j-1),heads{j});
 P=Z.(['country_' outcomes{j}]);V=squeeze(P(:,:,5)-P(:,:,3))';
 assert(isequal(size(V),[29 4]),'Invalid country effect matrix.');
 vectorheat(a,V,map,[-100 100]);
 for row=1:29
  for col=1:4
   colour=C.ink;if abs(V(row,col))>=60,colour=[1 1 1];end
   text(a,col,row,sprintf('%.1f',V(row,col)),'HorizontalAlignment','center',...
    'VerticalAlignment','middle','FontName','Arial','FontSize',8.5,'Color',colour);
  end
 end
 set(a,'YTick',1:29,'FontSize',8.5,'XTick',1:4,'XTickLabel',...
  {'410M','1B','1,547B','3,094B'},'TickLength',[0 0]);
 if j==1,a.YTickLabel=names;else,a.YTickLabel=[];end
end
ca=axes(f,'Position',[.40 .095 .40 .018],'FontName','Arial','FontSize',9);
vectorheat(ca,linspace(-100,100,201),map,[-100 100]);
set(ca,'YTick',[],'XTick',[1 101 201],'XTickLabel',{'-100','0','+100'},'TickLength',[0 0]);
xlabel(ca,'Unbound - true source relation (pp)','FontSize',9);
end
