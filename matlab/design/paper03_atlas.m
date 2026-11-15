function f=paper03_atlas(D)
C=edfig.colours();f=edfig.new(7.8);fields={'mass_sum','verified_correct_rate','literal_gold_rate'};
heads={sprintf('Reference-candidate\nscore'),sprintf('Verified initial\nclaim'),sprintf('Name-selection\ndiagnostic')};
endcols=[C.blue;C.teal;C.purple];
for j=1:3
 map=interp1([0 1],[.97 .98 1;endcols(j,:)],linspace(0,1,256));
 a=edfig.ax(f,[.285+(j-1)*.23 .18 .185 .70],char('a'+j-1),heads{j});
 vectorheat(a,D.c2.(fields{j}),map,[0 1]);a.FontSize=8.5;
 set(a,'XTick',1:3,'XTickLabel',{'10k','340k','658k'},'YTick',1:42,'TickLength',[0 0]);
 if j==1,a.YTickLabel=D.c2.subjects;else,a.YTickLabel=[];end
 ca=axes(f,'Position',[.285+(j-1)*.23 .14 .185 .018],'FontName','Arial','FontSize',8.5);
 vectorheat(ca,linspace(0,1,64),map,[0 1]);set(ca,'XTick',[1 64],'XTickLabel',{'0','1'},'YTick',[],'TickLength',[0 0],'Box','off');
end

end
