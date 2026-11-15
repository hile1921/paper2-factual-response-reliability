function f=paper02_training(D)
C=edfig.colours();f=edfig.new(2.8);
fields={'S','Q','E'};heads={'Scorable / all','Error / scorable','Error / all'};
for j=1:3
 a=edfig.ax(f,[.10+(j-1)*.32 .23 .235 .55],char('a'+j-1),heads{j});
 h1=plot(a,D.c1_1.step,D.c1_1.(fields{j}),'-','Color',C.blue,'LineWidth',1.25);
 h2=plot(a,D.c1_2.step,D.c1_2.(fields{j}),'-','Color',C.red,'LineWidth',1.25);
 xlim(a,[0 143]);xticks(a,[0 70 143]);xlabel(a,'Step (thousands)');ylabel(a,'Responses (%)');a.YGrid='on';
 if j==1
  v=[D.c1_1.S(:);D.c1_2.S(:)];ylim(a,[0 max(5,ceil(max(v)/5)*5)]);
 elseif j==2
  ylim(a,[0 100]);
 else
  v=[D.c1_1.E(:);D.c1_2.E(:)];ylim(a,[0 max(.5,ceil(max(v)/.5)*.5)]);
 end
end
lg=legend(a,[h1 h2],{'Pythia-410M','Pythia-1B'},'Box','off','Orientation','horizontal','FontSize',9);lg.Position=[.24 .90 .68 .055];
end
