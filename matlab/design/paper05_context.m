function f=paper05_context(D)
% Pooled observed rates; inferential intervals remain in the contrast figure.
C=edfig.colours();f=edfig.new(4.7);
names={'Pythia-410M','Pythia-1B','OLMo-1B / 1,547B tokens','OLMo-1B / 3,094B tokens'};
for k=1:4
 x=.22+mod(k-1,2)*.44;y=.60-floor((k-1)/2)*.47;
 a=edfig.ax(f,[x y .32 .29],char('a'+k-1),names{k});
 rates=[D.c3.correct(k,:)' D.c3.wrong(k,:)'];
 h=barh(a,1:5,rates,.78,'grouped','EdgeColor','none');
 h(1).FaceColor=C.blue;h(2).FaceColor=C.red;
 set(a,'YDir','reverse','YTick',1:5);ylim(a,[.5 5.5]);xlim(a,[0 110]);xticks(a,[0 50 100]);
 if mod(k,2)==1
  a.YTickLabel={'No context','Filler','True source','Other source','Unbound'};
 else
  a.YTickLabel=[];
 end
 xlabel(a,'Detected answer use (%)');a.XGrid='on';
 % Fixed-size numbers at exact group-bar offsets; no point-area encoding.
 for c=1:5
  for o=1:2
   label=sprintf('%.1f',rates(c,o));
   if rates(c,o)>0 && rates(c,o)<.05,label='<0.1';end
   text(a,rates(c,o)+1,c+(o-1.5)*.39,label,...
    'FontName','Arial','FontSize',8.5,'VerticalAlignment','middle','Color',C.ink);
  end
 end
end
lg=legend(a,h,{'Reference answer','Designated entity'},'Box','off','Orientation','horizontal','FontSize',9);
lg.Position=[.20 .005 .76 .045];
end
