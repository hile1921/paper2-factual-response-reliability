function f=paper06_contrasts(D)
C=edfig.colours();f=edfig.new(6.3);
heads={'Unbound - true relation','Unbound - filler','True relation - filler','True - unrelated source'};
names={'Pythia-410M','Pythia-1B','OLMo / 1,547B','OLMo / 3,094B'};
for k=1:4
 if k==1,pos=[.22 .18 .28 .66];else,pos=[.70 .72-(k-2)*.265 .28 .15];end
 a=edfig.ax(f,pos,char('a'+k-1),heads{k});
 for i=1:4
  for j=1:2
   value=D.c3.delta_pp(i,j,k);if ~isfinite(value),continue;end
   mk={'o','s'};cc=C.blue;if j==2,cc=C.red;end
   edfig.interval(a,value,i+(j-1.5)*.23,D.c3.ci_lo(i,j,k),D.c3.ci_hi(i,j,k),cc,mk{j},D.c3.p_holm(i,j,k)<.05);
  end
 end
 xline(a,0,'--','Color',C.ink);xlim(a,[-10 40]);xticks(a,[-10 0 20 40]);ylim(a,[.5 4.5]);set(a,'YDir','reverse','YTick',1:4,'FontSize',8.5);
 if k==1,a.YTickLabel=names;else,a.YTickLabel={'410M','1B','OLMo early','OLMo late'};end
 if k==1||k==4,xlabel(a,'Difference (pp)');end
 a.XGrid='on';a.TickLength=[0 0];
end
edfig.txt(f,[.20 .93 .33 .045],'Primary comparison',10,'FontWeight','bold');
edfig.txt(f,[.65 .93 .34 .045],'Supporting contrasts',10,'FontWeight','bold');
edfig.txt(f,[.13 .055 .86 .07],sprintf('Blue circle: development    Red square: extension\nOpen marker: Holm p >= 0.05; bars: unadjusted cluster-bootstrap 95%% CI.'),8.5);
end
