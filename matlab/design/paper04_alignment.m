function f=paper04_alignment(D)
C=edfig.colours();f=edfig.new(4.6);
fields={'delta_mass_sum','delta_mass_mean'};heads={'Total sequence LL','Mean token LL'};shape={'o','o'};
for j=1:2
 titletext=sprintf('%s\nrho %.3f [%.3f, %.3f]',heads{j},D.c2.rho(j),D.c2.rho_ci(j,1),D.c2.rho_ci(j,2));
 a=edfig.ax(f,[.13+(j-1)*.49 .60 .33 .24],char('a'+j-1),titletext);
 for k=1:2
  use=logical(D.c2.original);if k==2,use=~use;end;cols=[C.blue;C.cyan];
  h(k)=scatter(a,D.c2.(fields{j})(use),100*D.c2.delta_verified_correct_rate(use),18,cols(k,:),shape{k},'filled');
 end
 xline(a,0,'Color',C.grid);yline(a,0,'Color',C.grid);ylim(a,[-60 110]);xlabel(a,'Preference change');ylabel(a,'Answer change (pp)');
end
lg=legend(a,h,{'Original cohort','Confirmation cohort'},'Box','off','Orientation','horizontal','FontSize',8.5);lg.Position=[.24 .48 .72 .045];
for j=1:2
 heads={'Candidate development','Verified initial claims'};a=edfig.ax(f,[.13+(j-1)*.49 .16 .33 .24],char('c'+j-1),heads{j});
 if j==1
  z=D.c2.sample_mass_sum;errorbar(a,D.c2.steps,z(:,1),z(:,1)-z(:,2),z(:,3)-z(:,1),'-o','Color',C.blue,'MarkerFaceColor',C.blue,'LineWidth',1.2);ylabel(a,'Candidate score');
 else
  z=D.c2.sample_verified_correct_rate;h1=errorbar(a,D.c2.steps,z(:,1)*100,(z(:,1)-z(:,2))*100,(z(:,3)-z(:,1))*100,'-o','Color',C.blue,'MarkerFaceColor',C.blue,'LineWidth',1.2);
  z=D.c2.greedy_verified_correct_rate;h2=errorbar(a,D.c2.steps,z(:,1)*100,(z(:,1)-z(:,2))*100,(z(:,3)-z(:,1))*100,'-o','Color',C.cyan,'MarkerFaceColor',C.cyan,'LineWidth',1.2);
  ylabel(a,'Responses (%)');lg=legend(a,[h1 h2],{'Sampled','Greedy'},'Box','off','FontSize',9,'Location','northwest');
 end
 set(a,'XTick',D.c2.steps,'XTickLabel',{'10','340','658'});xlabel(a,'Step (thousands)');a.YGrid='on';
end
end
