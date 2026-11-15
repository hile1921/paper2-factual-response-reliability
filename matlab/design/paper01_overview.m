function f=paper01_overview(D)
C=edfig.colours();f=edfig.new(5.2);
ys=[.655 .36 .065];cols=[C.blue;C.teal;C.red];
heads={sprintf('Training\ntrajectories'),sprintf('Preference /\nanswer alignment'),sprintf('Contextual\nanswer selection')};
for j=1:3
 y=ys(j);annotation(f,'rectangle',[.025 y .95 .25],'Color',cols(j,:),'FaceColor',[.97 .985 1],'LineWidth',.85);
 edfig.txt(f,[.04 y+.17 .19 .065],heads{j},9,'FontWeight','bold','Color',cols(j,:));
 annotation(f,'line',[.245 .245],[y+.025 y+.225],'Color',cols(j,:),'LineWidth',.6);
 annotation(f,'line',[.76 .76],[y+.025 y+.225],'Color',cols(j,:),'LineWidth',.6);
end
% Observed checkpoints -> target-span audit; concrete states from the archive.
y=ys(1);
edfig.txt(f,[.04 y+.02 .19 .13],sprintf('117 checkpoint tasks\n273 fact queries\n319,410 outputs'),8.5);
edfig.txt(f,[.27 y+.195 .47 .035],'The capital of Belarus is ...',9,'FontWeight','bold');
labels={'Nonanswer','Krasnoyarsk','Minsk'};cc=[C.blue;C.red;C.teal];
for k=1:3
 x=.275+(k-1)*.155;
 edfig.card(f,[x y+.095 .135 .065],labels{k},cc(k,:),'w',8.5,'HorizontalAlignment','center');
 edfig.txt(f,[x y+.045 .135 .04],sprintf('%gk steps',D.case_c1.steps(k)),8.5,'HorizontalAlignment','center');
 if k<3,edfig.arrow(f,[x+.138 y+.127],[x+.151 y+.127],C.ink);end
end
edfig.txt(f,[.78 y+.035 .18 .18],sprintf('Readouts\nS: scorable / all\nQ: error / scorable\nE: error / all'),8.5);
% Same fact and prompt, with two measured tracks and an explicit model label.
y=ys(2);
edfig.txt(f,[.04 y+.02 .19 .13],sprintf('42 countries\n3 common stages\nLMEnt-1B-6E'),8.5);
edfig.txt(f,[.27 y+.195 .47 .035],'Same fact + prompt + checkpoint',9,'FontWeight','bold');
edfig.txt(f,[.465 y+.15 .10 .035],'10k',8.5,'HorizontalAlignment','center');
edfig.txt(f,[.625 y+.15 .10 .035],'658k',8.5,'HorizontalAlignment','center');
edfig.txt(f,[.27 y+.10 .18 .035],'Candidate score',8.5);
edfig.txt(f,[.27 y+.04 .18 .035],'Sample s0 name',8.5);
edfig.card(f,[.46 y+.09 .105 .055],sprintf('%.4f',D.case_c2.scores(1)),C.teal,'w',8.5,'HorizontalAlignment','center');
edfig.card(f,[.62 y+.09 .105 .055],sprintf('%.4f',D.case_c2.scores(3)),C.teal,'w',8.5,'HorizontalAlignment','center');
edfig.card(f,[.46 y+.025 .105 .055],'No name',C.blue,'w',8.5,'HorizontalAlignment','center');
edfig.card(f,[.62 y+.025 .105 .055],'Minsk',C.teal,'w',8.5,'HorizontalAlignment','center');
edfig.arrow(f,[.57 y+.117],[.612 y+.117],C.teal);edfig.arrow(f,[.57 y+.052],[.612 y+.052],C.teal);
edfig.txt(f,[.78 y+.035 .18 .18],sprintf('Pair changes\nby country\nrho = %.3f\n42 countries',D.c2.rho(1)),8.5);
% Two context constructions feed separate fixed-checkpoint runs.
y=ys(3);
edfig.txt(f,[.04 y+.02 .19 .13],sprintf('46 extension pairs\n29 target countries\n4 model settings'),8.5);
edfig.txt(f,[.27 y+.195 .47 .035],'The capital of Tuvalu is ...',9,'FontWeight','bold');
edfig.card(f,[.275 y+.095 .28 .09],sprintf('New Caledonia\ncapital: Noumea'),C.blue,'w',8.5,'HorizontalAlignment','center');
edfig.card(f,[.275 y+.015 .28 .065],'Unbound city: Noumea',C.red,'w',8.5,'HorizontalAlignment','center');
edfig.card(f,[.62 y+.065 .105 .105],sprintf('Fixed\nweights'),C.blue,'w',8.5,'HorizontalAlignment','center');
edfig.arrow(f,[.56 y+.14],[.612 y+.135],C.blue);edfig.arrow(f,[.56 y+.047],[.612 y+.095],C.red);
edfig.txt(f,[.78 y+.025 .18 .195],sprintf('Separate runs\nDetect name use\nReference entity\nDesignated entity'),8.5);
end
