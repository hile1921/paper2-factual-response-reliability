function render_full90_atlas(D,C,out)
% Complete observed matrix; query 1 at bottom, query 90 at top.
% Values, shared row order and checkpoint ordering are unchanged.
map=interp1([0 .5 1],[247 164 164;222 65 65;150 0 0]/255,linspace(0,1,6));
zeroColour=[.94 .97 1];fullmap=[zeroColour;map];
ff=paperfig.canvas(5.4);
for i=1:2
    z=D.(['c1_' num2str(i)]);
    aa=paperfig.panel(ff,[.10+(i-1)*.49 .19 .38 .68],char('a'+i-1),z.name);
    vectorheat(aa,round(z.heat*10),fullmap,[0 6]);
    % Override the image/matrix default only for this atlas.
    aa.YDir='normal';aa.Box='on';aa.LineWidth=.65;
    ii=round(linspace(1,numel(z.heat_steps),4));
    set(aa,'XTick',ii,'XTickLabel',compose('%gk',z.heat_steps(ii)),...
        'YTick',[1 30 60 90],'FontSize',8.5);
    xlabel(aa,'Observed checkpoints (ordered)');
    if i==1,ylabel(aa,'Query index (1 at bottom; shared order)');end
end
ca=axes(ff,'Position',[.29 .075 .47 .026],'FontName','Arial','FontSize',8.5);
vectorheat(ca,0:6,fullmap,[0 6]);
set(ca,'YTick',[],'XTick',1:7,'XTickLabel',0:6,'Box','off','TickLength',[0 0]);
paperfig.note(ff,'Pale blue = zero confirmed errors; all 90 queries retained. Zero does not certify correctness.');
paperfig.export(ff,out,'F10_C1_full90_atlas');close(ff);
end
