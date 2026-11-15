classdef edfig
 methods(Static)
  function C=colours()
   C.blue=[0 70 139]/255;C.red=[213 42 48]/255;C.teal=[0 137 123]/255;
   C.cyan=[58 165 190]/255;C.purple=[118 85 159]/255;
   C.ink=[37 55 70]/255;C.pale=[.94 .97 1];C.grid=[.88 .92 .95];
   C.conditions=[C.ink;C.cyan;C.blue;C.purple;C.red];
  end
  function f=new(h)
   f=figure('Visible','off','Color','w','Units','inches','Position',[1 1 5.65 h]);
   f.PaperPositionMode='auto';f.InvertHardcopy='off';
  end
  function a=ax(f,pos,letter,heading)
   C=edfig.colours();a=axes(f,'Position',pos,'FontName','Arial','FontSize',9,...
    'TickDir','out','Box','off','LineWidth',.65,'XColor',C.ink,'YColor',C.ink);
   hold(a,'on');a.GridColor=C.grid;a.GridAlpha=1;a.XLabel.FontSize=9;a.YLabel.FontSize=9;
   title(a,heading,'FontSize',9.5,'FontWeight','normal','Interpreter','none');
   text(a,-.035/pos(3),1.11,letter,'Units','normalized','Clipping','off','FontSize',11,'FontWeight','bold');
  end
  function t=txt(f,pos,str,fs,varargin)
   C=edfig.colours();t=annotation(f,'textbox',pos,'String',str,'FontName','Arial','FontSize',fs,...
    'Color',C.ink,'Interpreter','none','EdgeColor','none','Margin',0,varargin{:});
  end
  function card(f,pos,str,edge,fill,fs,varargin)
   annotation(f,'rectangle',pos,'Color',edge,'FaceColor',fill,'LineWidth',.8);
   pp=pos+[.009 .008 -.018 -.016];edfig.txt(f,pp,str,fs,varargin{:});
  end
  function arrow(f,p,q,c)
   annotation(f,'arrow',[p(1) q(1)],[p(2) q(2)],'Color',c,'LineWidth',.9,'HeadLength',4,'HeadWidth',4);
  end
  function s=hi(s,word,c)
   tag=['{\bf\color[rgb]{' sprintf('%.3f %.3f %.3f',c) '}' word '}'];
   s=strrep(s,word,tag);
  end
  function interval(a,x,y,lo,hi,c,mk,filled)
   plot(a,[lo hi],[y y],'-','Color',c,'LineWidth',1);
   plot(a,[lo lo],[y-.045 y+.045],'-','Color',c,'LineWidth',.75);
   plot(a,[hi hi],[y-.045 y+.045],'-','Color',c,'LineWidth',.75);
   fc='w';if filled,fc=c;end
   plot(a,x,y,mk,'MarkerSize',4.5,'MarkerEdgeColor',c,'MarkerFaceColor',fc,'LineWidth',.9);
  end
  function export(f,out,name)
   drawnow;savefig(f,fullfile(out,[name '.fig']));
   exportgraphics(f,fullfile(out,[name '.pdf']),'ContentType','vector','BackgroundColor','white');
   exportgraphics(f,fullfile(out,[name '.svg']),'ContentType','vector','BackgroundColor','white');
   exportgraphics(f,fullfile(out,[name '.png']),'Resolution',220,'BackgroundColor','white');
   close(f);
  end
 end
end
