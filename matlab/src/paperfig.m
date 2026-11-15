classdef paperfig
    % Shared native MATLAB graphics. No Python images are loaded or embedded.
    methods(Static)
        function C=style()
            C.green=[20 133 119]/255; C.red=[237 0 0]/255;
            C.blue=[0 70 139]/255; C.cyan=[77 187 213]/255;
            C.ink=[36 59 74]/255; C.grid=[231 236 239]/255;
            C.models=[C.green;C.red;C.cyan;C.blue];
            C.correct=interp1([0 .5 1],[244 247 246;165 215 201;20 133 119]/255,linspace(0,1,256));
            C.error=interp1([0 .5 1],[255 245 245;245 163 163;237 0 0]/255,linspace(0,1,256));
        end
        function f=canvas(h)
            f=figure('Visible','off','Color','w','Units','inches','Position',[1 1 7.2 h]);
            f.PaperPositionMode='auto';f.InvertHardcopy='off';
        end
        function a=panel(f,pos,letter,heading)
            a=axes('Parent',f,'Position',pos,'FontName','Arial','FontSize',7.5,...
                'LineWidth',.65,'Box','off','TickDir','out','XColor',[.35 .42 .47],'YColor',[.35 .42 .47]);
            hold(a,'on');a.GridColor=[.82 .87 .91];a.GridAlpha=.45;
            a.XLabel.FontSize=8;a.YLabel.FontSize=8;
            title(a,heading,'Interpreter','none','FontSize',8.5,'FontWeight','normal');
            text(a,-.035/pos(3),1.08,letter,'Units','normalized','Clipping','off',...
                'FontName','Arial','FontSize',11,'FontWeight','bold','Color',[36 59 74]/255);
        end
        function txt(f,pos,str,size,varargin)
            annotation(f,'textbox',pos,'String',str,'Interpreter','none','FontName','Arial',...
                'FontSize',size,'Color',[36 59 74]/255,'EdgeColor','none','Margin',0,varargin{:});
        end
        function note(f,str)
            paperfig.txt(f,[.08 .005 .90 .04],str,6.5);
        end
        function grid(a)
            a.YGrid='on';a.Layer='bottom';
        end
        function interval(a,x,y,lo,hi,c,mk,filled)
            plot(a,[lo hi],[y y],'-','Color',c,'LineWidth',1.0);
            plot(a,[lo lo],[y-.04 y+.04],'-','Color',c,'LineWidth',.8);
            plot(a,[hi hi],[y-.04 y+.04],'-','Color',c,'LineWidth',.8);
            face='w';if filled,face=c;end
            plot(a,x,y,mk,'MarkerSize',4.5,'MarkerEdgeColor',c,'MarkerFaceColor',face,'LineWidth',1);
        end
        function box(a,rect,str,edge,fill,fs)
            rectangle(a,'Position',rect,'Curvature',.10,'EdgeColor',edge,'FaceColor',fill,'LineWidth',.9);
            text(a,rect(1)+rect(3)/2,rect(2)+rect(4)/2,str,'HorizontalAlignment','center',...
                'VerticalAlignment','middle','Interpreter','none','FontSize',fs,'FontName','Arial','Color',[36 59 74]/255);
        end
        function arrow(a,p,q,c)
            quiver(a,p(1),p(2),q(1)-p(1),q(2)-p(2),0,'Color',c,'LineWidth',1,'MaxHeadSize',.3);
        end
        function report=export(f,folder,name)
            drawnow;
            savefig(f,fullfile(folder,[name '.fig']));
            exportgraphics(f,fullfile(folder,[name '.pdf']),'ContentType','vector','BackgroundColor','white');
            exportgraphics(f,fullfile(folder,[name '.png']),'Resolution',200,'BackgroundColor','white');
            exportgraphics(f,fullfile(folder,[name '.tiff']),'Resolution',1000,'BackgroundColor','white');
            exportgraphics(f,fullfile(folder,[name '.svg']),'ContentType','vector','BackgroundColor','white');
            report=struct('figure',name,'engine','MATLAB','version',version,'status','complete');
        end
    end
end
