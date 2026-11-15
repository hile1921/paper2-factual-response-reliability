function h = vectorheat(ax,Z,map,limits)
% One opaque flat face per observed data cell. No interpolation or imagesc.
% NaNs are omitted; zero remains an observed zero, never a missing value.
[nr,nc]=size(Z);[col,row]=meshgrid(1:nc,1:nr);
col=col(:);row=row(:);value=Z(:);ok=isfinite(value);
col=col(ok);row=row(ok);value=value(ok);n=numel(value);
vertices=zeros(4*n,2);
vertices(1:4:end,:)=[col-.5,row-.5];
vertices(2:4:end,:)=[col+.5,row-.5];
vertices(3:4:end,:)=[col+.5,row+.5];
vertices(4:4:end,:)=[col-.5,row+.5];
faces=reshape(1:4*n,4,n)';
idx=1+round((value-limits(1))/(limits(2)-limits(1))*(size(map,1)-1));
idx=max(1,min(size(map,1),idx));
h=patch(ax,'Vertices',vertices,'Faces',faces,'FaceVertexCData',map(idx,:),...
    'FaceColor','flat','EdgeColor','none','FaceAlpha',1);
xlim(ax,[.5 nc+.5]);ylim(ax,[.5 nr+.5]);ax.YDir='reverse';
end
