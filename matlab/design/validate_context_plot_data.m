function validate_context_plot_data(D)
% Check every numeric slice before creating graphics objects.
Z=D.context_dist;
for outcome={'wrong','correct'}
    P=Z.(['pair_' outcome{1}]);
    Q=Z.(['country_' outcome{1}]);
    assert(isnumeric(P) && isequal(size(P),[4 46 5]),'Invalid pair-rate dimensions.');
    assert(isnumeric(Q) && isequal(size(Q),[4 29 5]),'Invalid country-rate dimensions.');
    assert(all(isfinite(P(:))) && all(P(:)>=0 & P(:)<=100),'Invalid pair-rate values.');
    assert(all(isfinite(Q(:))) && all(Q(:)>=0 & Q(:)<=100),'Invalid country-rate values.');
    for k=1:4
        xy=squeeze(P(k,:,[3 5]));
        assert(isequal(size(xy),[46 2]),'Invalid scatter coordinates.');
        [uv,~,g]=unique(xy,'rows');
        n=accumarray(g,1);
        assert(size(uv,2)==2 && numel(n)==size(uv,1) && sum(n)==46,'Invalid marker multiplicities.');
        delta=sort(squeeze(Q(k,:,5)-Q(k,:,3)));
        assert(numel(delta)==29 && all(isfinite(delta(:))),'Invalid country differences.');
        countryRates=squeeze(Q(k,:,:));
        assert(isequal(size(countryRates),[29 5]),'Invalid condition-distribution coordinates.');
    end
end
assert(isnumeric(D.c3.wrong) && isequal(size(D.c3.wrong),[4 5]),'Invalid pooled-rate dimensions.');
assert(all(isfinite(D.c3.wrong(:))) && all(D.c3.wrong(:)>=0 & D.c3.wrong(:)<=100),'Invalid pooled rates.');
fprintf('Context data preflight passed: 46 pairs, 29 countries, 4 settings, 5 conditions.\n');
end
