function bound = inbounds(sz, row, col)
m = sz(1);
n = sz(2);
if row > 0 && row <= m
    rowbound = true;
else
    rowbound = false;
end
if col > 0 && col <= n
    colbound = true;
else
    colbound = false;
end
bound = rowbound && colbound;
end