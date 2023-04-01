clear all;
close all;
tree = table2array(readtable('/home/akshay/Downloads/optimization/project/forest/7.csv'));
size(tree);

tree = tree/1000;

% s_0 = [0.1, 0.2, 0, 0, 0.5];
x_c = mean(tree, 1);
x_c
s_0 = [x_c(1), x_c(2), 0, 0, 0.51122];

res1 = lsqnonlin(@(s)myobj_least_squares(s, tree), s_0);
res1(1), res1(2), rad2deg(res1(3)), rad2deg(res1(4)), res1(5)

function [obj]=myobj(p, t)
% x0 = s[1]
% x1 = s[2]
% s[3] alpha: about x-axis
% s[4] beta: about y-axis
% s[5] radius
% - np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) 
% - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2

% obj = sum( ( (-cos(s(4))*(s(1)-t(:,1))- t(:, 3)*cos(s(3))*sin(s(4))- ...
%               sin(s(3))*sin(s(4))*(s(2)-t(:,2)).^2+ ...
%              (t(:,3)*sin(s(3))-cos(s(3))*(s(2)-t(:, 2))).^2 ) - s(5)^2 ) .^2);
% obj = 
% obj=sum((s(3)^2 - (t(:, 1) - s(1)).^2 - (t(:,2) - s(2)).^2).^2);
obj = 0;
for i = 1:size(t)[0]
    point = t(i, :);
    x = point(1);
    y = point(2);
    z = point(3);
    obj = obj + ((-cos(p(4)) * (p(1) - x) - z*cos(p(3))*sin(p(4)) - sin(p(3))*sin(p(4))*(p(2)-y))^2 + (z*sin(p(3))-cos(p(3))*(p(2)-y))^2-p(5)^2)^2;

% disp(disp_str)
% X = [name,' will be ',num2str(age),' this year.'];
disp(obj)
end
end


function [obj]=myobj_least_squares(p, t)
% x0 = s[1]
% x1 = s[2]
% s[3] alpha: about x-axis
% s[4] beta: about y-axis
% s[5] radius

obj = [];
for i = 1:size(t)[0]
    point = t(i, :);
    x = point(1);
    y = point(2);
    z = point(3);
    obj_curr = (-cos(p(4)) * (p(1) - x) - z*cos(p(3))*sin(p(4)) - sin(p(3))*sin(p(4))*(p(2)-y))^2 + (z*sin(p(3))-cos(p(3))*(p(2)-y))^2-p(5)^2;
    obj(end+1) = obj_curr;
% disp(obj__curr)
end
end
