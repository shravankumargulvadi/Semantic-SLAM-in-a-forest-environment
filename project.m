clear all;
close all;
tree = table2array(readtable('tree.csv'));
size(tree);

tree = tree/1000;


fun = @(s) sum((s(3)^2 - (tree(:, 1) - s(1)).^2 - (tree(:,2) - s(2)).^2).^2);
s_0 = mean(tree, 1);
s_0 = [s_0(1)*10, s_0(2)/10, 10];
res = fmincon(@(s)myobj(s, tree), s_0, [0, 0, 0; ...
                                        0, 0, 0; ...
                                        0, 0, -1], [0, 0, 0], [], [], [], [], []);
res

% - min(tree(:, 1)) + max(tree(:, 1))
% - min(tree(:, 2)) + max(tree(:, 2))
% min(tree(:, 3)), max(tree(:, 3))
% figure
% % hold on;
% scatter3(tree(:, 1), tree(:, 2), tree(:, 3))
% opts=optimoptions('fmincon', 'MaxFunctionEvaluations',1);
% s_0 = [0.1, 0.2, 0.2, 0.5, 5];
% res2 = fmincon(@(s)myobj_vijay_raw(s, tree), s_0, [-1, 0, 0, 0, 0; ...
%                                                    0, 0, 0, 0, 0; ...
%                                                    0, 0, 0, 0, 0; ...
%                                                    0, 0, 0, 0, 0; ...
%                                                    0, 0, 0, 0, -1], [0, 0, 0, 0, 0], [], [], [], [], []);
% res2
% myobj_vijay([0, 0, 0, 0, 0], tree);


function [obj]=myobj(s, t)
obj=sum((s(3)^2 - (t(:, 1) - s(1)).^2 - (t(:,2) - s(2)).^2).^2);
end

function [obj]=myobj_vijay(s, t);
% rho = s[1]
% phi = s[2]
% v = s[3]
% alpha = s[4]
% K = s[5]

n = [cos(s(2))*sin(s(3)), sin(s(2))*sin(s(3)), cos(s(3))];
n_v = [cos(s(2))*cos(s(3)), sin(s(2))*cos(s(3)), -sin(s(3))];
n_phi = [-sin(s(2)), cos(s(2)), 0];
a = n_v*cos(s(4)) + n_phi*sin(s(4));


% disp_str = ['a', a, 'n_v', n_v, 'a_size', size(a)];
% % size(a), a, n_v
% disp(disp_str)
% X = [name,' will be ',num2str(age),' this year.'];
% disp(X)
obj = 0;
for i = 1:size(t)[0]
    point = t(i, :);
%     point, n, dot(point, n)
    obj = obj + (((s(5)/2)*norm(point)^2 -2*s(1)*dot(point, n) - dot(point, a)^2 + s(1)^2) + s(1) - dot(point, n))^2;

% disp(disp_str)
% X = [name,' will be ',num2str(age),' this year.'];
disp(obj)
end
end

function [obj]=myobj_vijay_raw(s, t);
% rho = s[1]
% phi = s[2]
% v = s[3]
% alpha = s[4]
% K = s[5]

n = [cos(s(2))*sin(s(3)), sin(s(2))*sin(s(3)), cos(s(3))];
n_v = [cos(s(2))*cos(s(3)), sin(s(2))*cos(s(3)), -sin(s(3))];
n_phi = [-sin(s(2)), cos(s(2)), 0];
a = n_v*cos(s(4)) + n_phi*sin(s(4));


% disp_str = ['a', a, 'n_v', n_v, 'a_size', size(a)];
% % size(a), a, n_v
% disp(disp_str)
% X = [name,' will be ',num2str(age),' this year.'];
% disp(X)
obj = 0;
for i = 1:size(t)[0]
    point = t(i, :);
%     point, n, dot(point, n)
    obj = obj + (norm(cross(point-(s(1)+(1/s(5)))*n, a)) - (1/s(5)))^2;
disp(obj)
end
end

