clear all
close all
clc

settings.history1video = 1; % History for first stage
settings.history2video = 1; % History for second stage
settings.axesguess = 0; % Cartesian axes or principal components for initial guesses
settings.testname = 'test51'; % name of the folder where the videos will be stored

%% Load point cloud

num_p1 = 150; % Points in stage 1
num_p2 = 1000; % Points in stage 2

realcloud = pcread(['rolltall2.pcd']);
realcloud1 = pcdownsample(realcloud, 'random', num_p1/realcloud.Count); % Downsampling to 150 points
realcloud2 = pcdownsample(realcloud, 'random', num_p2/realcloud.Count); % Downsampling to 2k points
points1 = double(realcloud1.Location);
points2 = double(realcloud2.Location);

%% Center point cloud and get initial guesses and bounds for the first optimization

center = (max(points2)+min(points2))/2;
amp = norm((max(points2)-min(points2))/2);
points2 = (points2 - center)./amp;
points1 = (points1 - center)./amp;

if settings.axesguess
    comps = pca(points1);
else
    comps = [1 0 0; 0 1 0; 0 0 1];
end

theta = acos(comps(:,3)./...
    sqrt(comps(:,1).^2 + comps(:,2).^2 + comps(:,3).^2));
phi = atan2(comps(:,2),comps(:,1));

% Initial guesses 1
x0 = [phi theta repmat([0 0 0 1 1/4 1/2 0.5],[3,1])]; % [phi theta psi xc yc re si so eps]

% Bounds

lb = [phi - pi/2, theta - pi/2, repmat([-pi, -1/2.*10, -1/2.*10, 0.001, 0, 0, 0.1],[3,1])];
ub = [phi + pi/2, theta + pi/2, repmat([pi, 1/2.*10, 1/2.*10, 1000, 1/2.*3, 1/2.*3, 2.1],[3,1])];

%% First optimization stage

start_time1 = tic;

% options1 = optimoptions('lsqnonlin','Algorithm','interior-point',...
%     'CheckGradients',true,'FunctionTolerance',1e-2,'StepTolerance',...
%     1e-4,'InitDamping',1e-3,'FinDiffType','central','DiffMinChange',1e-7,...
%     'MaxIterations',15000,'MaxFunctionEvaluations',20000, ...
%     'Display', 'iter-detailed', 'OutputFcn', @hist);

options1 = optimoptions('lsqnonlin','Algorithm','interior-point',...
    'CheckGradients',true,'FunctionTolerance',1e-2,'StepTolerance',...
    1e-3,'FinDiffType','forward','DiffMinChange',1e-7,...
    'MaxIterations',15000,'MaxFunctionEvaluations',20000, ...
    'Display', 'iter-detailed', 'OutputFcn', @hist, 'BarrierParamUpdate', ...
    'predictor-corrector');

% n_w = [1 0.1 1 0.1 0.5]; % Weights penalize:
% n_w1 = [0.1 0.01 0.2 0.02 5 0.5]; % Weights penalize:
% n_w1 = [0.1 0 0.2 0 5 0.5]; % Weights penalize:
n_w1 = [0.25 0.1 0.5 0.1 50 0.1]; % Weights penalize:

% [distance from points to outer superellipse if they are outside;
% distance from points to outer superellipse if they are inside;
% distance from points to inner superellipse if they are inside;
% distance from points to inner superellipse if they are outside;
% relative quarter population
% relative scale of superellipses (maximize inner superellipse)];

objfun1 = @(param)J1(points1,param,n_w1);


sol1 = zeros(length(theta),9);
objectiveValue1 = zeros(length(theta),1);
exitflag1 = zeros(length(theta),1);
h1 = {[],[],[]};

for i = 1:length(theta)
    global resultsLog
    % x0 = [orientations(i,1) orientations(i,2) 0 0 0 1 amp/2 amp 0.5]; % [phi theta psi xc yc re si so eps]
    [sol1(i,:),objectiveValue1(i),~,exitflag1(i)] = ...
        lsqnonlin(objfun1,x0(i,:), lb(i,:), ub(i,:), options1);

    h1{i} = resultsLog;

    resultsLog = [];
end

[~,solindex] = min(objectiveValue1);
solution = sol1(solindex,:); % index with minimal cost value

time1 = toc(start_time1);

%% Second optimization stage

start_time2 = tic;


% c0 = [...
%     cos(solution(2)) 0 sin(solution(2));
%     0 1 0;
%     -sin(solution(2)) 0 cos(solution(2))]*[...
%     1 0 0;
%     0 cos(solution(1)) -sin(solution(1));
%     0 sin(solution(1)) cos(solution(1))]*inv([...
%     cos(solution(3)) -sin(solution(3)) 0;
%     sin(solution(3)) cos(solution(3)) 0;
%     0 0 1])*[-solution(4);-solution(5);0];

% qv0 = quatmultiply(eul2quat([0 solution(2) solution(1)]),eul2quat([solution(3) 0 0]));
% qv0(2:4) = -qv0(2:4);

qv0 = quatmultiply(eul2quat([solution(3) 0 0]), eul2quat([0 0 -solution(1)]));
qv0 = quatmultiply(qv0, eul2quat([0 -solution(2) 0]));

c0 = [solution(4); solution(5); 0];

% qv0 = eul2quat();

global resultsLog

options2 = optimoptions('lsqnonlin','Algorithm','interior-point',...
    'CheckGradients',true,'FunctionTolerance',1e-5,'StepTolerance',...
    1e-3,'InitDamping',1e3,'FinDiffType','forward','DiffMinChange',1e-7,...
    'MaxIterations',15000,'MaxFunctionEvaluations',20000, ...
    'Display', 'iter-detailed', 'OutputFcn', @hist); %...
% 'SubProblemAlgorithm', 'cg');

% a0 = [solution(8)-solution(7), (solution(8)-solution(7))*solution(6), ...
%     1/2, solution(8)/(solution(8) - solution(7)) - 1]; % Initial guess of the second stage

a0 = [(solution(8)-solution(7))/2, (solution(8)-solution(7))/2*solution(6), ...
    0.6, 2*solution(7)/(solution(8) - solution(7)) + 1]; % Initial guess of the second stage

a0([1 2 4]) = a0([1 2 4]).*0.8;
% a0(4) = a0(4)./0.8;

% x02 = [a0 0.4 solution(9) -c0.' qv0]; %[a1 a2 a3 a4 e1 e2 xc yc zc q0 q1 q2 q3]
x02 = [a0 1/2 solution(9) c0.' qv0]; %[a1 a2 a3 a4 e1 e2 xc yc zc q0 q1 q2 q3]
lb2 = [a0(1:3)./3 1 0.1 0.05 c0.'-1/2 -1 -1 -1 -1];
% lb2 = [a0(1:3)./3 0 0.1 0.05 c0.'-amp -1 -1 -1 -1];
ub2 = [a0(1:3).*3 20 2.1 2.1 c0.'+1/2 1 1 1 1];

n_w2 = [0.1 1 10 10 1]; % Weights penalize:

% [distance to surface;
% inverse of a4 (penalize smaller holes);
% scaling factor a1 (penalize bigger ST);
% scaling factor a2 (penalize bigger ST);
% scaling factor a3 (penalize bigger ST)];

objfun2 = @(param)J2(points2,param,n_w2);

[sol2,objectiveValue2,~,exitflag2] = ...
    lsqnonlin(objfun2,x02, lb2, ub2, [], [], [], ...
    [], @nonlcon, options2);

% [sol2,objectiveValue2,~,exitflag2] = ...
%     lsqnonlin(objfun2,x02, [], ub2, [], [], [], ...
%     [], @nonlcon, options2);

% [sol2,objectiveValue2,~,exitflag2] = ...
%     lsqnonlin(objfun2,x02, [], [], [], [], [], ...
%     [], @nonlcon, options2);

h2 = resultsLog;

clear resultsLog;


time2 = toc(start_time2);

%% Postprocessing

disp(['Time phase 1: ', num2str(time1), 's'])
disp(['Time phase 2: ', num2str(time2), 's'])
disp(['Total time: ', num2str(time1+time2), 's'])

% Cost function functionals

n_J1 = J1(points1, solution, n_w1);
n_j11 = sum(n_J1(1:2*num_p1).^2);
n_j12 = sum(n_J1(2*num_p1+1:2*num_p1+8).^2);
n_j13 = sum(n_J1(2*num_p1+8:2*num_p1+9).^2);

n_J2 = J2(points2, sol2,n_w2);
n_j21 = sum(n_J2(1:num_p2).^2);
n_j22 = sum(n_J2(num_p2+1).^2);
n_j23 = sum(n_J2(num_p2+2).^2);
n_j24 = sum(n_J2(num_p2+3).^2);
n_j25 = sum(n_J2(num_p2+4).^2);

% plotResults(points1, solution, n_w);

%% Show results of preliminary optimization

figure(1)
tiledlayout(1,2)
nexttile
plotResults_3D(points1,solution)
nexttile
plotResults(points1,solution, n_w1);

%% Show final results

figure(3)

[dist, p_s, ptransf] = J2(points2,sol2,n_w2);
p_s = real(p_s);
d = p_s-ptransf;

toroidSolution = PlotSupertoroid([sol2(1), sol2(2), sol2(3), sol2(4)], ...
    [sol2(5), sol2(6)], 180);

viewer = pcshow(toroidSolution, 'red');
hold on
pcshow(p_s, 'blue')
pcshow(ptransf, 'green')

figure(4)

[dist, p_s, ptransf] = J2(points2,sol2,n_w2);
p_s = real(p_s);
d = p_s-ptransf;

toroidSolution = PlotSupertoroid([sol2(1), sol2(2), sol2(3), sol2(4)], ...
    [sol2(5), sol2(6)], 180);
% toroidSolution_pc = pointCloud(PlotSupertoroid([sol2(1), sol2(2), sol2(3), sol2(4)], ...
%     [sol2(5), sol2(6)], 180));n_w1

solt = scatter3(toroidSolution(:,1), toroidSolution(:,2), toroidSolution(:,3), 'red', '.');
hold on
axis equal
scatter3(p_s(:,1), p_s(:,2), p_s(:,3), 'blue', '.')
scatter3(ptransf(:,1), ptransf(:,2), ptransf(:,3), 'green', 'o')
set(gca,'Color','k')


% plots for the paper
paper1 = figure(10);
tiledlayout(1,5)
nexttile([1,3])
plotResults_3DPaper(points1,solution)
% axis off
% grid on
nexttile([1,2])
plotResultsPaper(points1,solution, n_w1);
axis off
% grid on

figure(11)

[dist, p_s, ptransf] = J2(points2,sol2,n_w2);
p_s = real(p_s);
d = p_s-ptransf;

toroidSolution = PlotSupertoroid([sol2(1), sol2(2), sol2(3), sol2(4)], ...
    [sol2(5), sol2(6)], 180);

solt = scatter3(toroidSolution(:,1), toroidSolution(:,2), toroidSolution(:,3), 'red', '.');
hold on
axis equal
% scatter3(p_s(:,1), p_s(:,2), p_s(:,3), 100, 'blue', '.')
s1 = scatter3(ptransf(:,1), ptransf(:,2), ptransf(:,3), 100, 'green', '.');
set(gca,'Color','k')

xticks([])
yticks([])
zticks([])

figure(12)
pointsall = double(realcloud.Location);
scatter3(pointsall(:,1), pointsall(:,2), pointsall(:,3), '.', 'MarkerEdgeColor', ...
    [12/255 123/255 220/255])
axis equal

figure(13)

pointsall = double(realcloud.Location);
pointsall = (pointsall - center)./amp;

[dist, p_s, ptransf] = J2(pointsall,sol2,n_w2);
p_s = real(p_s);
d = p_s-ptransf;

toroidSolution = PlotSupertoroid([sol2(1), sol2(2), sol2(3), sol2(4)], ...
    [sol2(5), sol2(6)], 180);

solt = scatter3(toroidSolution(:,1), toroidSolution(:,2), toroidSolution(:,3), 'red', '.');
hold on
axis equal
% scatter3(p_s(:,1), p_s(:,2), p_s(:,3), 100, 'blue', '.')
s1 = scatter3(ptransf(:,1), ptransf(:,2), ptransf(:,3), 100, 'green', '.');
set(gca,'Color','k')

xticks([])
yticks([])
zticks([])

keyboard;

%% Videos
% Stage 1

if settings.history1video
    close all
    figure(1)
    mkdir(settings.testname)
    for i = 1:length(theta)

        maxvidt = size(h1{i},2) + 19;
        F = struct('cdata', cell(1,maxvidt), 'colormap', cell(1,maxvidt));

        for s = 1:size(h1{i},2)
            clf
            t = tiledlayout(2,2);
            nexttile(1,[2,1])
            plotResults_3D(points1, h1{i}(:,s))
            nexttile
            [cost, Qs, cost_functionals] = plotResults(points1, h1{i}(:,s), n_w1);
            nexttile
            QuarterName = {'Northeast'; 'Northwest'; 'Southwest'; 'Southeast';
                'East'; 'North'; 'West'; 'South'; 'Total'};
            QuarterAmounts = Qs;
            xval = [1];
            % tbl = table(QuarterName, QuarterAmounts);
            h = heatmap(xval, QuarterName, QuarterAmounts);
            title(t,['cost = ', num2str(cost), '(j_1 = ', ...
                num2str(cost_functionals(1)), ', j_2 = ', ...
                num2str(cost_functionals(2)), ', j_3 = ', ...
                num2str(cost_functionals(3)), ')'])

            drawnow;
            F(s) = getframe(gcf);
        end

        for s = size(h1{i},2)+1:maxvidt
            F(s) = getframe(gcf);
        end
        vid = VideoWriter([settings.testname, '\Stage1sol', ...
            num2str(i),'_Exitf', num2str(exitflag1(i)), '.mp4'],'MPEG-4');
        vid.FrameRate = 10;
        open(vid)
        writeVideo(vid,F)
        close(vid)
    end

end

if settings.history2video
    mkdir(settings.testname)
    close all
    figure(5)
    maxvidt = size(h2,2);
    F = struct('cdata', cell(1,maxvidt), 'colormap', cell(1,maxvidt));
    for i = 1:maxvidt
        [dist2, p_s, p, param_debug] = J2(points2,h2(:,i),n_w2);
        clf
        toroidSolution = PlotSupertoroid([h2(1,i), h2(2,i), h2(3,i), h2(4,i)], ...
            [h2(5,i), h2(6,i)], 180);
        pcshow(toroidSolution, 'red')
        hold on
        pcshow(p_s, 'blue')
        % pcshow(ptransf, 'green')
        pcshow(p, 'green')
        title(['iter ',num2str(i-1),', f(x) = ', num2str(norm(dist2).^2)])

        drawnow;
        F(i) = getframe(gcf);
    end
    vid = VideoWriter([settings.testname, '\Stage2sol',...
        '_Exitf', num2str(exitflag2), '.mp4'],'MPEG-4');
    vid.FrameRate = 2;
    open(vid)
    writeVideo(vid,F)
    close(vid)
end


function createscatter3(X1, Y1, Z1, Size1, Color1)
%CREATESCATTER3(X1, Y1, Z1, Size1, Color1)
%  X1:  vector of scatter3 x data
%  Y1:  vector of scatter3 y data
%  Z1:  vector of scatter3 z data
%  SIZE1:  vector of scatter3 size data
%  COLOR1:  vector of scatter3 color data

%  Auto-generated by MATLAB on 02-Sep-2024 12:28:25

% Create scatter3
scatter3(X1,Y1,Z1,Size1,Color1,'Tag','pcviewer','Marker','o');
end

function [dist, p_si, p_so] = J1(points, x, w)

phi = x(1);
theta = x(2);
psi = x(3);
xc = x(4);
yc = x(5);
re = x(6);
si = x(7);
so = x(8);
eps = x(9);

%% Projection to a plane

v = [...
    cos(theta) 0 sin(theta);
    0 1 0;
    -sin(theta) 0 cos(theta)]*[...
    1 0 0;
    0 cos(phi) -sin(phi);
    0 sin(phi) cos(phi)]*[0;0;1];

t = (-v(1).*(points(:,1)) - v(2).*(points(:,2)) - v(3).*(points(:,3)))/...
    (v(1)^2 + v(2)^2 + v(3)^2);

pointsproj = points + t.*[v(1) v(2) v(3)];

q2d = eul2quat([0 theta phi]);
q2dq = quaternion([q2d(1), -q2d(2), -q2d(3), -q2d(4)]);
pointsproj_rot = rotatepoint(q2dq, pointsproj);

qpsi = eul2quat([psi 0 0]);
qpsiq = quaternion(qpsi);
pointsproj_rot2 = rotatepoint(qpsiq, ...
    [pointsproj_rot(:,1), pointsproj_rot(:,2), zeros(size(pointsproj_rot,1),1)]);

pointsproj_rot2d = [pointsproj_rot2(:,1)+xc, pointsproj_rot2(:,2)+yc];

%% Distance to outer superellipse

eta_pi = atan2(so.*pointsproj_rot2d(:,2), (re*so).*pointsproj_rot2d(:,1));

Re_pi = [sign(cos(eta_pi)).*abs(cos(eta_pi)).^(1/eps), ...
    sign(sin(eta_pi)).*abs(sin(eta_pi)).^(1/eps)];

eta = atan2(Re_pi(:,2), Re_pi(:,1));

p_so = [so.*sign(cos(eta)).*abs(cos(eta)).^(eps), ...
    (so*re).*sign(sin(eta)).*abs(sin(eta)).^(eps)];

d = vecnorm(pointsproj_rot2d,2,2);
do = vecnorm(p_so,2,2);
do_cl = d-do;
dow = weighDistanceOut(do_cl,1000,[w(1) w(2)]);
dowsq = dow.*do_cl;

%% Distance to inner superellipse

p_si = [si.*sign(cos(eta)).*abs(cos(eta)).^(eps), ...
    (si*re).*sign(sin(eta)).*abs(sin(eta)).^(eps)];

di = vecnorm(p_si,2,2);
di_cl = d-di;
diw = weighDistanceIn(di_cl,1000,[w(3) w(4)]);
diwsq = diw.*di_cl;

% scatter(p_si(:,1), p_si(:,2))

dist = [diwsq; dowsq]./length(diwsq)./sqrt(2.5e-7);

% Add contribution of log imbalance and its weights (w)

k = 1000;

isEast = (tanh(k*pointsproj_rot2d(:,1))+1)/2;
isWest = 1-isEast;

isNorth = (tanh(k*pointsproj_rot2d(:,2))+1)/2;
isSouth = 1-isNorth;

% Find which quarter the points are in (NE - NW - SW - SE)

pQ1 = isNorth.*isEast; % Northeast
pQ2 = isNorth.*isWest; % Northwest
pQ3 = isSouth.*isWest; % Southwest
pQ4 = isSouth.*isEast; % Southeast

pT = size(pointsproj_rot2d,1);

isNE = (tanh(k*(pointsproj_rot2d(:,1) + pointsproj_rot2d(:,2)))+1)/2;
isSW = 1 - isNE;

isNW = (tanh(k*(pointsproj_rot2d(:,1) - pointsproj_rot2d(:,2)))+1)/2;
isSE = 1 - isNW;

% Find which quarter the points are in (E - N - W - S)

pQE = isNE.*isSE;
pQN = isNE.*isNW;
pQW = isNW.*isSW;
pQS = isSW.*isSE;

pQ8 = [sum(pQ1); sum(pQ2); sum(pQ3); sum(pQ4); sum(pQE); sum(pQN); sum(pQW); sum(pQS)];

pQ8_processed = (1./(log10(pQ8.*4)/log10(pT) + 0.01) - 1)./sqrt(10.95);

dist = [dist; pQ8_processed.*sqrt(w(5)); so/si*sqrt(w(6))];

end

function [dist, p_s, p, param_debug] = J2(xin,param, n_w)
param_debug = param;
a1=param(1);
a2=param(2);
a3=param(3);
a4=param(4);
e1=param(5);
e2=param(6);
xc = param(7);
yc = param(8);
zc = param(9);
q0 = param(10);
q1 = param(11);
q2 = param(12);
q3 = param(13);
x = xin(:,1);
y = xin(:,2);
z = xin(:,3);
allp = [x(:),y(:),z(:)];
qnew = quaternion(q0,q1,q2,q3);
rotpoint = rotatepoint(qnew,allp(:,[1,2,3]));
finpoint = rotpoint(:,[1,2,3]) + [xc,yc,zc];
x = finpoint(:,1);
y = finpoint(:,2);
z = finpoint(:,3);

p = [x,y,z];

a = [a1 a2 a3 a4];
e = [e1 e2];

w_pi_apparent = atan(p(:,2)./p(:,1));
p_pi = [p(:,1:2), zeros(size(p,1),1)];

w_pi_bis = atan2(a(1).*p(:,2), a(2).*p(:,1));
w_s = atan2(sign(sin(w_pi_bis)).*abs(sin(w_pi_bis)).^(1/e(2)), ...
    sign(cos(w_pi_bis)).*abs(cos(w_pi_bis)).^(1/e(2)));

aws = sqrt(a1.^2.*(cos(w_s).^2).^e2+a2.^2.*(sin(w_s).^2).^e2);

R_bar = MeanRadius(a,e,w_s);
R_pi = R_bar.*p_pi./vecnorm(p_pi,2,2);

pSec = TransformLocal(a,e,w_s,aws,p(:,1),p(:,2),p(:,3));

eta_pi = atan2(aws.*pSec(:,2),(a(3).*pSec(:,1)));
eta_s = atan2(sign(sin(eta_pi)).*abs(sin(eta_pi)).^(1/e(1)), ...
    sign(cos(eta_pi)).*abs(cos(eta_pi)).^(1/e(1)));

p_s = [a(1).*(a(4) + sign(cos(eta_s)).*abs(cos(eta_s)).^e(1)).*...
    (sign(cos(w_s)).*abs(cos(w_s)).^e(2)), ...
    a(2).*(a(4) + sign(cos(eta_s)).*abs(cos(eta_s)).^e(1)).*...
    (sign(sin(w_s)).*abs(sin(w_s)).^e(2)), ...
    a(3).*sign(sin(eta_s)).*abs(sin(eta_s)).^e(1)];

d = p - p_s;
dist = vecnorm(d,2,2)./size(p,1).*5e3;
dist = [dist.*sqrt(n_w(1)); 1/a4.*sqrt(n_w(2)); a1.*sqrt(n_w(3)); ...
    a2.*sqrt(n_w(4)); a3.*sqrt(n_w(5))];

p_e = p_s - R_pi;
p_R = p - R_pi;

beta1 = vecnorm(p_e,2,2)./vecnorm(p_R,2,2);
beta2 = vecnorm(R_pi,2,2)./vecnorm(p_pi,2,2);
dbis = (1-beta1).*(p - beta2.*p_pi);
p_sbis = beta1.*p + beta2.*(1-beta1).*p_pi;

end

function R = MeanRadius(a,e,w)

R = a(4)*(a(1).^2*(cos(w).^2).^e(2) + a(2).^2*(sin(w).^2).^e(2)).^(1/2);

end

function pointprime = TransformLocal(a, e, ws, aws, x, y, z)

xprime = a(1).*sign(cos(ws)).*abs(cos(ws)).^e(2).*...
    (x - a(1).*a(4).*sign(cos(ws)).*abs(cos(ws)).^e(2))./aws + ...
    a(2).*sign(sin(ws)).*abs(sin(ws)).^e(2).*...
    (y-a(2).*a(4).*sign(sin(ws)).*abs(sin(ws)).^e(2))./aws;

zprime = z;
pointprime = [xprime, zprime];
end

function dw = weighDistanceIn(d,k,w)
% This funcion penalizes having points inside more than points outside

dw = (tanh(-k.*d)+1)./2*(w(1)-w(2)) + w(2);

end

function dw = weighDistanceOut(d,k,w)
% This funcion penalizes having points outside more than points inside

dw = (tanh(k.*d)+1)./2*(w(1)-w(2)) + w(2);

end

function [c,ceq] = nonlcon(x)

c = [];
ceq = x(10)^2 + x(11)^2 + x(12)^2 + x(13)^2 - 1;

end

function stop = hist(x, optimValues, state)
global resultsLog
resultsLog = [resultsLog, x];
stop = false;
end

function plotResults_3D(points, x)

scatter3(points(:,1), points(:,2), points(:,3))
amp = norm((max(points)-min(points))/2);

hold on
axis equal

phi = x(1);
theta = x(2);
psi = x(3);
xc = x(4);
yc = x(5);

v = [...
    cos(theta) 0 sin(theta);
    0 1 0;
    -sin(theta) 0 cos(theta)]*[...
    1 0 0;
    0 cos(phi) -sin(phi);
    0 sin(phi) cos(phi)]*[0;0;1].*(2*amp);

c = [...
    cos(theta) 0 sin(theta);
    0 1 0;
    -sin(theta) 0 cos(theta)]*[...
    1 0 0;
    0 cos(phi) -sin(phi);
    0 sin(phi) cos(phi)]*inv([...
    cos(psi) -sin(psi) 0;
    sin(psi) cos(psi) 0;
    0 0 1])*[-xc;-yc;0];

quiver3(c(1), c(2), c(3), v(1), v(2), v(3))

end

function [fx, Qs, cost_functionals] = plotResults(points, x, w)

phi = x(1);
theta = x(2);
psi = x(3);
xc = x(4);
yc = x(5);
re = x(6);
si = x(7);
so = x(8);
eps = x(9);

v = [...
    cos(theta) 0 sin(theta);
    0 1 0;
    -sin(theta) 0 cos(theta)]*[...
    1 0 0;
    0 cos(phi) -sin(phi);
    0 sin(phi) cos(phi)]*[0;0;1];

t = (-v(1).*(points(:,1)) - v(2).*(points(:,2)) - v(3).*(points(:,3)))/...
    (v(1)^2 + v(2)^2 + v(3)^2);

pointsproj = points + t.*[v(1) v(2) v(3)];

q2d = eul2quat([0 theta phi]);
q2dq = quaternion([q2d(1), -q2d(2), -q2d(3), -q2d(4)]);
pointsproj_rot = rotatepoint(q2dq, pointsproj);

qpsi = eul2quat([psi 0 0]);
qpsiq = quaternion(qpsi);

pointsproj_rot2 = rotatepoint(qpsiq, ...
    [pointsproj_rot(:,1), pointsproj_rot(:,2), zeros(size(pointsproj_rot,1),1)]);

pointsproj_rot2d = [pointsproj_rot2(:,1)+xc, pointsproj_rot2(:,2)+yc];

%% Distance to outer superellipse

eta_pi = atan2(so.*pointsproj_rot2d(:,2), (re*so).*pointsproj_rot2d(:,1));

Re_pi = [sign(cos(eta_pi)).*abs(cos(eta_pi)).^(1/eps), ...
    sign(sin(eta_pi)).*abs(sin(eta_pi)).^(1/eps)];

eta = atan2(Re_pi(:,2), Re_pi(:,1));

p_so = [so.*sign(cos(eta)).*abs(cos(eta)).^(eps), ...
    (so*re).*sign(sin(eta)).*abs(sin(eta)).^(eps)];

scatter(p_so(:,1), p_so(:,2))

%% Distance to inner superellipse

p_si = [si.*sign(cos(eta)).*abs(cos(eta)).^(eps), ...
    (si*re).*sign(sin(eta)).*abs(sin(eta)).^(eps)];

scatter(p_si(:,1), p_si(:,2))

scatter(pointsproj_rot2d(:,1), pointsproj_rot2d(:,2))

hold on
axis equal

SEi = PlotSuperellipse(re, si, eps, 500);
SEo = PlotSuperellipse(re, so, eps, 500);

plot(SEi(1,:), SEi(2,:), 'red')
plot(SEo(1,:), SEo(2,:), 'red')

d = vecnorm(pointsproj_rot2d,2,2);
do = vecnorm(p_so,2,2);
do_cl = d-do;
dow = weighDistanceOut(do_cl,1000,[w(1) w(2)]);
dowsq = dow.*do_cl;

di = vecnorm(p_si,2,2);
di_cl = d-di;
diw = weighDistanceIn(di_cl,1000,[w(3) w(4)]);
diwsq = diw.*di_cl;

dist = [diwsq; dowsq]./length(diwsq)./sqrt(2.5e-7);

% Add contribution of log imbalance and its weights (w)

k = 1000;

isEast = (tanh(k*pointsproj_rot2d(:,1))+1)/2;
isWest = 1-isEast;

isNorth = (tanh(k*pointsproj_rot2d(:,2))+1)/2;
isSouth = 1-isNorth;

% Find which quarter the points are in

pQ1 = isNorth.*isEast; % Northeast
pQ2 = isNorth.*isWest; % Northwest
pQ3 = isSouth.*isWest; % Southwest
pQ4 = isSouth.*isEast; % Southeast

pT = size(pointsproj_rot2d,1);

isNE = (tanh(k*(pointsproj_rot2d(:,1) + pointsproj_rot2d(:,2)))+1)/2;
isSW = 1 - isNE;

isNW = (tanh(k*(pointsproj_rot2d(:,1) - pointsproj_rot2d(:,2)))+1)/2;
isSE = 1 - isNW;

% Find which quarter the points are in (E - N - W - S)

pQE = isNE.*isSE;
pQN = isNE.*isNW;
pQW = isNW.*isSW;
pQS = isSW.*isSE;

Qs = [sum(pQ1); sum(pQ2); sum(pQ3); sum(pQ4); sum(pQE); sum(pQN); sum(pQW);
    sum(pQS); pT];

pQ8 = [sum(pQ1); sum(pQ2); sum(pQ3); sum(pQ4); sum(pQE); sum(pQN); sum(pQW); sum(pQS)];

pQ8_processed = (1./(log10(pQ8.*4)/log10(pT) + 0.01) - 1)./sqrt(10.95);

dist = [dist; pQ8_processed.*sqrt(w(5)); so/si*sqrt(w(6))];

cost_functionals = [sum(dist(1:2*pT).^2), sum(dist(2*pT+1:end-1).^2), ...
    sum(dist(end).^2)];
% 2*num_p1+1:2*num_p1+8
fx = sum(dist.^2);

end

function plotResults_3DPaper(points, x)

scatter3(points(:,1), points(:,2), points(:,3), '.', 'MarkerEdgeColor', ...
    [12/255 123/255 220/255])
amp = norm((max(points)-min(points))/2);

hold on
axis equal

phi = x(1);
theta = x(2);
psi = x(3);
xc = x(4);
yc = x(5);

v = [...
    cos(theta) 0 sin(theta);
    0 1 0;
    -sin(theta) 0 cos(theta)]*[...
    1 0 0;
    0 cos(phi) -sin(phi);
    0 sin(phi) cos(phi)]*[0;0;1].*(2*amp);

c = [...
    cos(theta) 0 sin(theta);
    0 1 0;
    -sin(theta) 0 cos(theta)]*[...
    1 0 0;
    0 cos(phi) -sin(phi);
    0 sin(phi) cos(phi)]*inv([...
    cos(psi) -sin(psi) 0;
    sin(psi) cos(psi) 0;
    0 0 1])*[-xc;-yc;0];

quiver3(c(1), c(2), c(3), v(1), v(2), v(3), 'AutoScale','off','Color', ...
    [255/255 194/255 10/255],'linewidth',1)

ti = -2.5:0.25:2.5;
til = {'','','','','','','','','','','','','','','','','','','','',''};

xticks(ti)
xticklabels(til)
yticks(ti)
yticklabels(til)
zticks(ti)
zticklabels(til)

end


function [fx, Qs, cost_functionals] = plotResultsPaper(points, x, w)

phi = x(1);
theta = x(2);
psi = x(3);
xc = x(4);
yc = x(5);
re = x(6);
si = x(7);
so = x(8);
eps = x(9);

v = [...
    cos(theta) 0 sin(theta);
    0 1 0;
    -sin(theta) 0 cos(theta)]*[...
    1 0 0;
    0 cos(phi) -sin(phi);
    0 sin(phi) cos(phi)]*[0;0;1];

t = (-v(1).*(points(:,1)) - v(2).*(points(:,2)) - v(3).*(points(:,3)))/...
    (v(1)^2 + v(2)^2 + v(3)^2);

pointsproj = points + t.*[v(1) v(2) v(3)];

q2d = eul2quat([0 theta phi]);
q2dq = quaternion([q2d(1), -q2d(2), -q2d(3), -q2d(4)]);
pointsproj_rot = rotatepoint(q2dq, pointsproj);

qpsi = eul2quat([psi 0 0]);
qpsiq = quaternion(qpsi);

pointsproj_rot2 = rotatepoint(qpsiq, ...
    [pointsproj_rot(:,1), pointsproj_rot(:,2), zeros(size(pointsproj_rot,1),1)]);

pointsproj_rot2d = [pointsproj_rot2(:,1)+xc, pointsproj_rot2(:,2)+yc];

%% Distance to outer superellipse

eta_pi = atan2(so.*pointsproj_rot2d(:,2), (re*so).*pointsproj_rot2d(:,1));

Re_pi = [sign(cos(eta_pi)).*abs(cos(eta_pi)).^(1/eps), ...
    sign(sin(eta_pi)).*abs(sin(eta_pi)).^(1/eps)];

eta = atan2(Re_pi(:,2), Re_pi(:,1));

p_so = [so.*sign(cos(eta)).*abs(cos(eta)).^(eps), ...
    (so*re).*sign(sin(eta)).*abs(sin(eta)).^(eps)];

scatter(p_so(:,1), p_so(:,2))

%% Distance to inner superellipse

p_si = [si.*sign(cos(eta)).*abs(cos(eta)).^(eps), ...
    (si*re).*sign(sin(eta)).*abs(sin(eta)).^(eps)];

scatter(p_si(:,1), p_si(:,2))

scatter(pointsproj_rot2d(:,1), pointsproj_rot2d(:,2), '.', 'Color', ...
    [12/255 123/255 220/255])

hold on
axis equal

SEi = PlotSuperellipse(re, si, eps, 500);
SEo = PlotSuperellipse(re, so, eps, 500);

plot(SEi(1,:), SEi(2,:), 'Color', [255/255 194/255 10/255])
plot(SEo(1,:), SEo(2,:), 'Color', [255/255 194/255 10/255])

d = vecnorm(pointsproj_rot2d,2,2);
do = vecnorm(p_so,2,2);
do_cl = d-do;
dow = weighDistanceOut(do_cl,1000,[w(1) w(2)]);
dowsq = dow.*do_cl;

di = vecnorm(p_si,2,2);
di_cl = d-di;
diw = weighDistanceIn(di_cl,1000,[w(3) w(4)]);
diwsq = diw.*di_cl;

dist = [diwsq; dowsq]./length(diwsq)./sqrt(2.5e-7);

% Add contribution of log imbalance and its weights (w)

k = 1000;

isEast = (tanh(k*pointsproj_rot2d(:,1))+1)/2;
isWest = 1-isEast;

isNorth = (tanh(k*pointsproj_rot2d(:,2))+1)/2;
isSouth = 1-isNorth;

% Find which quarter the points are in

pQ1 = isNorth.*isEast; % Northeast
pQ2 = isNorth.*isWest; % Northwest
pQ3 = isSouth.*isWest; % Southwest
pQ4 = isSouth.*isEast; % Southeast

pT = size(pointsproj_rot2d,1);

isNE = (tanh(k*(pointsproj_rot2d(:,1) + pointsproj_rot2d(:,2)))+1)/2;
isSW = 1 - isNE;

isNW = (tanh(k*(pointsproj_rot2d(:,1) - pointsproj_rot2d(:,2)))+1)/2;
isSE = 1 - isNW;

% Find which quarter the points are in (E - N - W - S)

pQE = isNE.*isSE;
pQN = isNE.*isNW;
pQW = isNW.*isSW;
pQS = isSW.*isSE;

Qs = [sum(pQ1); sum(pQ2); sum(pQ3); sum(pQ4); sum(pQE); sum(pQN); sum(pQW);
    sum(pQS); pT];

pQ8 = [sum(pQ1); sum(pQ2); sum(pQ3); sum(pQ4); sum(pQE); sum(pQN); sum(pQW); sum(pQS)];

pQ8_processed = (1./(log10(pQ8.*4)/log10(pT) + 0.01) - 1)./sqrt(10.95);

dist = [dist; pQ8_processed.*w(5); so/si*w(6)];

cost_functionals = [sum(dist(1:2*pT).^2), sum(dist(2*pT+1:end-1).^2), ...
    sum(dist(end).^2)];
% 2*num_p1+1:2*num_p1+8
fx = sum(dist.^2);

end

function points = PlotSuperellipse(re,s,eps,n)

eta_pi = linspace(-pi, pi, n);

Re_pi = [sign(cos(eta_pi)).*abs(cos(eta_pi)).^(1/eps); ...
    sign(sin(eta_pi)).*abs(sin(eta_pi)).^(1/eps)];

eta = atan2(Re_pi(2,:), Re_pi(1,:));

points= [s.*sign(cos(eta)).*abs(cos(eta)).^(eps);
    (s*re).*sign(sin(eta)).*abs(sin(eta)).^(eps)];

end

function points = PlotSupertoroid(a,e,n)

omega_pi = linspace(-pi,pi,n);
eta_pi = linspace(-pi,pi,n);

Ro_pi = [sign(cos(omega_pi)).*abs(cos(omega_pi)).^(1/e(2)); ...
    sign(sin(omega_pi)).*abs(sin(omega_pi)).^(1/e(2))];
Re_pi = [sign(cos(eta_pi)).*abs(cos(eta_pi)).^(1/e(1)); ...
    sign(sin(eta_pi)).*abs(sin(eta_pi)).^(1/e(1))];

omega = atan2(Ro_pi(2,:), Ro_pi(1,:));
eta = atan2(Re_pi(2,:), Re_pi(1,:));

points = zeros(n^2,3);

for i = 1:n
    for j = 1:n
        points((n*(i-1))+j,:) = [...
            a(1)*(a(4) + sign(cos(eta(j)))*abs(cos(eta(j)))^e(1))*...
            (sign(cos(omega(i)))*abs(cos(omega(i)))^e(2)), ...
            a(2)*(a(4) + sign(cos(eta(j)))*abs(cos(eta(j)))^e(1))*...
            (sign(sin(omega(i)))*abs(sin(omega(i)))^e(2)), ...
            a(3)*sign(sin(eta(j)))*abs(sin(eta(j)))^e(1)];
    end
end

end