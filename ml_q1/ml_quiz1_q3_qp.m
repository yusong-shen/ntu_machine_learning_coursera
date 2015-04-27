%% Quadratic Programming to sovle SVM
N = 7;
y = [-1, -1, -1, 1, 1, 1, 1]'; % y: 7x1
f = -ones(N,1);
Aeq = y';
beq = [0];
lb = zeros(N,1);
X = [1 0; 0 1; 0 -1; -1 0; 0 2; 0 -2; 2 0]';
% z_i = [1, 2^0.5*x1, 2^0.5*x2, x1^2, x2^2]
z = [1 2^0.5 0 1 0; 
    1 0 2^0.5 0 1; 
    1 0 -2^0.5 0 1;
    1 -2^0.5 0 1 0;
    1 0 2*2^0.5 0 4;
    1 0 -2*2^0.5 0 4;
    1 -2*2^0.5 0 4 0]; % z: 7x5
H = (y*y').*(z*z');
H = H + eye(N, N) * 0.0000001;
options = optimoptions('quadprog','Algorithm','interior-point-convex');
alpha = quadprog(H,f,[], [], Aeq, beq, lb, [], [], options);
% alpha = quadprog(H, -ones(N, 1), -eye(N), zeros(N, 1), y, 0);





