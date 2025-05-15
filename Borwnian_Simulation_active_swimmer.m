%Active Brownian motion: homogeneous envirnonment
%Langevin simulation for the active Brownian particle
close, clear, clc
kB = 1.38e-23;%Boltzman constant J/K
T  = 300;%tempertaure K
eta = 0.001;%Pa.s
R = 1e-6;%size of the particle
D_T = kB*T./(6*pi*eta*R);%translational diffusion m^2/s
D_R = kB*T./(8*pi*eta*R^3);%rotational diffusion rad^2/s
X(1,:) = [0, 0];%X0;
theta = 0;
N = 1e3;%Number of steps
dt = 1e-3;%s
v = 1e-5;%m/s
W = 3.14;%rad/s
for n=1:1:N
    X(n+1,:) = X(n,:) + sqrt(2*D_R*dt)*randn(1,2) + v*cos(theta);
    theta = theta + sqrt(2*D_T*dt)*randn(1,1) + dt*W;
end
x = X(:,1); y = X(:,2);
plot(x,y)
xlabel('x'), ylabel('y'), title('x-y plot')

