% Seth Thor
% Demonstrate the random walk algorithm with 1000 simulations

function stochwalk_NG()
npts = 1000;
nplot = 100;
seed = 1;
all_xplots = [];
for i = 1:nplot
   xplot = [];
   x = 0;
   for j=1:npts
      seed = rem(8121 * seed + 28411, 134456);
      if (seed > (134456/2))
         x = x + 1;
      else
         x = x - 1;
      end
      xplot = [xplot x];
   end
   all_xplots(i,:) = xplot;
   plot(xplot,'g')
   hold on
end
t = 1:(npts/10):npts;
error = sqrt(t)*1.5;
errorbar(t,zeros(size(t)),error,error,'.')
xlabel('t')
ylabel('x')
title('100 random walk paths')
axis([0 npts -2*sqrt(npts) 2*sqrt(npts)])
hold off
print -deps plot.eps

m_N = mean(all_xplots(:,npts))
ms_N = mean(all_xplots(:,npts).^2)
ms_N_pred = npts
D = ms_N/npts

ms_t = [];
for time = 1:1:npts
    ms_t(time) = mean(all_xplots(:,time).^2);
end
f = figure();
plot(ms_t)
xlabel('t')
ylabel('ms(t)')

end

