function [dsig]=dsig(x)
l=length(x);
dsig=zeros(l,1);
for i=1:l

dsig(i,1)=sig(x(i))*(1-sig(x(i)));
end