function grad = tanhgradient( x )
%tahngradient compute the gradient of tanh(x)
%   Detailed explanation goes here

grad = (1 - tanh(x).*tanh(x));

end

