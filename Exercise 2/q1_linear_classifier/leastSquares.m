function [weight, bias] = leastSquares(data, label)
	% Sum of squared error shoud be minimized
	%
	% INPUT:
	% data        : Training inputs  (num_samples x dim)
	% label       : Training targets (num_samples x 1)
	%
	% OUTPUT:
	% weights     : weights   (dim x 1)
	% bias        : bias term (scalar)
	%

	% insert your code here

    weight = inv(data'*data)*(data'*label);
    bias = 0.02;
end
