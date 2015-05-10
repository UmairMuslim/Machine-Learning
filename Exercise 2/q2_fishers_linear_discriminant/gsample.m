function X = gsample(mean, sigma, n)
	% Sample from a multinomial Gaussian distribution
	%
	% INPUT:
	% mean     : Mean vector of the Gaussian (dim x 1)
	% sigma    : Covariance matrix of the Gaussian (dim x dim)
	% n        : Number of points to be sampled (scalar)
	%
	% OUTPUT:
	% X        : Points sampled from the Gaussian (dim x n)

	dim = size(sigma, 1);
	X = zeros(dim, n);

	for i = 1:n
		X(:, i) = mean + (sigma * randn(dim, 1));
	end

end

