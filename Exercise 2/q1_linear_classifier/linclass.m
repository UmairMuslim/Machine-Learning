function class = linclass(weight, bias, data)
	% Linear Classifier
	%
	% INPUT:
	% weight      : weights                (dim x 1)
	% bias        : bias term              (scalar)
	% data        : Input to be classifiednos (num_samples x dim)
	%
	% OUTPUT:
	% class       : Predicted class (+-1) values  (num_samples x 1)

	% insert your code here

    [num_samples, ~] = size(data);
    class = zeros(num_samples,1);
    for index=1:num_samples
        y = weight'.*data(index,:) + bias;
        if(y(1) > y(2))
            class(index) = -1;
        else
            class(index) = 1;
        end
    end
end
