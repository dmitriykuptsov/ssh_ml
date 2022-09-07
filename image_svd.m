[I, map] = imread ("nature.bmp");
J = rgb2gray(I);


function [Uc, Sc, Vc] = compress_matrix(A, N)
    [U, S, V] = svd(A);
    Uc = U(:, 1:N);
    Sc = S(1:N, 1:N);
    Vc = V(:, 1:N);
end

[Uc, Sigmac, Vc] = compress_matrix(J, 20);
Jc20 = uint8(Uc * Sigmac * Vc');

[Uc, Sigmac, Vc] = compress_matrix(J, 50);
Jc50 = uint8(Uc * Sigmac * Vc');

[Uc, Sigmac, Vc] = compress_matrix(J, 100);
Jc100 = uint8(Uc * Sigmac * Vc');

figure
subplot(2,2,1)
xlabel("original")
imshow(J)
subplot(2,2,2)
xlabel("Compressed (first 20 components)")
imshow(Jc20)
subplot(2,2,3)
xlabel("Compressed (first 50 components)")
imshow(Jc50)
subplot(2,2,4)
xlabel("Compressed (first 100 components)")
imshow(Jc100);
