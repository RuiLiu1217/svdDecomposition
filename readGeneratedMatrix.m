fid = fopen('MatrixA10000x10000.matrix');
A = fread(fid,10000*10000,'float');
A = reshape(A, 10000, 10000);
A = A';
