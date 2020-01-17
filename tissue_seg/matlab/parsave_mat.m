function [] = parsave_mat(saveName, mat) %#ok<INUSD>

save(saveName,'mat', '-v6');

end