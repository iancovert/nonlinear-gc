%read in all the datas

FileNames = {'13_29',  '13_30',  '13_31',  '14_06',  '14_14',  '14_20'};
[D, Names] = readDataMatrixFromAMC('amc/13_29.amc');
ChannelNames = {
    'root.tx', 'root.ty','root.tz', 'root.rx', 'root.ry', 'root.rz', 'lowerback.rx', ...
    'lowerback.ry', 'lowerback.rz', 'upperback.rx', 'upperback.ry', 'upperback.rz', ...
    'thorax.rx', 'thorax.ry', 'thorax.rz', 'lowerneck.rx', 'lowerneck.ry', ...
    'lowerneck.rz', 'upperneck.rx', 'upperneck.ry', 'upperneck.rz', 'head.rx', 'head.ry', ...
    'head.rz', 'rclavicle.ry', 'rclavicle.rz', 'rhumerus.rx', 'rhumerus.ry', ...
    'rhumerus.rz', 'rradius.rx', 'rwrist.ry', 'rhand.rx', 'rhand.rz', ...
    'lclavicle.ry', 'lclavicle.rz', 'lhumerus.rx', ...
    'lhumerus.ry', 'lhumerus.rz', 'lradius.rx', 'lwrist.ry', 'lhand.rx', 'lhand.rz', ...
    'rfemur.rx', 'rfemur.ry', 'rfemur.rz', ...
    'rtibia.rx', 'rfoot.rx', 'rfoot.rz', 'rtoes.rx', 'lfemur.rx', 'lfemur.ry', ...
    'lfemur.rz', 'ltibia.rx', 'lfoot.rx', 'lfoot.rz', 'ltoes.rx'};


for n = 1:length(FileNames)
   amcfpath = ['amc/' FileNames{n} '.amc'];
   [X(n).dat, ColNames] = readDataMatrixFromAMC(amcfpath, ChannelNames);
   X(n).ColNames = ColNames;
   X(n).fn = FileNames{n};
end
save('mocapdata_.mat','X')