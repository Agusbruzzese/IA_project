%% PRESENTACION.m
% =========================================================================
% Corre este script UNA VEZ antes de la presentacion.
% Hace todo: carga el modelo, procesa el archivo, prepara Simulink.
%
% USO:
%   1. Abre MATLAB
%   2. En Command Window escribe:  run('/home/augustin/PRESENTACION')
%   3. Espera ~2 minutos
%   4. Todo listo para mostrar
% =========================================================================

clc; clear;
fprintf('==============================================\n');
fprintf('  PREPARANDO PRESENTACION...\n');
fprintf('==============================================\n\n');

cd('/home/augustin');

%% ── 1. CARGAR MODELO ONNX ────────────────────────────────────────────────
fprintf('[1/4] Cargando modelo ONNX...\n');
ONNX_PATH = "/home/augustin/Documents/IA_project/final_versionV1/simulink_visualizer/transformer_v4_single.onnx";

net = importNetworkFromONNX(ONNX_PATH);
x_d = dlarray(single(randn(16,2400,1,1)),'SCBT');
t_d = dlarray(single(randn(6,1)),'CB');
net = initialize(net, x_d, t_d);
fprintf('    OK — modelo cargado (803k parametros)\n\n');

%% ── 2. PROCESAR ARCHIVO DE TEST ──────────────────────────────────────────
fprintf('[2/4] Procesando archivo de test...\n');

Fs=2400; fc=24.9; bw=5.0;
edge_trim=round(0.5*Fs); detrend_k=12001; iqr_clip=0.5;
SEQ_LEN=2400; STEP_SIZE=240; CONTEXT_W=16; THRESH=0.6;
[bp_b, bp_a] = butter(4, [(fc-bw)/(Fs/2), (fc+bw)/(Fs/2)], 'bandpass');

TEST_DIR = "/home/augustin/Documents/IA_project/Spintronic Sensor Dataset for Vehicle Detection and Car Model Recognition (14 Days, Unlabeled with Timestamps)/Run 20250224 14h55m20s - 20250227 14h38m27s";
files     = dir(fullfile(TEST_DIR,'**','*.bsd'));
bin_files = files(contains({files.name},'BIN Time'));

bsd_path = fullfile(bin_files(1).folder, bin_files(1).name);
fid = fopen(bsd_path,'rb'); raw = fread(fid, 165888, 'int16'); fclose(fid);

% Demodulate
bp  = filtfilt(bp_b, bp_a, double(raw));
env = single(hilbert_envelope(bp));
env = env(edge_trim+1 : end-edge_trim);

chunk=24000; baseline=zeros(length(env),1,'single');
for ci=1:chunk:length(env)
    i1=max(1,ci-detrend_k); i2=min(length(env),ci+chunk+detrend_k);
    b=movmedian(double(env(i1:i2)),detrend_k);
    o1=ci-i1+1; o2=min(o1+chunk-1,length(b));
    baseline(ci:ci+o2-o1)=single(b(o1:o2));
end
det=env-baseline;

n=length(det); half=floor(detrend_k/2);
anchors=(1:2400:n)'; vals=zeros(length(anchors),1,'single');
for ai=1:length(anchors)
    c=anchors(ai); seg=sort(double(det(max(1,c-half):min(n,c+half))));
    ns=length(seg);
    vals(ai)=max(single(seg(max(1,round(0.75*ns)))-seg(max(1,round(0.25*ns)))),single(iqr_clip));
end
scale=single(interp1(double(anchors),double(vals),(1:n)','linear','extrap'));
normed=det./scale;
fprintf('    OK — señal demodulada: [%.2f, %.2f] sigma\n\n', min(normed), max(normed));

%% ── 3. CORRER INFERENCIA ─────────────────────────────────────────────────
fprintf('[3/4] Corriendo inferencia...\n');
n_wins=floor((length(normed)-SEQ_LEN)/STEP_SIZE)+1;
probs=zeros(n_wins,1,'single');
buf=zeros(CONTEXT_W,SEQ_LEN,'single');

for i=1:n_wins
    buf=circshift(buf,-1,1);
    buf(end,:)=normed((i-1)*STEP_SIZE+1:(i-1)*STEP_SIZE+SEQ_LEN);
    if i<CONTEXT_W; continue; end
    x_in=dlarray(reshape(buf,[CONTEXT_W,SEQ_LEN,1,1]),'SCBT');
    t_in=dlarray(zeros(6,1,'single'),'CB');
    p=predict(net,x_in,t_in);
    probs(i)=extractdata(p);
end
n_det=sum(probs>THRESH);
fprintf('    OK — %d detecciones en %d ventanas\n\n', n_det, n_wins);

%% ── 4. PREPARAR VARIABLE PARA SIMULINK ───────────────────────────────────
fprintf('[4/4] Preparando workspace para Simulink...\n');
t_vec = (0:length(normed)-1)'/Fs;
simin.time              = t_vec;
simin.signals.values    = double(normed);
simin.signals.dimensions = 1;
fprintf('    OK — simin listo (%d samples, %.1f segundos)\n\n', ...
        length(t_vec), t_vec(end));

%% ── PLOT DE RESULTADOS ───────────────────────────────────────────────────
t_sig = (0:length(normed)-1)/Fs;
t_win = ((0:n_wins-1)*STEP_SIZE + SEQ_LEN/2)/Fs;

figure('Color',[0.04 0.04 0.08], 'Position',[100 100 1300 550], ...
       'Name','Vehicle Detection — INL Gate Sensor');

subplot(2,1,1);
plot(t_sig, normed,'Color',[0.23 0.62 0.85],'LineWidth',0.5);
hold on;
yline(1.5,'--','Color',[1 0.8 0.2],'LineWidth',1.2,'Label','1.5\sigma');
ylabel('Signal (\sigma)','Color','w','FontSize',11);
title('AMR Signal — Demodulated','Color','w','FontSize',12);
set(gca,'Color',[0.05 0.05 0.1],'XColor','w','YColor','w');
xlim([0 t_sig(end)]);

subplot(2,1,2);
area(t_win, probs .* (probs>THRESH), ...
     'FaceColor',[0 1 0.53],'FaceAlpha',0.6,'EdgeColor','none');
hold on;
area(t_win, probs .* (probs<=THRESH), ...
     'FaceColor',[0.23 0.62 0.85],'FaceAlpha',0.3,'EdgeColor','none');
yline(THRESH,'--','Color',[1 0.4 0.2],'LineWidth',1.5,'Label','Threshold 0.6');
ylabel('P(vehicle)','Color','w','FontSize',11);
xlabel('Time (s)','Color','w','FontSize',11);
ylim([0 1.1]); xlim([0 t_sig(end)]);
title(sprintf('Model Output P(vehicle) — %d detections', n_det),'Color','w','FontSize',12);
set(gca,'Color',[0.05 0.05 0.1],'XColor','w','YColor','w');

sgtitle('INL Gate Sensor — TemporalTransformer v4 — ONNX Inference', ...
        'Color','w','FontSize',13,'FontWeight','bold');

%% ── LISTO ────────────────────────────────────────────────────────────────
fprintf('==============================================\n');
fprintf('  TODO LISTO PARA LA PRESENTACION\n');
fprintf('==============================================\n');
fprintf('\n');
fprintf('  MATLAB plot  -> ya visible\n');
fprintf('  Simulink     -> escribe:  open vehicle_detection_simulink\n');
fprintf('                  luego click Run (el simin ya esta cargado)\n');
fprintf('\n');
