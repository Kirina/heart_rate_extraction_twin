
dataFolder='C:\Users\kirin\Documents\Internship\output\test_predictions'; %fill in the pathname where data is stored
saveFolder='C:\Users\kirin\Documents\Internship\output\predictions';

fileList=dir(dataFolder);
    
%for iFile=1:length(fileList)
% kan ik op een andere manier door de data heen gaan? 
% hr_1_list = ['006', '080', '319', '121', '440', '120', '316', '384', '089', '266']
% hr_2_list = ['263', '040', '386', '320', '086', '392', '338', '258', '237', '096']
for iFile=1:length(fileList)
    if ~isempty(strfind(lower(fileList(iFile).name),'.signal'))
        
        clearvars -except dataFolder fileList iFile saveFolder
        clearvars -GLOBAL
               
        
        fprintf('%i of %i\n',iFile,length(fileList));
        
        fECG=h5read([dataFolder,filesep,fileList(iFile).name],'/hr_2_extracted');
        [~,filename,~]=fileparts(fileList(iFile).name);
        fECG=double(fECG);
        
        saveName=[saveFolder,filesep,filename,'_peaks.mat'];
        
        if exist(saveName,'file')
            continue;
        end
        
        
        %WHEN LOADING DATA FROM THE NEURAL NETWORK, SAVE THEM EITHER AS HDF5 FILES
        %OR AS MAT FILE (E.G. WITH SCIPY.IO.SAVEMAT). IN CASE OF HDF5 UNCOMMENT THE
        %TWO LINES BELOW AND COMMENT THE LINES BELOW THAT. YOU CAN ALSO USE THE UI
        %TO LOAD DATA (uigetfile) AS SPECIFICIED BELOW FOR LOADING .MAT FILES
        
        % [filename,pathname]=uigetfile('Select data file','hdf5');
        % fECG=h5read([pathname,filename],'/data');
        % dataFile='testData.mat';
        % load(dataFile);
        
        warning('off');
        
        if size(fECG,1)>size(fECG,2)
            fECG=fECG';
        end
        
        fECG_rs=zeros(size(fECG,1),2*size(fECG,2));
        for ch=1:size(fECG,1)
            fECG_rs(ch,:)=resample(fECG(ch,:),1000,500);
        end
        
       %fECG(:,1:60*1000)=[];
       % length of fECG
       %fECG=fECG(:,1:min([60*1000*5,length(fECG)]));
       %fECG = fECG(:,1:60*1000);
        fECG=fECG_rs;
        
        %start processing from here
        %clc
        %    clear all
        close all;
        
        
        % [filename,pathname]=uigetfile('Select data file','hdf5');
        % fetalSignal=h5read([pathname,filename],'/data');
        
        
        global Fs nCh nVar order QRSwidth bCompEff
        global tsearch tQRS tnoise minRR maxRR RRmean
        global b0 w0 sigma0
        global sigmafactor I
        
        curPath  = cd;
        functionPath = [curPath,filesep,'functions'];
        %pathData ='C:\Users\gwarmerdam\Documents\phd\publications\Journal\fetal R-peak detection\scripts and data';
        cd(functionPath)
        
        selCh=1:size(fECG,1);
        s = fECG(selCh,:);
        s0=fECG(selCh,:);
        
        %I= Panno;
        
        % General parameters
        bCompEff=false;
        Fs  = 1000;
        nCh = size(s,1);
        nVar     = 3*nCh+1; % 3 amplitudes per channel + 1 central frequency
        QRSwidth = 40*1e-3*Fs;
        order    = 6;
        cutoffHP = 3;
        
        % time indices
        minBPM = 80; maxBPM = 240; % << based on refractory period. Simulations in physionet are crap
        minRR  =round(60*Fs/maxBPM);
        maxRR  =round(60*Fs/minBPM);
        tsearch = minRR+1:maxRR; % search window for new QRS complexes
        tQRS    = -QRSwidth/2+1:QRSwidth/2; % samples used for QRS-fit
        tnoise  = [-QRSwidth/2-0.1*Fs:-QRSwidth/2-20,...
            QRSwidth/2+20:QRSwidth/2+0.1*Fs];
        
        % Parameters for initialization
        F0 = 42; % central frequency for fetal QRS
        b0   = Fs*0.2251/F0;
        w0   = round(60*Fs/150); % meanRR, initial settings for r-peak detection
        sigma0 = round(0.25*w0);
        RRmean = round(60*Fs/150);
        sigmafactor= 1e-5;
        
        %//////////////////////////////////////////////////////////////////////////
        %% Pre-process baseline by HP filter
        % For now only use xECG
        [b_hp,a_hp]=butter(4,cutoffHP*2/Fs,'high');
        x=FecgICAf(s',Fs);
        x= x';
        for ch =1:nCh
            x(ch,:)  = filtfilt(b_hp,a_hp,x(ch,:));
        end
        
        
        %//////////////////////////////////////////////////////////////////////////
        %% Run algorithm
        
        % Estimated model parameters
        estNpeaks = floor(size(x,2)/(Fs/3)); % initial estimate of the number of peaks
        thetahat = NaN(order,estNpeaks);
        Vhat = NaN(order,order,estNpeaks);
        zhat = NaN(nVar,estNpeaks);
        Phat = NaN(nVar,nVar,estNpeaks);
        qhat = NaN(1,estNpeaks);
        Lambdahat = NaN(nCh,estNpeaks); % noise covariance
        evidence  = NaN(1,size(x,2)); % weighted convolution
        posterior = NaN(1,size(x,2)); % weighted convolution
        
        % output signals
        rPeaks    =[];% NaN(1,estNpeaks); % Estimated Rpeak locations
        allPeaks  =[];% NaN(1,estNpeaks); % Estimated Rpeak locations
        rPeaksb   =[];
        allPeaksb =[];
        rPeaksf   =[];
        allPeaksf =[];
        
        % Model parameters for each Rpeak
        allPDF_f.muhat    = [];
        allPDF_f.muSigma  = [];
        allPDF_f.zhat     = [];
        allPDF_f.Phat     = [];
        allPDF_f.Sigmahat = [];
        
        xhat = zeros(nCh,size(x,2)); % Estimated QRS complexes
        
        allBestCh =[];
        mu_k = 0;
        k = 0;
        bFoundIni=false;
        bBadAR=false;
        
        try
            while mu_k+tsearch(end)<size(x,2);
                %     if mu_k>7000
                %   if mu_k>2.9*1e4
                %       break
                %   end
                %% Initialization
                if bFoundIni==false
                    
                    %         if mu_k>4.7*1e4
                    %             break
                    %         end
                    
                    disp('initialize')
                    [bFoundIni,mu_k,wvecf,thetaf,Vf,Qf,R,wvecb,thetab,Vb,Qb,zf,Sigmaf,Pf,bestCh,bEoF] = initialization(x,mu_k,bBadAR);
                    if bFoundIni==false
                        % No new initial R-peaks were detected.
                        break
                    end
                    
                    if bEoF==true
                        allPeaks =[allPeaks mu_k]; %in this case mu_k contains all remaining detected r-Peaks.
                        rPeaks   =[rPeaks mu_k];
                        break
                    end
                    allBestCh(length(allBestCh)+1)= bestCh;
                    bBackwards=true; % after initialization run backwards prediction
                    mu_final = mu_k;
                    
                    
                    % Forward and backward QRS model are initially the same.
                    zb     = zf;
                    Pb     = Pf;
                    Sigmab = Sigmaf;
                end
                
                %% Backwards prediction
                % After initialization, first look backwards for missed peaks.
                % Stop whenever <order> R-peaks are the same in forwards and
                % backwards, or when no new backwards Rpeak is detected.
                if bBackwards==true
                    %     tmpidx = 0;
                    %        figure;plot(x(bestCh,:));hold on;plot(allPeaksf,x(bestCh,allPeaksf),'ro')
                    %        hold on;plot(mu_k,x(bestCh,mu_k),'go')
                    %        hold on;plot(detPeaksb,x(bestCh,detPeaksb),'go')
                    %        hold on;plot(intPeaksb,x(bestCh,intPeaksb),'mo')
                    %        hold on;plot(I,x(bestCh,I),'yo')
                    rPeaksb =[]; allPeaksb=[];
                    nSimilar=0;
                    while nSimilar<order
                        %            tmpidx = tmpidx+1;
                        %            if tmpidx==9
                        %                break
                        %            end
                        % Step 1: Backwards prediction. Note: combines both forward and backwards prediction
                        [intPeaksb,detPeaksb] = backwardsRpeakPrediction(x,mu_k,wvecb,thetab,Vb,R,zb,Pb,Sigmab,allPDF_f,bBadAR);
                        if isempty(detPeaksb)
                            allPeaksb = [intPeaksb allPeaksb];
                            rPeaksb   =[NaN(1,length(allPeaksb)) rPeaksb];
                            break
                        end
                        
                        % Update R-peaks. Note that in backwards prediction we use <mu_k> and <w_k> to
                        % update the model and not <mu_k_1> and <w_k_1> as is done in forwards prediction.
                        newPeaksb = sort([intPeaksb, detPeaksb]);
                        mu_k_1   = detPeaksb(2);
                        mu_k     = detPeaksb(1);
                        w_k      = mu_k_1-mu_k-RRmean;
                        if ~isempty(intPeaksb)
                            wvecb = [diff(newPeaksb(2:end))-RRmean wvecb]; % store interpolated peaks in wvec
                            wvecb = wvecb(1:order);
                        end
                        
                        % Step 2: update noise covariances
                        [Qb,R,Sigmab,lambda]= updateCovariances(x,w_k,mu_k,wvecb,thetab,Vb,Qb,R,Sigmab);
                        
                        % Step 3: Update model parameter (theta,z) for each particle
                        % RR model : theta, V, Q, R
                        % QRS model: z, P, Sigma, Lambda
                        [thetab,Vb,zb,Pb] = updateModel(x,w_k,mu_k,wvecb,thetab,Vb,Qb,R,zb,Pb,Sigmab,lambda);
                        wvecb = [w_k wvecb(1:end-1)]; % Shift wvec by one value.
                        
                        %             hold on;plot(mu_k,x(mu_k),'go')
                        % Step 4: store output
                        % Check if mu_k is different from the peaks found in the forward run.
                        if ~isempty(allPeaks) && min(abs(mu_k-allPeaks)) < 20
                            nSimilar=nSimilar+1;
                        else
                            nSimilar=0;
                        end
                        
                        newPeaksb(end)=[]; % final peak was starting point
                        allPeaksb =[newPeaksb allPeaksb];
                        tmpPeaksb = newPeaksb; tmpPeaksb(ismember(newPeaksb,intPeaksb)) = NaN; % only store QRS complexes
                        rPeaksb   =[tmpPeaksb rPeaksb];
                    end
                    
                    % Add backwards r-Peaks to total R-peak vector (both forward and
                    % backward). Replace previous run
                    if ~isempty(allPeaksb)
                        allPeaks(allPeaks>allPeaksb(1)-minRR)=[];
                        rPeaks(rPeaks>allPeaksb(1)-minRR)=[];
                        allPeaks =[allPeaks allPeaksb mu_final];
                        rPeaks =[rPeaks rPeaksb mu_final];
                    end
                    
                    % Continue forward load with mu_final
                    mu_k=mu_final;
                    bBackwards=false; % No new peaks were detected. stop backwards prediction
                    
                    
                    % Update wvecf using the peaks found from backwards prediction
                    if length(allPeaksb)>order-1
                        tmp  =diff([allPeaksb(end-5:end) mu_final])-RRmean;
                        if all(abs(tmp)<0.5*RRmean)
                            % no missed beats
                            wvecf = tmp(end:-1:1);
                        end
                    end
                end
                
                %        figure;
                %        plot(x(1,:));hold on;
                %        plot(xhat(1,:),'r')
                
                %         figure;plot(x(2,:));
                %     hold on;plot(I,x(bestCh,I),'yo')
                %     hold on;plot(detPeaksf,x(2,detPeaksf),'ro')
                %     hold on;plot(intPeaksf,x(ch,intPeaksf),'mo')
                %     hold on;plot(mu_k,x(bestCh,mu_k),'ro')
                %% Forwards prediction
                % Step 1: Locate new Rpeaks assuming model parameters are known
                [intPeaksf,detPeaksf,bBadAR,pdf_f] = forwardsRpeakPrediction(x,mu_k,wvecf,thetaf,Vf,R,zf,Pf,Sigmaf);
                if isempty(detPeaksf)
                    bFoundIni=false; % no new peak has been detected, return to initialization
                    continue;
                end
                
                
                % Update R-peaks. In case one or multiple peaks were interpolated, the RR-intervals
                % of the interpolated peaks are also stored in wvec. Note that the
                % newest RR-interval w(k+1) is not stored in wvec yet.
                newPeaksf = sort([intPeaksf, detPeaksf]);
                mu_k_1   = detPeaksf(end);
                mu_k     = detPeaksf(end-1); % newest R-peak
                w_k_1    = mu_k_1-mu_k - RRmean; % newest RR-interval
                if ~isempty(intPeaksf)
                    wvecf = [diff(newPeaksf(1:end-1))-RRmean wvecf];
                    wvecf = wvecf(1:order);
                end
                
                % Step 2: update noise covariances
                [Qf,R,Sigmaf,lambda]= updateCovariances(x,w_k_1,mu_k_1,wvecf,thetaf,Vf,Qf,R,Sigmaf);
                
                % Step 3: Update model parameter (theta,z) for each particle
                % RR model : theta, V, Q, R
                % QRS model: z, P, Sigma, Lambda
                [thetaf,Vf,zf,Pf] = updateModel(x,w_k_1,mu_k_1,wvecf,thetaf,Vf,Qf,R,zf,Pf,Sigmaf,lambda);
                wvecf = [w_k_1 wvecf(1:end-1)]; % Shift wvec by one value.
                
                
                % Step 4: Store output
                xhat(:,mu_k_1+tQRS) = reshape(MexHat(zf,tQRS),length(tQRS),nCh)';
                
                newPeaksf(1)=[]; % first peak was starting point in previous run
                allPeaksf(k+1:k+length(newPeaksf)) = newPeaksf; % allpeaks also contained interpolated peaks
                tmpPeaksf = newPeaksf; tmpPeaksf(ismember(newPeaksf,intPeaksf)) = NaN; % only store QRS complexes
                rPeaksf(k+1:k+length(newPeaksf)) = tmpPeaksf;
                
                % Add forward peaks to total R-peak vector
                allPeaks = [allPeaks newPeaksf]; % allpeaks also contained interpolated peaks
                rPeaks   = [rPeaks tmpPeaksf];
                
                
                % Note: model parameters are only updated for mu_k_1
                allPDF_f.muhat = [allPDF_f.muhat; pdf_f.muhat];
                allPDF_f.muSigma = [allPDF_f.muSigma; pdf_f.muSigma];
                allPDF_f.zhat = [allPDF_f.zhat; pdf_f.zhat];
                allPDF_f.Phat = [allPDF_f.Phat; pdf_f.Phat];
                allPDF_f.Sigmahat = [allPDF_f.Sigmahat; pdf_f.Sigmahat];
                
                
                % Model parameters
                thetahat(:,k+length(newPeaksf))  = thetaf;
                Vhat(:,:,k+length(newPeaksf))    = Vf;
                zhat(:,k+length(newPeaksf))      = zf;
                Phat(:,:,k+length(newPeaksf))    = Pf;
                Lambdahat(:,k+length(newPeaksf)) = lambda;
                
                % continue to next run
                mu_k = mu_k_1;
                k    =  k+length(newPeaksf);
            end
        catch
            cd(curPath);
            error(lasterr);
        end
        cd(curPath);
        
        allPeaks(allPeaks<0 | allPeaks>length(fECG))=[];
        %allPeaks=allPeaks+60*Fs;
        %allPeaks=floor(allPeaks/2);
        %Evaluate performance by plotting some examples
                %ch=3;
                ch=1;
                figure;
                plot(fECG(ch,:)); hold on;plot(allPeaks,fECG(ch,allPeaks),'or');
                legend('fetal ECG signal','detected peaks')
        
                %Determine FHR
                FHR(1,:)=allPeaks(2:end);
                FHR(2,:)=60*Fs./diff(allPeaks);
                figure;plot(FHR(1,:)/Fs,FHR(2,:));
                xlabel('Time (s)');
                ylabel('FHR (bpm)');
        
        
        save(saveName,'allPeaks','FHR');
    end
end
