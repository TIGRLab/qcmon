"""Refactored code from qc-fbirn-fmri and associated scripts for use in datman.

#### Update docs and reorganize code.
"""
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

matplotlib.use('Agg')

# Should probably be 4 in python given 0 indexing
# If updated, remove all -1 instances
TRIM_TR = 5 - 1 # Formerly 'i1'


def analyze_fmri_phantom(input_nii, output_prefix):
    """Refactored code from the matlab folder.
    """
    # addpath(genpath('/home/desmith/Desktop/code/work/linked_nii/qcmon/assets/matlab'));

    # I4d = matrix of nifti data
    img_4d = nib.load(input_nii)
    img_4d = img_4d.get_fdata()
    # fid is open file handle for _stats.csv file
    # Write header to output file

    # Set some variables (numsl etc)

    img_4d, num_pix = center_volume(img_4d, [90, 90])

    if num_pix is None:
        num_pix = 128

    if num_pix == 128:
        roi_width = 30
    elif num_pix == 64:
        roi_width = 15
    else:
        roi_width = 20

    # More constants being used in the main loop
    npo2 = num_pix // 2
    ro2 = int(roi_width / 2)
    X1 = Y1 = (npo2 - ro2) - 1 # subtract one to make equal to matlab result
    X2 = Y2 = X1 + roi_width   # Dont subtract one since upper bound exclusive
    r1 = 1 # Maybe should be 0?
    r2 = roi_width # why have this in addition to roi_width (aka R)???

    # Make a square mask of ones of size roi_width x roi_width
    mask = np.ones((roi_width, roi_width))
    npx = sum(sum(mask))
    # img = square matrix of zeroes of size num_pix
    # mag == img

    # These may be deletable to reduce clutter (ugh)
    # i1 = 5 (trim first five TRs)
    # i2 = img_4d.shape[3] (i.e. size of last dimension)
    # N = size of last dim - 5 + 1 (num of TRs)
    # M = (calculated by other chunk of vars)
    N = img_4d.shape[3] - TRIM_TR   # Add one or no?
    M = roi_width

    # Are these deletable?
    roir = np.zeros((N, M))
    # fkern = output_prefix
    # numwin = 3 (??)
    size_4d = img_4d.shape[3] # formerly i2

    # This should maybe be 19 in python, given 0 index
    # If updated, remove all -1 instances
    num_slices = 20 # formerly slicenum = floor(numsl/2) (numsl is 40. why 40? wtf?)

    total_slices = 40 # formerly == numsl (which, again, is just 40)

    # Init the arrays for summing cells
    img = np.zeros(num_pix)
    mag = np.zeros(num_pix)
    roi = []  # Replacement for matlab roi(S0) syntax that just automatically accumulates in list

    Iodd, Ieven, Syy, Syt, St, Stt, S0, roi, roir = modify_matrix(
        img_4d, num_pix, npx, roi_width)

    # write out diff image
    Isub = Iodd - Ieven;
    # I think this is just to convert from a 4096,1 vector to a 64x64 matrix.
    # img(:) = Isub;
    img = Isub.reshape((num_pix, num_pix)).T  # Take transpose to avoid flipping rows and cols
    # And then this slices the roi values
    # sub = img(X1:X2,Y1:Y2);
    sub = img[X1:X2, Y1:Y2]  # Watch the upper bounds. sub should be 15x15
    # This takes the variance of the flattened sub (to get a single val instead of row of variance)
    # varI = var(sub(:));
    varI = np.var(sub.flatten(), ddof=1)  # Set degrees of freedom to 1 to mimic matlab

    # write out ave image
    Sy = Iodd + Ieven
    Iave = Sy / N
    img = Iave.reshape((num_pix, num_pix)).T
    sub = img[X1:X2, Y1:Y2]
    meanI = sub.flatten().mean()

    # find trend line at a + b
    D = (Stt*S0 - St*St);
    a = (Syt*S0 - St*Sy)/D;  # You might need a check if D == 0 to avoid exception
    b = (Stt*Sy - St*Syt)/D;

    # Make sd image
    Var = Syy + a * a * Stt + b * b * S0 + 2 * a * b * St - 2 * a * Syt - 2 * b * Sy
    Isd = np.sqrt(Var/(N-1));

    # make sfnr image
    sfnr = Iave / (Isd + np.finfo(float).eps)
    img = sfnr.reshape((num_pix, num_pix)).T
    sub = img[X1:X2+1, X1:X2+1]
    sfnrI = sub.flatten().mean()

    snr = meanI / np.sqrt(varI / N)
    # Here is where values are written / displayed

    make_figures(Iave, Isub, Isd, sfnr, num_pix, output_prefix)
    make_plots(N, roi, r2)


def center_volume(img_4d, end_shape=None):
    if not end_shape:
        end_shape = [90, 90]

    # Need to init an empty one with same size
    for idx, max_len in enumerate(end_shape):
        if img_4d.shape[idx] <= max_len:
            continue

        # Do cropping here (see rest of centre_volume)

    # Return fixed version here
    return img_4d, img_4d.shape[0]


def modify_matrix(img_4d, num_pix, npx, roi_width, num_slices=20,
                  total_slices=40):
    """This needs a better name and doc string.

    It's mostly just a place holder function to make it easier to test
    the code that comes after.
    """
    # Current values:
    # num_pix = 64
    # npx = 225.0
    # roi_width = 15

    size_4d = img_4d.shape[3]
    Iodd = np.zeros((num_pix * num_pix, 1))
    Ieven = np.zeros((num_pix * num_pix, 1))
    Syy = np.zeros((num_pix * num_pix, 1))
    Syt = np.zeros((num_pix * num_pix, 1))
    St = Stt = S0 = 0
    roi = []
    roir = np.zeros((size_4d - TRIM_TR, roi_width))
    X1 = (num_pix // 2) - int(roi_width / 2) - 1
    X2 = X1 + roi_width
    for idx in range(TRIM_TR, size_4d):
        # Ok, alg seems to be:
        #   Loop over TRs from 5 to end (inclusive)
        #   For each TR take the whole image (64x64 in this case) for slice 20 in 3rd dim
        #   Then, concatenate each row (all columns) into one long column

        # Squeeze may be useless here...
        A = np.squeeze(img_4d[:, :, num_slices - 1, idx])

        # This piece of code is unraveling all the pixels into one long matrix
        # of columns
        # tmp_I = []
        # for pixel in range(num_pix):
        #     tmp = A[pixel, :num_pix]
        #     tmp_I = [tmp_I, tmp]
        tmpI = A.flatten()

        # This line appears to convert cells to double precision and turn
        # the long row into one long column
        # I(1:NPIX*NPIX,1)=double(tmpI(1:NPIX*NPIX));
        I = tmpI[:, np.newaxis] # Option 1: 'promote' to a column
        # I = tmpI.reshape(-1, 1) # Option 2: reshape

        # Sum the cells by odd and even frames
        # Must modify idx to figure out if frame is odd or even
        # Actually... adding one here might screw up the result, given that
        # Syy etc. are multiplying by the index
        if (idx + 1) % 2 == 0:
            Ieven = Ieven + I
        else:
            Iodd = Iodd + I

        # Add one to index here to replicate what happened in matlab
        Syt = Syt + I * (idx + 1)
        Syy = Syy + I * I
        S0 += 1
        St += (idx + 1)
        Stt = Stt + (idx + 1) * (idx + 1)

        # This seems to be to convert I from a column back into a 64x64 matrix
        # aka: it's equal to A from earlier
        # img(:) = I;
        # Sub is a square (in this case 14x14)
        # sub = img(X1:X2,Y1:Y2);
        sub = A[X1:X2, X1:X2]
        # roi(S0) = sum(sum(sub))/npx;
        roi.append(sum(sum(sub))/npx)

        for size in range(1, roi_width + 1):
            # This was originally ro2, shadows the one outside loop.
            # Can probably delete that var since only used to set X2
            ro2 = int(size / 2)
            # Note that these are lower case (terrible, terrible var names)
            x1 = (num_pix // 2) - ro2 - 1 # Subtract one here to account for 0 indexing
            x2 = x1 + size # DONT subtract one here to account for exclusive upper bound
            # They reuse x1 and x2 for Y here, but why not with the capital versions?
            # A here is flipped... compared with x1 = 30 (31 in matlab)
            #   and x2 = 33 and columns and rows were flipped.
            sub = A[x1:x2, x1:x2]
            # This should have been declared outside loop as size NxM
            roir[idx - TRIM_TR, size - 1] = sub.mean()

        # 'do the phase' (????)
        # This part in the orig code never executes because numwin
        # gets hardcoded to 3 and the if statement executes when == 4. So...
        # Ignore for now I guess.

    return Iodd, Ieven, Syy, Syt, St, Stt, S0, roi, roir


def make_figures(Iave, Isub, Isd, sfnr, num_pix, output_prefix):
    # This is the 'generate images' part of Joe's code

    # Watch it, may need the transpose here
    Iave = Iave.reshape((num_pix, num_pix))
    Isub = Isub.reshape((num_pix, num_pix))
    Isd = 10 * Isd.reshape((num_pix, num_pix))
    Isfnr = 10 * sfnr.reshape((num_pix, num_pix))

    ######## Make figures here
    # figure(1)
    fig, ((Iaveimg, Isdimg), (Isubimg, Isfnrimg)) = plt.subplots(2, 2)

    # Should pcolormesh be used instead of imshow?
    Iaveimg.imshow(Iave, cmap='gray')
    Iaveimg.title.set_text('Average')

    Isdimg.imshow(Isd, cmap='gray')
    Isdimg.title.set_text('Std')

    Isubimg.imshow(Isub, cmap='gray')
    Isubimg.title.set_text('Noise')

    # This one is much lighter in the matlab image... need to investigate closer
    Isfnrimg.imshow(Isfnr, cmap='gray')
    Isfnrimg.title.set_text('SFNR')

    plt.tight_layout()
    plt.savefig(output_prefix + "_images.jpg")


def fluctuation_analysis(N, roi):
    x = range(N)
    p = np.polyfit(x, roi, 2)
    yfit = np.polyval(p, x)
    y = roi - yfit
    return x, y, yfit


def make_plots(N, roi, roir, r2, output_prefix, num_win=3, TR=2.0):
    """This does the big from 'generate plots' comment onward
    """
    fig, (signal, spectrum, rel_std) = plt.subplots(num_win, 1)

    x, y, yfit = fluctuation_analysis(N, roi)


    # Return to adding the y-axis + grid and correct scale etc.
    # Must also make the csv of values (some of the comments here needed)


    # 'signal'
    plt.grid()
    signal.plot(x, roi, x, yfit)
    signal.set(xlabel='Frame Num', yaxis='Raw Signal')
    # signal.xaxis.set_label_text('Frame Num')
    # signal.yaxis.set_label_text('Raw Signal')
    # grid   -> This toggles the visibility of grid lines
    # m=mean(roi);
    # sd=std(y);
    # drift = (yfit(N)-yfit(1))/m;
    # title(sprintf('%s   percent fluct (trend removed), drift= %5.2f %5.2f', fkern, 100*sd/m, 100*drift));

    # 'spectrum'
    # z = fft(y);
    z = np.fft.fft(y) # The first value is wrong... but only first?
    # fs = 1/TR;
    fs = 1 / TR
    # nf = N/2+1;
    nf = N // 2 + 1
    # f = 0.5*(1:nf)*fs/nf;
    f = [0.5 * num * fs / nf for num in range(1, nf + 1)]

    # subplot(numwin,1,2);plot(f, abs(z(1:nf)));grid
    vals = np.abs(z[:nf]) # As a result of z being wrong on first val, this is too
    spectrum.plot(f, vals)
    spectrum.xaxis.set_label_text('Frequency, Hz')
    spectrum.yaxis.set_label_text('Spectrum')
    # ylabel('spectrum');
    # xlabel('frequency, Hz');
    # ax = axis;
    # text(ax(2)*.2, ax(4)*.8, sprintf('mean, SNR, SFNR = %5.1f  %5.1f  %5.1f', meanI, snr, sfnrI));


    # % ROI-varied analysis
    # t = (1:N);
    t = list(range(N))
    rr = list(range(r2))
    F = []
    for num in rr:
        # this used the ' operator originally.. ctranspose (?)
        y = roir[:, num]
        yfit = np.polyval(np.polyfit(t, y, 2), t)
        F.append(np.std(y - yfit) / np.mean(yfit))

    # rr = (r1:r2); -> this is 1-15 (i.e. equal to range(r2) from above)
    # % percent
    # F = 100*F;
    F = [100 * item for item in F] # Could probably move *100 into loop
    # fcalc = F(1)./rr;
    fcalc = [F[0] / (num + 1) for num in rr]
    # % decorrelation distance
    # rdc = F(1)/F(r2);
    rdc = F[0]/F[-1]
    #
    # subplot(numwin,1,3);
    # loglog(rr, F, '-x', rr, fcalc, '--');
    # '-x' and '--' seem to be setting the line markers to use (two lines on plot)
    rel_std.xaxis.set_label_text('ROI full width, pixels')
    rel_std.yaxis.set_label_text('Relative STD, %')
    rel_std.set_yscale('log')
    rel_std.set_xscale('log')
    rel_std.plot(rr, F, rr, fcalc)
    # plt.loglog(rr, F, rr, fcalc)
    # plt.grid()
    # plt.tight_layout()
    plt.savefig(output_prefix + "_plots.jpg")
    # grid
    # xlabel('ROI full width, pixels');
    # ylabel('Relative std, %');
    # axis([r1 r2 .01 1]);
    # text(6, 0.5, 'solid: meas   dashed: calc');
    # text(6, 0.25, sprintf('rdc = %3.1f pixels',rdc));
