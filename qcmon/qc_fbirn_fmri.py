"""Refactored code from qc-fbirn-fmri and associated scripts for use in datman.

#### Update docs and reorganize code.
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)
matplotlib.use('Agg')

TRIM_TR = 4 # Trim the first 5 TRs (it's 4 because of 0 indexing)
NUM_SLICES = 20


class FMRIPhantom:

    def __init__(self, input_nii):
        # data == 'I4d'
        self.data = read_volume(input_nii)
        # num_pix == 'NPIX'
        self.num_pix = self.data.shape[0]
        # size_4d == 'i2'
        self.size_4d = self.data.shape[3]
        # num_TRs == 'N'
        self.num_TRs = self.size_4d - TRIM_TR
        # roi_width == 'R'
        self.roi_width = self.get_roi_width()
        # roi_low_idx == 'X1' (aka 'Y1')
        self.roi_low_idx = ((self.num_pix // 2) - (self.roi_width // 2)) - 1
        # roi_high_idx == 'X2' (aka 'Y2')
        self.roi_high_idx = self.roi_low_idx + self.roi_width
        # roi_num_pix == 'npx'
        self.roi_num_pix = sum(sum(np.ones((self.roi_width, self.roi_width))))

        (self.Iave, self.Isub, self.Isd, self.sfnr, self.roi, self.meanI,
         self.snr, self.sfnrI, roir) = self.calc_values()

        self.fluc_x, self.fluc_y, self.fluc_yfit = fluctuation_analysis(
            self.num_TRs, self.roi)
        self.drift = 100 * (self.fluc_yfit[-1] - self.fluc_yfit[0]) / np.mean(
            self.roi)
        self.sd = 100 * np.std(self.fluc_y) / np.mean(self.roi)

        trs = list(range(self.num_TRs))
        roi_widths = list(range(self.roi_width))
        self.F = []
        for num in roi_widths:
            tmp = roir[:, num]
            yfit = np.polyval(np.polyfit(trs, tmp, 2), trs)
            self.F.append(100 * (np.std(tmp - yfit) / np.mean(yfit)))

        self.fcalc = [self.F[0] / (num + 1) for num in roi_widths]
        self.rdc = self.F[0] / self.F[-1]

    def get_roi_width(self):
        if self.num_pix == 128:
            roi_width = 30
        elif self.num_pix == 64:
            roi_width = 15
        else:
            roi_width = 20
        return roi_width

    def calc_values(self):
        roi = []
        St = Stt = S0 = 0
        Iodd = np.zeros((self.num_pix * self.num_pix, 1))
        Ieven = np.zeros((self.num_pix * self.num_pix, 1))
        Syy = np.zeros((self.num_pix * self.num_pix, 1))
        Syt = np.zeros((self.num_pix * self.num_pix, 1))
        roir = np.zeros((self.num_TRs, self.roi_width))

        half_pix = self.num_pix // 2
        for idx in range(TRIM_TR, self.size_4d):
            A = np.squeeze(self.data[:, :, NUM_SLICES - 1, idx])
            I = A.flatten()[:, np.newaxis]

            if (idx + 1) % 2 == 0:
                Ieven = Ieven + I
            else:
                Iodd = Iodd + I

            Syt = Syt + I * (idx + 1)
            Syy = Syy + I * I
            S0 += 1
            St += (idx + 1)
            Stt = Stt + (idx + 1) * (idx + 1)

            sub = A[self.roi_low_idx:self.roi_high_idx,
                    self.roi_low_idx:self.roi_high_idx]
            roi.append(sum(sum(sub)) / self.roi_num_pix)

            for size in range(1, self.roi_width + 1):
                x1 = half_pix - int(size / 2) - 1
                x2 = x1 + size
                sub = A[x1:x2, x1:x2]
                roir[idx - TRIM_TR, size - 1] = sub.mean()

        # diff image
        Isub = Iodd - Ieven
        img = Isub.reshape((self.num_pix, self.num_pix)).T
        sub = img[self.roi_low_idx:self.roi_high_idx,
                  self.roi_low_idx:self.roi_high_idx]
        varI = np.var(sub.flatten(), ddof=1)

        # ave image
        Sy = Iodd + Ieven
        Iave = Sy / self.num_TRs
        img = Iave.reshape((self.num_pix, self.num_pix)).T
        sub = img[self.roi_low_idx:self.roi_high_idx,
                  self.roi_low_idx:self.roi_high_idx]
        meanI = sub.flatten().mean()

        # Trend line
        D = (Stt * S0 - St * St)
        a = (Syt * S0 - St * Sy) / D
        b = (Stt * Sy - St * Syt) / D

        # SD image
        Var = Syy + a * a * Stt + b * b * S0 + 2 * a * b * St - 2 * a * Syt - 2 * b * Sy
        Isd = np.sqrt(Var / (self.num_TRs - 1))

        # sfnr image
        sfnr = Iave / (Isd + np.finfo(float).eps)
        img = sfnr.reshape((self.num_pix, self.num_pix)).T
        sub = img[self.roi_low_idx:self.roi_high_idx + 1,
                  self.roi_low_idx:self.roi_high_idx + 1]
        sfnrI = sub.flatten().mean()

        snr = meanI / np.sqrt(varI / self.num_TRs)

        return Iave, Isub, Isd, sfnr, roi, meanI, snr, sfnrI, roir


def read_volume(input_nii, end_shape=None):
    """Read the nii file and raise an exception if it's too large.

    In the original matlab code (analyze_fmri_phantom) a 'centre_volume'
    function was used on images larger than [90, 90]. Our phantoms are never
    this large so I didnt bother to port the code over. If an exception is
    being raised here it means it's time to finally port it over :)
    """
    if not end_shape:
        end_shape = [90, 90]

    img_4d = nib.load(input_nii).get_fdata()
    for idx, max_len in enumerate(end_shape):
        if img_4d.shape[idx] > max_len:
            raise NotImplementedError(
                "Image is too large. Please center volume (see docstring note)."
            )

    return img_4d


def fluctuation_analysis(N, roi):
    x = range(N)
    p = np.polyfit(x, roi, 2)
    yfit = np.polyval(p, x)
    y = roi - yfit
    return x, y, yfit


def analyze_fmri_phantom(input_nii, output_prefix):
    """Refactored code from the matlab folder.
    """
    phantom = FMRIPhantom(input_nii)
    make_images_file(phantom, output_prefix)
    make_plots_file(phantom, output_prefix)
    make_stats_file(phantom, output_prefix)


def make_images_file(phantom, output_prefix):
    Iave = phantom.Iave.reshape((phantom.num_pix, phantom.num_pix))
    Isub = phantom.Isub.reshape((phantom.num_pix, phantom.num_pix))
    Isd = 10 * phantom.Isd.reshape((phantom.num_pix, phantom.num_pix))
    Isfnr = 10 * phantom.sfnr.reshape((phantom.num_pix, phantom.num_pix))

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


def make_plots_file(phantom, output_prefix, num_win=3, TR=2.0):
    fig, (signal, spectrum, rel_std) = plt.subplots(num_win, 1)

    fig.suptitle(
        f"Percent fluct. (trend removed), drift = {phantom.sd:5.2f} "
        f"{phantom.drift:5.2f}"
    )

    signal.grid()
    signal.plot(phantom.fluc_x, phantom.roi, phantom.fluc_x, phantom.fluc_yfit)
    signal.set(xlabel='Frame Num', ylabel='Raw Signal')
    signal.locator_params(axis="y", nbins=5)

    z = np.fft.fft(phantom.fluc_y)
    fs = 1 / TR
    nf = phantom.num_TRs // 2 + 1
    f = [0.5 * num * fs / nf for num in range(1, nf + 1)]
    vals = np.abs(z[:nf])

    spectrum.grid()
    spectrum.plot(f, vals)
    spectrum.set(xlabel='Frequency, Hz', ylabel='Spectrum')
    spectrum.locator_params(axis="y", nbins=5)
    title = (f"Mean = {phantom.meanI:5.1f}, SNR = {phantom.snr:5.1f}, "
             f"SFNR = {phantom.sfnrI:5.1f}")
    spectrum.set_title(title, fontsize='10')

    rel_std.grid()
    rel_std.set(xlabel='ROI full width, pixels', ylabel='Relative STD, %',
                yscale='log', xscale='log')
    roi_widths = list(range(phantom.roi_width))
    rel_std.plot(roi_widths, phantom.F, label="meas")
    rel_std.plot(roi_widths, phantom.fcalc, label="calc")
    rel_std.set_title(f"RDC = {phantom.rdc:3.1f} pixels", fontsize='10')
    rel_std.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_prefix + "_plots.jpg")


def make_stats_file(phantom, output_prefix):
    output_file = output_prefix + "_stats.csv"
    header = "mean,std,%fluct,drift,snr,sfnr,rdc\n"
    contents = (
        f"{phantom.meanI:09.3f},"
        f"{np.std(phantom.fluc_y):09.3f},"
        f"{phantom.sd:09.3f},"
        f"{phantom.drift:09.3f},"
        f"{phantom.snr:09.3f},"
        f"{phantom.sfnrI:09.3f},"
        f"{phantom.rdc:09.3f}\n"
    )
    with open(output_file, "w") as fh:
        fh.writelines([header, contents])
