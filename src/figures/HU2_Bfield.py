import numpy as np
import matplotlib.pyplot as plt
from _methods import AB09_theory_withVA
#from _methods import pm_color

pm_color = {"+":"#c70b00", "-":"#789aff"}

# We will be using blue colors to represent right/left handed modes
# Right handed modes (+): #c70b00
# Left handed modes (-): #789aff

# Setup figure
plt.figure(1).clf()
fig = plt.figure(1)
fig.set_size_inches(7, 9.)
ax = [fig.add_subplot(311+_) for _ in range(3)]

# Load data
pathEB = "../data/EB_HU2.npy"
pathfft = "../data/fft_HU2.npy"

def load(path):
    return np.load(path, allow_pickle=True).all()

d = load(pathEB)
f = load(pathfft)

#================ subplot 1 (ax[0]) ================
# Left (-) and right (+) handed theory predictions

kk, wp, wm = AB09_theory_withVA(4.5381174, 100., 0.02, 20.)
ww = {"+":wp, "-":wm}
wmax = {"+":wp.max(),"-":wm.max()}
kmax = {"+":kk[wp.argmax()], "-":kk[wm.argmax()]}


yl = [1e-3, 1e-1]

for pm in "+-":
    ax[0].loglog(kk, ww[pm], color=pm_color[pm])
    ax[0].plot(2*[kmax[pm]], yl, '--', color=pm_color[pm])


#================ subplot 2 (ax[1]) ================
# Magnetic field vs x for different times (color)

# pick the time range, 0 is noise, late times is saturation
ip_first = 1
ip_last = 25

tt = d['tt'][ip_first:ip_last]
bp2 = np.mean(np.array(d['by'])**2 + np.array(d['bz'])**2, axis=1)

for _c,_t in enumerate(tt):
    # Plot By of x for different times
    cid = plt.cm.jet(1. - _t/tt[-1])
    x = d['xx']
    By = d['by'][ip_last - _c]
    if _c%3 == 0:
        ax[1].plot(x, By, color=cid)

    # pick out the subset of time for color
    jp0 = ip_last - _c - 1
    jp1 = ip_last - _c + 1
    t = d['tt'][jp0:jp1]
    Bp = bp2[jp0:jp1]

    # Plot the mean Bperp in time
    ax[2].plot(t, Bp, color=cid)


#================ subplot 3 (ax[2]) ================
# FFT[B] in time for + and - of fastest growing modes
fpm = {"+":np.abs(np.array(f['by']) + 1.j*np.array(f['bz']))**2,
       "-":np.abs(np.array(f['by']) - 1.j*np.array(f['bz']))**2}

for pm in "+-":
    kp = np.abs(f['kk'] - kmax[pm]).argmin()

    # select wave number & time subset and normalize to Bperp
    _f = np.mean(fpm[pm][ip_first:ip_last, kp-1:kp+2], axis=1)
    _f = _f/_f[0]*bp2[ip_first]

    ax[2].plot(tt, _f, color=pm_color[pm])

# plotting the growth rates from the theory
theta = np.arange(0.,2.*np.pi, .1)
xshft = 700.
yshft = {"+":1.7, "-":2.4}

A0 = {"+":bp2[ip_first], "-":.4*bp2[ip_first]}
te = {"+":7, "-":15}

for pm in "+-":
    _t = tt[:te[pm]]
    ax[1].plot(theta/kmax[pm] + xshft, .15*np.sin(theta) + yshft[pm],
               color=pm_color[pm])

    ax[1].text(xshft, yshft[pm], '$\lambda^{}_{{\\rm max}}$'.format(pm),
               ha='right', va='center', color=pm_color[pm], size=10)

    ax[2].plot(_t, A0[pm]*np.exp(2.*wmax[pm]*_t), '--',
               color=pm_color[pm])

#================ Cleaning up the figures ================

ax[0].set_xlabel(r"$k\ (d_i^{-1})$")
ax[0].set_ylabel(r"$\Gamma\ (\Omega_{ci})$")

ax[0].set_xlim(kk[0], kk[-1])
ax[0].set_ylim(yl)

ax[1].set_xlabel(r"$x\ (d_i)$")
ax[1].set_ylabel(r"$B_y\ (B_0)$")
ax[1].set_xlim(d['xx'][0], d['xx'][-1])

ax[2].set_xlabel(r"time$\ (\Omega_{pi}^{-1})$")
ax[2].set_ylabel(r"$B_\perp\ (B_0)$")
ax[2].set_yscale('log')
ax[2].set_xlim(tt[0], tt[-1])

fig.savefig("./HU2_Bgrowth.pdf")

