"""Fresnel normal and hemispherical emissivity of an optically thick smooth water/ice surface
(opaque sphere disc-average == hemispherical emissivity). n,k from Segelstein 1981 (water),
Warren & Brandt 2008 (ice)."""
import math, numpy as np, sys
sys.path.insert(0,'.')
from mie_rain import load, nk
def R(m, th):
    c=math.cos(th); s=math.sin(th)
    ct=np.sqrt(1-(s/m)**2+0j)
    rs=(c-m*ct)/(c+m*ct); rp=(m*c-ct)/(m*c+ct)
    return 0.5*(abs(rs)**2+abs(rp)**2)
seg=load('h2o_Segelstein.yml'); ice=load('ice_Warren-2008.yml'); w0=load('ice_Rowe-273K.yml')
for name,tab in [('water25C',seg),('water0C(Rowe)',w0),('ice(-7C)',ice)]:
    for wl in [0.55,1.0,1.55,3.8,4.0,8.0,9.0,10.0,11.0,12.0]:
        if wl<tab[0,0]: continue
        m=nk(tab,wl)
        th=np.linspace(0,math.pi/2,2001)
        Rv=np.array([R(m,t) for t in th])
        eh=np.trapezoid((1-Rv)*2*np.sin(th)*np.cos(th),th)
        print(f"{name:14s} {wl:5.2f} um m={m:.4f} eps_normal={1-Rv[0]:.4f} eps_hemi={eh:.4f} eps(60deg)={1-Rv[np.argmin(abs(th-math.radians(60)))]:.4f}")
