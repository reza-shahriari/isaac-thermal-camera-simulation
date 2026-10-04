"""Fresnel reflectance / emissivity of a smooth water surface vs angle, band-averaged, plus
1/e absorption depth of liquid water (Segelstein 1981 n,k from the repo). Scratch computation."""
import numpy as np
rows=[l for l in open("/home/hunter/irsim-scaffold/data/nk/water.csv") if l[0].isdigit()]
wl,n,k=np.array([[float(v) for v in l.split(",")[:3]] for l in rows]).T
h=6.62607015e-34;c=2.99792458e8;kb=1.380649e-23
def planck(lam_um,T):
    lam=lam_um*1e-6
    return 2*h*c**2/lam**5/(np.exp(h*c/(lam*kb*T))-1)
def fresnel_R(m,theta):
    ct=np.cos(theta); st=np.sin(theta)
    ctt=np.sqrt(1-(st/m)**2+0j)
    rs=(ct-m*ctt)/(ct+m*ctt); rp=(m*ct-ctt)/(m*ct+ctt)
    return 0.5*(abs(rs)**2+abs(rp)**2)
for name,(a,b) in {"MWIR 3-5":(3.0,5.0),"LWIR 8-14":(8.0,14.0),"SWIR 0.9-1.7":(0.9,1.7)}.items():
    lam=np.linspace(a,b,400); m=np.interp(lam,wl,n)+1j*np.interp(lam,wl,k)
    w=planck(lam,293.15)
    out=[]
    for th in [0,30,45,60,70,75,80,85,88]:
        R=fresnel_R(m,np.radians(th))
        out.append((th,float(np.sum(R*w)/np.sum(w))))
    kk=np.interp(lam,wl,k); depth=lam/(4*np.pi*kk)
    print(name, " ".join(f"{t}deg:R={r:.3f}" for t,r in out))
    print("   1/e absorption depth (um): min %.1f  median %.1f  max %.1f" % (depth.min(), np.median(depth), depth.max()))
for lam0 in [3.5,4.0,4.5,8.0,10.0,12.0]:
    kk=np.interp(lam0,wl,k); nn=np.interp(lam0,wl,n)
    print(f"lambda={lam0} um: n={nn:.3f} k={kk:.4f} depth={lam0/(4*np.pi*kk):.1f} um  R0={abs((nn+1j*kk-1)/(nn+1j*kk+1))**2:.4f}")
