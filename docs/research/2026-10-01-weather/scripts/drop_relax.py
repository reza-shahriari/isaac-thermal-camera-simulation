"""Linearised thermal relaxation of a falling raindrop toward its psychrometric
(wet-bulb) temperature. Heat+vapour balance as in Loftus & Wordsworth 2021 eqs 10-11
(Rogers & Yau 1996 ch.7); ventilation f = 0.78 + 0.308 Re^0.5 X^(1/3) (X=Sc or Pr),
Beard & Pruppacher 1971 / Pruppacher & Rasmussen 1979; v_t from Atlas et al. 1973.
Linearising: tau = rho_w c_w r^2 / (3 [k_a f_h + L D_v f_v d(rho_vs)/dT])."""
import math
T=283.15; p=101325.0
rho_a=p/(287.05*T); mu=1.76e-5; nu=mu/rho_a
k_a=0.0250; D_v=2.45e-5; L=2.477e6; Rv=461.5
es=611.2*math.exp(17.67*(T-273.15)/(T-29.65)); rho_vs=es/(Rv*T)
drho=rho_vs*(L/(Rv*T*T)-1/T)
Sc=nu/D_v; Pr=mu*1005/k_a
print(f"T=10C rho_a={rho_a:.3f} nu={nu:.3e} Sc={Sc:.2f} Pr={Pr:.2f} drho_vs/dT={drho:.3e} kg/m3/K; L*Dv*drho={L*D_v*drho:.4f} vs k_a={k_a}")
def f(Re,X):
    y=Re**0.5*X**(1/3)
    return 1+0.108*y*y if y<1.4 else 0.78+0.308*y
print(" D_mm  v_m/s   Re    f_h   f_v   tau_s  e-fold_dist_m  95%dist_m")
for D in [0.2,0.5,1.0,2.0,3.0,4.0,5.0]:
    v=9.65-10.3*math.exp(-0.6*D)
    r=D*1e-3/2
    Re=v*2*r/nu
    fh=f(Re,Pr); fv=f(Re,Sc)
    tau=1000*4186*r*r/(3*(k_a*fh+L*D_v*fv*drho))
    print(f"{D:5.1f} {v:6.2f} {Re:7.0f} {fh:5.2f} {fv:5.2f} {tau:6.2f} {v*tau:8.1f} {3*v*tau:8.1f}")
