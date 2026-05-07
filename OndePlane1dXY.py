from numpy import pi, exp, sqrt, real, imag, zeros, linspace
import matplotlib

def PlaneWave(amp, k, omega, x, t):
    if(k<0):
        print("ca marche pas")
        exit
    else:
        return amp * exp(1j*(k*x - omega*t))
