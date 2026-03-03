import matplotlib.pyplot as plt
import numpy as np

__all__ = ['powerLaw', 'doublePowerLaw', 'plotPowerLaw']

def powerLaw(x, C, gamma):
    return C*x**gamma

def doublePowerLaw(x, C, gamma1, gamma2, x_0):
    return np.piecewise(x, [x <= x_0, x > x_0], 
                        [
                            lambda x: powerLaw(x, C, gamma1), 
                            lambda x: powerLaw(x, C*x_0**(gamma1-gamma2), gamma2)
                        ])

def plotPowerLaw(power:float, c:float=1, xbounds:tuple=(0,100), xlog:bool=False, ylog:bool=False):
    fig, ax = plt.subplots()
    x, y = powerLaw(power, c=c, xbounds=xbounds)
    ax.plot(x,y)
    ax.set_title(f'Power law with power {power}')
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    plt.show()

# def doublePowerLaw(power:float, c:float=1, dpower:float=0, powermult=0, xbounds:tuple=(0,100), xlog:bool=False, ylog:bool=False):
#     fig, ax = plt.subplots()
#     midx = (xbounds[1]-xbounds[0])/100
#     x1, y1 = powerLaw(power, c=c, xbounds=(xbounds[0], midx))
#     c2 = c*midx**(-dpower)
#     x2, y2 = powerLaw(power+dpower, c=c2, xbounds=(midx, xbounds[1]))
#     ax.plot(x1,y1*x1**powermult)
#     ax.plot(x2,y2*x2**powermult)
#     if xlog:
#         ax.set_xscale('log')
#     if ylog:
#         ax.set_yscale('log')
#     plt.show()