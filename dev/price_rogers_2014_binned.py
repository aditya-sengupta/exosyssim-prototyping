#  Downloaded from https://www.cfa.harvard.edu/~eprice/files/price_rogers_2014_binned.py on 2020-06-10.
#
#  Copyright 2014 Ellen Price <eprice@caltech.edu>
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following disclaimer
#    in the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of the California Institute of Technology nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import numpy as np


def prVarTc(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((-3*sigma**2*tau**2)/
            (2*texp*gamma*delta**2 - 6*gamma*delta**2*tau))
    else:
        return ((3*texp**2*sigma**2)/
            (6*texp*gamma*delta**2 - 2*gamma*delta**2*tau))


def prVarTau(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((24*sigma**2*tau**2*
            (-(texp**4*Ttot) + 8*texp**3*Ttot*tau -
            3*texp*(-10*T**2 + 10*T*Ttot + texp*(2*texp + 5*Ttot))*tau**2 +
            10*(-9*T**2 + 9*T*Ttot + texp*(3*texp + Ttot))*tau**3 -
            75*Ttot*tau**4 + 60*tau**5))/(gamma*delta**2*
            (9*texp**5*Ttot - 40*texp**2*Ttot*tau**3 +
            120*texp*(3*Ttot - 2*tau)*tau**4 + 360*tau**5*
            (-Ttot + tau) - 12*texp**3*tau**2*(Ttot + 4*tau) +
            texp**4*tau*(-35*Ttot + 54*tau) + 24*T**2*(texp - 3*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((24*texp**2*sigma**2*
            (15*texp**3*(-6*T**2 + texp*(4*texp - 5*Ttot) + 6*T*Ttot) +
            10*texp**2*(3*T**2 - 3*T*Ttot + texp*Ttot)*tau +
            15*texp**2*(2*texp - Ttot)*tau**2 + 2*texp*(-3*texp + 4*Ttot)*
            tau**3 - Ttot*tau**4))/(gamma*delta**2*tau*
            (12*texp**3*(-6*T**2 + texp*(4*texp - 5*Ttot) + 6*T*Ttot) +
            4*texp**2*(6*T**2 + texp*(30*texp - 13*Ttot) - 6*T*Ttot)*tau +
            4*texp**2*(-12*texp + 11*Ttot)*tau**2 + texp*(6*texp - 11*Ttot)*
            tau**3 + Ttot*tau**4)))


def prVarT(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((6*sigma**2*tau**2*
            (-3*texp**3*(8*T**2 - 8*T*Ttot + 3*texp*Ttot) + 8*texp**3*Ttot*tau +
            120*T*texp*(T - Ttot)*tau**2 + 40*(-3*T**2 + 3*T*Ttot + texp*Ttot)*
            tau**3 - 60*Ttot*tau**4))/(gamma*delta**2*
            (9*texp**5*Ttot - 40*texp**2*Ttot*tau**3 +
            120*texp*(3*Ttot - 2*tau)*tau**4 + 360*tau**5*
            (-Ttot + tau) - 12*texp**3*tau**2*(Ttot + 4*tau) +
            texp**4*tau*(-35*Ttot + 54*tau) + 24*T**2*(texp - 3*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((6*texp**2*sigma**2*
            (8*texp**2*(3*T**2 - 3*T*Ttot + texp*Ttot) + 20*texp**2*Ttot*tau -
            8*texp*Ttot*tau**2 + Ttot*tau**3))/(gamma*delta**2*
            (12*texp**3*(6*T**2 - 6*T*Ttot + texp*(-4*texp + 5*Ttot)) +
            4*texp**2*(-6*T**2 + 6*T*Ttot + texp*(-30*texp + 13*Ttot))*tau +
            4*texp**2*(12*texp - 11*Ttot)*tau**2 + texp*(-6*texp + 11*Ttot)*
            tau**3 - Ttot*tau**4)))


def prVarDelta(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((-24*sigma**2*(texp*Ttot - 3*(Ttot - 2*tau)*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3))/(gamma*
            (9*texp**5*Ttot - 40*texp**2*Ttot*tau**3 +
            120*texp*(3*Ttot - 2*tau)*tau**4 + 360*tau**5*
            (-Ttot + tau) - 12*texp**3*tau**2*(Ttot + 4*tau) +
            texp**4*tau*(-35*Ttot + 54*tau) + 24*T**2*(texp - 3*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((24*texp**2*sigma**2*(6*texp**2 - 3*texp*Ttot +
            Ttot*tau))/(gamma*
            (12*texp**3*(6*T**2 - 6*T*Ttot + texp*(-4*texp + 5*Ttot)) +
            4*texp**2*(-6*T**2 + 6*T*Ttot + texp*(-30*texp + 13*Ttot))*tau +
            4*texp**2*(12*texp - 11*Ttot)*tau**2 + texp*(-6*texp + 11*Ttot)*
            tau**3 - Ttot*tau**4)))


def prCovTauT(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((36*(2*T - Ttot)*sigma**2*tau**3*
            (2*texp**3 - 5*texp**2*tau + 10*tau**3))/(gamma*
            delta**2*(9*texp**5*Ttot - 40*texp**2*Ttot*tau**3 +
            120*texp*(3*Ttot - 2*tau)*tau**4 + 360*tau**5*
            (-Ttot + tau) - 12*texp**3*tau**2*(Ttot + 4*tau) +
            texp**4*tau*(-35*Ttot + 54*tau) + 24*T**2*(texp - 3*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((-36*texp**4*(2*T - Ttot)*sigma**2*
            (10*texp - 3*tau))/(gamma*delta**2*
            (12*texp**3*(6*T**2 - 6*T*Ttot + texp*(-4*texp + 5*Ttot)) +
            4*texp**2*(-6*T**2 + 6*T*Ttot + texp*(-30*texp + 13*Ttot))*tau +
            4*texp**2*(12*texp - 11*Ttot)*tau**2 + texp*(-6*texp + 11*Ttot)*
            tau**3 - Ttot*tau**4)))


def prCovTauDelta(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((-12*sigma**2*tau*
            (texp*Ttot - 3*(Ttot - 2*tau)*tau)*(2*texp**3 - 5*texp**2*tau +
            10*tau**3))/(gamma*delta*(9*texp**5*Ttot -
            40*texp**2*Ttot*tau**3 + 120*texp*(3*Ttot - 2*tau)*tau**4 +
            360*tau**5*(-Ttot + tau) - 12*texp**3*tau**2*
            (Ttot + 4*tau) + texp**4*tau*(-35*Ttot + 54*tau) +
            24*T**2*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((12*texp**2*sigma**2*(10*texp - 3*tau)*
            (6*texp**2 - 3*texp*Ttot + Ttot*tau))/(gamma*delta*
            (12*texp**3*(6*T**2 - 6*T*Ttot + texp*(-4*texp + 5*Ttot)) +
            4*texp**2*(-6*T**2 + 6*T*Ttot + texp*(-30*texp + 13*Ttot))*tau +
            4*texp**2*(12*texp - 11*Ttot)*tau**2 + texp*(-6*texp + 11*Ttot)*
            tau**3 - Ttot*tau**4)))


def prCovTDelta(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((72*(2*T - Ttot)*sigma**2*tau**2*
            (texp**3 - 5*texp*tau**2 + 5*tau**3))/(gamma*delta*
            (9*texp**5*Ttot - 40*texp**2*Ttot*tau**3 +
            120*texp*(3*Ttot - 2*tau)*tau**4 + 360*tau**5*
            (-Ttot + tau) - 12*texp**3*tau**2*(Ttot + 4*tau) +
            texp**4*tau*(-35*Ttot + 54*tau) + 24*T**2*(texp - 3*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((72*texp**4*(-2*T + Ttot)*sigma**2)/
            (gamma*delta*(12*texp**3*(6*T**2 - 6*T*Ttot +
            texp*(-4*texp + 5*Ttot)) + 4*texp**2*(-6*T**2 + 6*T*Ttot +
            texp*(-30*texp + 13*Ttot))*tau + 4*texp**2*(12*texp - 11*Ttot)*
            tau**2 + texp*(-6*texp + 11*Ttot)*tau**3 - Ttot*tau**4)))


def prVarBsq(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((sigma**2*(T**2*delta**2*(9*texp**5 - 35*texp**4*tau -
            12*texp**3*tau**2 - 40*texp**2*tau**3 + 360*texp*tau**4 -
            360*tau**5 - 24*T*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 +
            5*tau**3)) - 24*f0*T*delta*(2*T**2*(texp - 3*tau)*
            (texp**3 - 5*texp**2*tau + 5*texp*tau**2 + 5*tau**3) +
            6*T*tau**2*(3*texp**3 - 5*texp**2*tau - 5*texp*tau**2 +
            15*tau**3) + tau**2*(-9*texp**4 + 8*texp**3*tau +
            40*texp*tau**3 - 60*tau**4)) +
            24*f0**2*(120*T**4*(texp - 3*tau)*tau**2 + 120*T**3*Ttot*tau**2*
            (-texp + 3*tau) + 12*T*Ttot*tau**2*(3*texp**3 -
            5*texp**2*tau - 5*texp*tau**2 + 15*tau**3) +
            Ttot*tau**2*(-9*texp**4 + 8*texp**3*tau + 40*texp*tau**3 -
            60*tau**4) + T**2*(-(texp**4*Ttot) + 13*texp**3*Ttot*tau -
            texp**2*(54*texp + 25*Ttot)*tau**2 + 20*texp*(9*texp + 2*Ttot)*
            tau**3 + 15*(2*texp - 23*Ttot)*tau**4 + 90*tau**5))))/
            (4*f0**3*gamma*delta*tau**2*(9*texp**5*Ttot -
            40*texp**2*Ttot*tau**3 + 120*texp*(3*Ttot - 2*tau)*tau**4 +
            360*tau**5*(-Ttot + tau) - 12*texp**3*tau**2*
            (Ttot + 4*tau) + texp**4*tau*(-35*Ttot + 54*tau) +
            24*T**2*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((sigma**2*(T**2*delta**2*tau**3*
            (12*(6*T - 5*texp)*texp**3 - 4*texp**2*(6*T + 13*texp)*tau +
            44*texp**2*tau**2 - 11*texp*tau**3 + tau**4) +
            24*f0**2*texp**2*(-60*T**2*texp**3*(6*T**2 - 6*T*Ttot +
            texp*(-4*texp + 5*Ttot)) + 40*T**2*texp**2*(3*T**2 - 3*T*Ttot +
            texp*Ttot)*tau + 120*T*texp**2*(-T + texp)*Ttot*tau**2 +
            texp*(-24*T*texp*Ttot - 8*texp**2*Ttot + T**2*(6*texp + 73*Ttot))*
            tau**3 - (11*T**2 + 20*texp**2)*Ttot*tau**4 +
            8*texp*Ttot*tau**5 - Ttot*tau**6) + 24*f0*T*texp**2*delta*
            tau**2*(4*T**2*(5*texp - 2*tau)*(3*texp - tau) +
            12*T*texp**2*(-5*texp + tau) + tau*(8*texp**3 +
            20*texp**2*tau - 8*texp*tau**2 + tau**3))))/
            (4*f0**3*gamma*delta*tau**5*
            (12*texp**3*(-6*T**2 + texp*(4*texp - 5*Ttot) + 6*T*Ttot) +
            4*texp**2*(6*T**2 + texp*(30*texp - 13*Ttot) - 6*T*Ttot)*tau +
            4*texp**2*(-12*texp + 11*Ttot)*tau**2 + texp*(6*texp - 11*Ttot)*
            tau**3 + Ttot*tau**4)))


def prVarTau0sq(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((sigma**2*tau**2*(T**2*delta**2*(9*texp**5 - 35*texp**4*tau -
            12*texp**3*tau**2 - 40*texp**2*tau**3 + 360*texp*tau**4 -
            360*tau**5 - 24*T*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 +
            5*tau**3)) - 24*f0*T*delta*
            (-6*T*tau**2*(texp**3 + 5*texp**2*tau - 15*texp*tau**2 +
            5*tau**3) + 2*T**2*(texp - 3*tau)*(texp**3 - 5*texp**2*tau +
            5*texp*tau**2 + 5*tau**3) + tau**2*(9*texp**4 -
            8*texp**3*tau - 40*texp*tau**3 + 60*tau**4)) +
            24*f0**2*(120*T**4*(texp - 3*tau)*tau**2 + 120*T**3*Ttot*tau**2*
            (-texp + 3*tau) + 12*T*Ttot*tau**2*(texp**3 +
            5*texp**2*tau - 15*texp*tau**2 + 5*tau**3) +
            Ttot*tau**2*(-9*texp**4 + 8*texp**3*tau + 40*texp*tau**3 -
            60*tau**4) - T**2*(texp**4*Ttot + 15*(23*Ttot - 22*tau)*
            tau**4 + texp**3*tau*(-13*Ttot + 6*tau) +
            5*texp**2*tau**2*(5*Ttot + 12*tau) - 10*texp*tau**3*
            (4*Ttot + 27*tau)))))/(64*f0*gamma*delta**3*
            (9*texp**5*Ttot - 40*texp**2*Ttot*tau**3 +
            120*texp*(3*Ttot - 2*tau)*tau**4 + 360*tau**5*
            (-Ttot + tau) - 12*texp**3*tau**2*(Ttot + 4*tau) +
            texp**4*tau*(-35*Ttot + 54*tau) + 24*T**2*(texp - 3*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((sigma**2*(T**2*delta**2*tau**3*
            (12*(6*T - 5*texp)*texp**3 - 4*texp**2*(6*T + 13*texp)*tau +
            44*texp**2*tau**2 - 11*texp*tau**3 + tau**4) +
            24*f0**2*texp**2*(60*T**2*texp**3*(-6*T**2 + texp*(4*texp - 5*Ttot) +
            6*T*Ttot) + 40*T**2*texp**2*(3*T**2 - 3*T*Ttot + texp*Ttot)*tau +
            120*T*texp**2*(4*T*texp - (T + texp)*Ttot)*tau**2 +
            texp*(72*T*texp*Ttot - 8*texp**2*Ttot + T**2*(-186*texp + 73*Ttot))*
            tau**3 - (11*T**2 + 20*texp**2)*Ttot*tau**4 +
            8*texp*Ttot*tau**5 - Ttot*tau**6) + 24*f0*T*texp**2*delta*
            tau**2*(4*T**2*(5*texp - 2*tau)*(3*texp - tau) +
            12*T*texp**2*(-5*texp + 3*tau) - tau*(8*texp**3 +
            20*texp**2*tau - 8*texp*tau**2 + tau**3))))/
            (64*f0*gamma*delta**3*tau*
            (12*texp**3*(-6*T**2 + texp*(4*texp - 5*Ttot) + 6*T*Ttot) +
            4*texp**2*(6*T**2 + texp*(30*texp - 13*Ttot) - 6*T*Ttot)*tau +
            4*texp**2*(-12*texp + 11*Ttot)*tau**2 + texp*(6*texp - 11*Ttot)*
            tau**3 + Ttot*tau**4)))


def prVarR(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((sigma**2*(-24*f0**2*(texp*Ttot - 3*(Ttot - 2*tau)*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3) + 48*f0*delta*
            (T*(texp - 3*tau) + 3*tau**2)*(texp**3 - 5*texp*tau**2 +
            5*tau**3) + delta**2*(9*texp**5 - 35*texp**4*tau -
            12*texp**3*tau**2 - 40*texp**2*tau**3 + 360*texp*tau**4 -
            360*tau**5 - 24*T*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 +
            5*tau**3))))/(4*f0**3*gamma*delta*
            (9*texp**5*Ttot - 40*texp**2*Ttot*tau**3 +
            120*texp*(3*Ttot - 2*tau)*tau**4 + 360*tau**5*
            (-Ttot + tau) - 12*texp**3*tau**2*(Ttot + 4*tau) +
            texp**4*tau*(-35*Ttot + 54*tau) + 24*T**2*(texp - 3*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((sigma**2*(-48*f0*texp**2*delta*
            (3*texp*(-T + texp) + T*tau) + 24*f0**2*texp**2*
            (6*texp**2 - 3*texp*Ttot + Ttot*tau) +
            delta**2*(60*texp**4 + 52*texp**3*tau - 44*texp**2*tau**2 +
            11*texp*tau**3 - tau**4 + 24*T*texp**2*(-3*texp + tau))))/
            (4*f0**3*gamma*delta*
            (12*texp**3*(6*T**2 - 6*T*Ttot + texp*(-4*texp + 5*Ttot)) +
            4*texp**2*(-6*T**2 + 6*T*Ttot + texp*(-30*texp + 13*Ttot))*tau +
            4*texp**2*(12*texp - 11*Ttot)*tau**2 + texp*(-6*texp + 11*Ttot)*
            tau**3 - Ttot*tau**4)))


def prCovBsqTau0sq(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((sigma**2*(-48*f0*T**2*delta*(T*(texp - 3*tau) + 3*tau**2)*
            (texp**3 - 5*texp**2*tau + 5*texp*tau**2 + 5*tau**3) +
            T**2*delta**2*(9*texp**5 - 35*texp**4*tau - 12*texp**3*tau**2 -
            40*texp**2*tau**3 + 360*texp*tau**4 - 360*tau**5 -
            24*T*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3)) +
            24*f0**2*(120*T**4*(texp - 3*tau)*tau**2 + 120*T**3*Ttot*tau**2*
            (-texp + 3*tau) - 24*T*Ttot*tau**2*(texp**3 -
            5*texp*tau**2 + 5*tau**3) + Ttot*tau**2*
            (9*texp**4 - 8*texp**3*tau - 40*texp*tau**3 + 60*tau**4) +
            T**2*(-(texp**4*Ttot) + 13*texp**3*Ttot*tau +
            texp**2*(18*texp - 25*Ttot)*tau**2 + 20*texp*(3*texp + 2*Ttot)*
            tau**3 - 15*(6*texp + 23*Ttot)*tau**4 + 450*tau**5))))/
            (16*f0**2*gamma*delta**2*(9*texp**5*Ttot -
            40*texp**2*Ttot*tau**3 + 120*texp*(3*Ttot - 2*tau)*tau**4 +
            360*tau**5*(-Ttot + tau) - 12*texp**3*tau**2*
            (Ttot + 4*tau) + texp**4*tau*(-35*Ttot + 54*tau) +
            24*T**2*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((sigma**2*(96*f0*T**2*texp**2*delta*
            (5*texp - 2*tau)*tau**2*(3*(T - texp)*texp - T*tau) +
            T**2*delta**2*tau**3*(12*(6*T - 5*texp)*texp**3 -
            4*texp**2*(6*T + 13*texp)*tau + 44*texp**2*tau**2 -
            11*texp*tau**3 + tau**4) + 24*f0**2*texp**2*
            (60*T**2*texp**3*(-6*T**2 + texp*(4*texp - 5*Ttot) + 6*T*Ttot) +
            40*T**2*texp**2*(3*T**2 - 3*T*Ttot + texp*Ttot)*tau +
            120*T**2*texp**2*(2*texp - Ttot)*tau**2 +
            texp*(-24*T*texp*Ttot + 8*texp**2*Ttot + T**2*(-42*texp + 73*Ttot))*
            tau**3 + (-11*T**2 + 20*texp**2)*Ttot*tau**4 -
            8*texp*Ttot*tau**5 + Ttot*tau**6)))/(16*f0**2*gamma*
            delta**2*tau**3*(12*texp**3*(-6*T**2 + texp*(4*texp - 5*Ttot) +
            6*T*Ttot) + 4*texp**2*(6*T**2 + texp*(30*texp - 13*Ttot) - 6*T*Ttot)*
            tau + 4*texp**2*(-12*texp + 11*Ttot)*tau**2 +
            texp*(6*texp - 11*Ttot)*tau**3 + Ttot*tau**4)))


def prCovBsqR(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((sigma**2*(24*f0**2*(-(T*texp**4*Ttot) + 8*T*texp**3*Ttot*tau -
            2*texp**2*(9*T*texp + 10*T*Ttot - 3*texp*Ttot)*tau**2 +
            10*T*texp*(3*texp + Ttot)*tau**3 + 15*(-2*texp*Ttot +
            T*(2*texp + Ttot))*tau**4 + 30*(-3*T + Ttot)*tau**5) +
            T*delta**2*(-9*texp**5 + 35*texp**4*tau + 12*texp**3*tau**2 +
            40*texp**2*tau**3 - 360*texp*tau**4 + 360*tau**5 +
            24*T*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3)) -
            12*f0*delta*tau*(10*T**2*texp*(texp - 3*tau)*
            (texp - 2*tau) - 6*T*tau*(2*texp**3 - 5*texp**2*tau +
            10*tau**3) + tau*(9*texp**4 - 8*texp**3*tau -
            40*texp*tau**3 + 60*tau**4))))/(4*f0**3*gamma*
            delta*tau*(9*texp**5*Ttot - 40*texp**2*Ttot*tau**3 +
            120*texp*(3*Ttot - 2*tau)*tau**4 + 360*tau**5*
            (-Ttot + tau) - 12*texp**3*tau**2*(Ttot + 4*tau) +
            texp**4*tau*(-35*Ttot + 54*tau) + 24*T**2*(texp - 3*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((sigma**2*(-48*f0**2*texp**2*(15*T*texp**2*(2*texp - Ttot) +
            texp*(-6*T*texp + 11*T*Ttot - 3*texp*Ttot)*tau -
            2*T*Ttot*tau**2) - 12*f0*texp**2*delta*
            (60*T*(T - texp)*texp**2 + 2*texp*(-25*T**2 + 9*T*texp + 4*texp**2)*
            tau + 10*(T**2 + 2*texp**2)*tau**2 - 8*texp*tau**3 +
            tau**4) + T*delta**2*tau*(60*texp**4 + 52*texp**3*tau -
            44*texp**2*tau**2 + 11*texp*tau**3 - tau**4 +
            24*T*texp**2*(-3*texp + tau))))/(4*f0**3*gamma*delta*
            tau**2*(12*texp**3*(-6*T**2 + texp*(4*texp - 5*Ttot) + 6*T*Ttot) +
            4*texp**2*(6*T**2 + texp*(30*texp - 13*Ttot) - 6*T*Ttot)*tau +
            4*texp**2*(-12*texp + 11*Ttot)*tau**2 + texp*(6*texp - 11*Ttot)*
            tau**3 + Ttot*tau**4)))


def prCovTau0sqR(delta, T, tau, f0, texp, Ttot, gamma, sigma):
    if tau > texp:
        return ((sigma**2*tau*(24*f0**2*(-(T*texp**4*Ttot) + 8*T*texp**3*Ttot*tau +
            2*texp**2*(3*T*texp - 10*T*Ttot - 3*texp*Ttot)*tau**2 +
            10*T*texp*(3*texp + Ttot)*tau**3 +
            15*(2*texp*Ttot + T*(-6*texp + Ttot))*tau**4 +
            30*(T - Ttot)*tau**5) + T*delta**2*(-9*texp**5 +
            35*texp**4*tau + 12*texp**3*tau**2 + 40*texp**2*tau**3 -
            360*texp*tau**4 + 360*tau**5 + 24*T*(texp - 3*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3)) - 12*f0*delta*tau*
            (10*T**2*texp*(texp - 3*tau)*(texp - 2*tau) +
            6*T*tau*(2*texp**3 + 5*texp**2*tau - 20*texp*tau**2 +
            10*tau**3) + tau*(-9*texp**4 + 8*texp**3*tau +
            40*texp*tau**3 - 60*tau**4))))/(16*f0**2*gamma*
            delta**2*(9*texp**5*Ttot - 40*texp**2*Ttot*tau**3 +
            120*texp*(3*Ttot - 2*tau)*tau**4 + 360*tau**5*
            (-Ttot + tau) - 12*texp**3*tau**2*(Ttot + 4*tau) +
            texp**4*tau*(-35*Ttot + 54*tau) + 24*T**2*(texp - 3*tau)*
            (texp**3 - 5*texp*tau**2 + 5*tau**3) -
            24*T*Ttot*(texp - 3*tau)*(texp**3 - 5*texp*tau**2 + 5*tau**3))))
    else:
        return ((sigma**2*(48*f0**2*texp**2*(15*T*texp**2*(2*texp - Ttot) +
            texp*(-18*T*texp + 11*T*Ttot + 3*texp*Ttot)*tau -
            2*T*Ttot*tau**2) + T*delta**2*tau*
            (12*(6*T - 5*texp)*texp**3 - 4*texp**2*(6*T + 13*texp)*tau +
            44*texp**2*tau**2 - 11*texp*tau**3 + tau**4) -
            12*f0*texp**2*delta*(6*T*texp**2*(10*texp - 7*tau) -
            10*T**2*(6*texp**2 - 5*texp*tau + tau**2) +
            tau*(8*texp**3 + 20*texp**2*tau - 8*texp*tau**2 +
            tau**3))))/(16*f0**2*gamma*delta**2*
            (12*texp**3*(6*T**2 - 6*T*Ttot + texp*(-4*texp + 5*Ttot)) +
            4*texp**2*(-6*T**2 + 6*T*Ttot + texp*(-30*texp + 13*Ttot))*tau +
            4*texp**2*(12*texp - 11*Ttot)*tau**2 + texp*(-6*texp + 11*Ttot)*
            tau**3 - Ttot*tau**4)))

