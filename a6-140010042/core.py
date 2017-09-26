
# vpm solve for Vinf
# create new blobs
# advect vort part
# check for penetration
# diffuse every one 
# check for penetration
# repeat



import numpy as np
import copy as cp
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import jit
import VPM as VPM
import integrator as itg
import time

def ctr_circ_vert(cy_cen,cy_rad,nop):
	npanel = nop
	thetas = np.linspace(0,2*np.pi,num=npanel+1,endpoint = True)
	vert = lambda th: cy_cen + cy_rad*np.cos(th) - 1j*cy_rad*np.sin(th)
	return list(map(vert,thetas[:-1]))

def plotting_parttrace(SimulationData,fignum):
    
	number = len(SimulationData)
	# print(number)
	for i in range(number):
		plt.figure(fignum)
		time.sleep(1.0)
		plt.scatter(np.array(SimulationData[i]).real,np.array(SimulationData[i]).imag)
		plt.axis('equal')		
		plt.plot(np.array(ctr_circ_vert(0j,1,50)).real,np.array(ctr_circ_vert(0j,1,50)).imag)
		plt.show()

def plot_velfield(grid,SimulationData_pos,SimulationData_strg,body,fignum,Vinf=1+0j):
    x,y = grid
    z = x + 1j*y
    vel = np.zeros_like(z,dtype = complex)
    # print(np.shape(vel))
    for i in range(np.shape(vel)[1]):
        vel[i] = np.array([VPM.vel_body([body],z[i][j]) for j in range(np.shape(vel)[0])])[0] + Vinf
        vel[i] += itg.vel_vort(z[i].copy(),SimulationData_pos.copy(),SimulationData_strg.copy(),body.get_pnl_len()[0]/np.pi)
    plt.figure(fignum)
    plt.quiver(x,y,vel.real,vel.imag,scale = 75.0)
    plt.plot(body.get_vert().real,body.get_vert().imag,'b')
    plt.gca().set_aspect('equal',adjustable='box')
    plt.show()

def plot_vortextrace(SimulationData_pos,SimulationData_strg,bodyvert,fignum):
    neg_pos,positive_pos = np.array([]),np.array([])
    for i in range(len(SimulationData_strg)):
        if SimulationData_strg[i]<0:
            neg_pos = np.append(neg_pos,[SimulationData_pos[i]])
        else:
            positive_pos = np.append(positive_pos,[SimulationData_pos[i]])
    plt.figure(fignum)
    plt.plot(positive_pos.real,positive_pos.imag,'bo',ms=2,label='Positive blobs')
    plt.plot(neg_pos.real,neg_pos.imag,'ro',ms=2,label='Negative blobs')
    plt.plot(bodyvert.real, bodyvert.imag,'k',lw=2)
    plt.legend(loc='best')
    plt.xlim([-2,5])
    plt.ylim([-2,2])
    plt.gca().set_aspect('equal',adjustable='box')
    plt.show()

def update_data(i,SimulationData,lines):
	# print(i)
	# time.sleep(1.0)
	pos = np.array(SimulationData[i+1])
	lines.set_data(pos[:].real,pos[:].imag)
	return lines,	i

def moving_average(a, n=4) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def accelometer(SimulationData_pos,SimulationData_strg):
	tsteps = len(SimulationData_pos)-1
	Xforce = []
	Xmom = np.array([np.sum(np.array(SimulationData_pos[i+1]).real*np.array(SimulationData_strg[i+1])) for i in range(tsteps)])

	Xforce = [(-(Xmom[i] - Xmom[i-1])/0.1) for i in range(1,len(Xmom))]
	Cd = np.array(Xforce)/(0.5*1*2)


	return Cd	

if __name__ == "__main__":
	body_list = []
	body_list.append(VPM.sld_bd(ctr_circ_vert(0j,1,50),1,0j))
	# body_list.append(VPM.sld_bd(ctr_circ_vert(3j,1,50),1,3j))
	# body_list.append(VPM.sld_bd(ctr_circ_vert(3+0j,1,50),1,3+0j))

	Vinf = 1+0j

	vort_pos = np.array([-2]*100) + 1j*np.linspace(-4,4,100)
	vort_list = [] #[VPM.vortex(1j,0,0.005)]
	# for i in range(len(vort_pos)):
	# 	#print(vort_pos[i])
	# 	vort_list.append(VPM.vortex(vort_pos[i],0.0,0.005))

	post,circt,t,vort_list = itg.simulate(body_list,Vinf,vort_list,0.002,2,0.1)
	
	# plotting_ParticlesPath(post,1)
	
	Cd = accelometer(post,circt)
	
# ploting Cd  	
	# plt.figure(3)
	# plt.plot(Cd,label = 'Cd for circular cylinder with for Re = 1000 ')
	# print(Cd)

# for vortex particles plots
	# plot_vortexpath(np.array(post[-1]),np.array(circt[-1]),np.array(ctr_circ_vert(0j,1,50)),1)

# for velocity field
	# plot_velfield(np.mgrid[0:2:60j,0:2:60j],post[-1],circt[-1],body_list[0],2)
	

# for animation
	# fig = plt.figure(1)
	# ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5),)
	# ax.set_aspect('equal', 'datalim')
	# pos = np.array(post[0])
	# lines, = ax.plot(pos[:].real,pos[:].imag,'o',ms=1.2)
	# anim = animation.FuncAnimation(fig, update_data, frames=len(t) - 1, interval=50,repeat=True,fargs=(post,lines))
	

	# plt.show()
	
	
	


		
