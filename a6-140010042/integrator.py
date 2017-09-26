import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from numba import jit
import pdb
import VPM as VPM
from numpy.random import multivariate_normal as nmdist


@jit
def vortex_v(z,strg,delta):
	if abs(z) > 0 and abs(z)/delta <= 1.0: 
		return -1j * strg * (abs(z)/delta) / (2 * np.pi * z) 
	else:
		if abs(z) > 0 and abs(z)/delta > 1.0:
			return -1j * strg / (2 * np.pi * z)
		
		else:
			return 0j	


@jit
def vel_vort(z,pos,strg,delta):
	vel = np.zeros_like(z,dtype = complex)	
	for i in range(len(z)):
		for j in range(len(pos)):
			vel[i] += vortex_v(z[i] - pos[j],strg[j],delta)
	return np.conj(vel)

def rk2_step(sld_bodies,free_stream,vort_elems,dt):
	
			
	k1 = np.zeros((len(vort_elems),1),dtype = complex)
	pos,strg,delta = np.array([vort_elems[i].pos for i in range(len(vort_elems))]),np.array([vort_elems[i].strg for i in range(len(vort_elems))]),vort_elems[0].delta
	poscp = [vort_elems[i].poscp for i in range(len(vort_elems))]
	
	k1[:] += [VPM.vel_body(sld_bodies,vort_elems[i].pos) for i in range(len(vort_elems))]
	
	for i in range(len(vort_elems)):
		k1[i] += free_stream 
	k1[:,0] += vel_vort(pos,pos,strg,delta)
	
	for i in range(len(vort_elems)):
		vort_elems[i].pos += 0.5*k1[i][0]*dt
		
	for body in sld_bodies:
		body.move_body(0.5*dt)		
	VPM.depenetrator(sld_bodies,vort_elems)

	A = VPM.get_coefficient_mat(sld_bodies)
	b = VPM.get_rhs(sld_bodies,free_stream,vort_elems)
	gamma = VPM.solve_gamma(sld_bodies,A,b)
		
	k2 = np.zeros((len(vort_elems),1),dtype = complex)
	pos,strg,delta = np.array([vort_elems[i].pos for i in range(len(vort_elems))]),np.array([vort_elems[i].strg for i in range(len(vort_elems))]),vort_elems[0].delta
	poscp = [vort_elems[i].poscp for i in range(len(vort_elems))]
	
	k2[:] += [VPM.vel_body(sld_bodies,vort_elems[i].pos)  for i in range(len(vort_elems))]
	
	for i in range(len(vort_elems)):
		k2[i] += free_stream 
	k2[:,0] += vel_vort(pos,pos,strg,delta) 
	
	for i in range(len(vort_elems)):
		vort_elems[i].poscp += k2[i][0]*dt
		vort_elems[i].pos = cp.copy(vort_elems[i].poscp)
	
	for body in sld_bodies:
		body.move_body(0.5*dt)
	VPM.depenetrator(sld_bodies,vort_elems)


def diffusion_step(vort_list,nu,dt):
	for j in range(len(vort_list)):
		if abs(vort_list[j].strg) > 0.01:
			n = int(abs(vort_list[j].strg)/0.01)
	
			vort_list[j].strg = np.sign(vort_list[j].strg)*0.01 # + zeta[0]
			temp = [vort_list.append(VPM.vortex(vort_list[j].pos ,vort_list[j].strg,vort_list[j].delta)) for i in range(n-1)]

	zetax, zetay = nmdist([0, 0],[[2*nu*dt, 0],[0, 2*nu*dt]],len(vort_list)).T
	zeta = zetax + 1j*zetay		
	for i in range(len(vort_list)):	
		vort_list[i].pos += zeta[i] 	


def simulate(sld_bodies,free_stream,vort_elems,nu,final_time,time_step):
	t = np.linspace(0,final_time,(final_time/time_step) + 1)
	position_t = []
	position_t.append([cp.copy(vort_elems[j].pos) for j in range(len(vort_elems))])
	circulation_t = []
	circulation_t.append([cp.copy(vort_elems[j].strg) for j in range(len(vort_elems))])
	
	vort_elems.append(VPM.vortex(0j,0.0,0.1))	
	
	A = VPM.get_coefficient_mat(sld_bodies)
	b = VPM.get_rhs(sld_bodies,free_stream,vort_elems)
	gamma = cp.copy(VPM.solve_gamma(sld_bodies,A,b))
	
	new_blobs = VPM.slip_nullify(sld_bodies,vort_elems)
	
	rk2_step(sld_bodies,free_stream,vort_elems,time_step)
		
	len(vort_elems)
	pos,strg,delta = np.array([vort_elems[i].pos for i in range(len(vort_elems))]),np.array([vort_elems[i].strg for i in range(len(vort_elems))]),vort_elems[0].delta
				
	vort_elems.pop(-1)
	
	for i in range(len(t)):
		# print(i)
		
		[vort_elems.append(new_blobs[i]) for i in range(len(new_blobs))]	
	
		diffusion_step(vort_elems,nu,time_step)
	
		VPM.depenetrator(sld_bodies,vort_elems)
	
		position_t.append([cp.copy(vort_elems[j].pos) for j in range(len(vort_elems))])
		circulation_t.append([cp.copy(vort_elems[j].strg) for j in range(len(vort_elems))])
			
		A = VPM.get_coefficient_mat(sld_bodies)
		b = VPM.get_rhs(sld_bodies,free_stream,vort_elems)
		gamma = VPM.solve_gamma(sld_bodies,A,b)
		
		new_blobs = VPM.slip_nullify(sld_bodies,vort_elems)
	
		rk2_step(sld_bodies,free_stream,vort_elems,time_step)
	
		len(vort_elems)
		pos,strg,delta = np.array([vort_elems[i].pos for i in range(len(vort_elems))]),np.array([vort_elems[i].strg for i in range(len(vort_elems))]),vort_elems[0].delta
	
	return position_t, circulation_t, t, vort_elems

