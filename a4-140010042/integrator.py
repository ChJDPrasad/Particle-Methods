import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from numba import jit
import pdb

@jit
def vortex_v(z):
	return -1j / (2 * np.pi * z)


def kernel_d(delta):
	@jit	
	def kernel(z):
		return (np.abs(z)**2/(np.abs(z)**2 + delta**2))
	return kernel

@jit
def vel_vort(z,pos,strg,kernel):
	vel = 0j	
	vel = vortex_v(z -pos) * strg * kernel(z - pos)
	return np.conj(vel)
        
def rk2_step(sld_bodies,free_stream,vort_elems,dt):
	
	for body in sld_bodies:		
		A = body.get_coefficient_mat()
		b = body.get_rhs(free_stream,vort_elems)
		gamma = body.solve_gamma(A,b)	
			
	k1 = np.zeros((len(vort_elems),1),dtype = complex)

	for i in range(len(vort_elems)):
		#print(i)
		for j in range(len(sld_bodies)):
			#print(j)
			k1[i] += sld_bodies[j].vel_body(vort_elems[i].pos)
			#print(k1[i]) 
		vort_elems[i].pos += 0.5*k1[i]*dt
 
	for body in sld_bodies:
		body.move_body(0.5*dt)		
		A = body.get_coefficient_mat()
		b = body.get_rhs(free_stream,vort_elems)
		gamma = body.solve_gamma(A,b)
		
	k2 = np.zeros((len(vort_elems),1),dtype = complex)
	for i in range(len(vort_elems)):
		for j in range(len(sld_bodies)):
			k2[i] += sld_bodies[j].vel_body(vort_elems[i].pos) 
			#print(k2[i])		
		vort_elems[i].poscp += k2[i]*dt
		vort_elems[i].pos = cp.copy(vort_elems[i].poscp)
	
	for body in sld_bodies:
		body.move_body(0.5*dt)
	


def simulate(sld_bodies,free_stream,vort_elems,final_time,time_step):
	t = np.linspace(0,final_time,(final_time/time_step) + 1)
	position_t = np.zeros((int(final_time/time_step) + 1,len(vort_elems)),dtype=complex)
	#print(vort_elems[0].pos)	
	for j in range(len(vort_elems)):
		position_t[0][j] = cp.copy(vort_elems[j].pos)
	for i in range(len(t)-1):
		
		rk2_step(sld_bodies,free_stream,vort_elems,time_step)
		for j in range(len(vort_elems)):
			position_t[i+1][j] = cp.copy(vort_elems[j].pos)
	
	return position_t, t

