import numpy as np
import copy as cp
from scipy import linalg
import matplotlib.pyplot as plt
from numba import jit
import integrator as itg

@jit
def dot(x1,x2):
	return x1.real*x2.real + x1.imag*x2.imag



@jit
def coeff_gamma_const(vert,tan,z,i):
	coeff = np.conj((1j/(2*np.pi))*(np.log((z - vert[i+2])/(z - vert[i+1])))) 	
	coeff = coeff*(tan[i])		
	return coeff

@jit
def vel_panel_const(vert,tan,strg,z,i):
	coeff = coeff_gamma_const(vert,tan,z,i)
	vel = strg[i]*coeff
	return vel

@jit
def coeff_gamma_lin(vert,tan,z,i):
	i = i+1
	coeff1 = (1j/(2*np.pi))*(1 + ((z - vert[i-1])/(vert[i] - vert[i-1]))*np.log((z - vert[i])/(z - vert[i-1])))
	coeff2 = -(1j/(2*np.pi))*(1 + ((z - vert[i])/(vert[i+1] - vert[i]))*np.log((z - vert[i+1])/(z - vert[i])))
	coeff3 = (1j/(2*np.pi))*np.log((z - vert[i+1])/(z - vert[i]))
	coeff = np.conj(coeff1)*tan[i-2] + np.conj(coeff2 + coeff3)*tan[i-1]
	return coeff

@jit
def vel_panel_lin(vert,tan,strg,z,i):
	m = (strg[i+1]-strg[i])/(vert[i+2] - vert[i+1])
	c = strg[i]		
	vel1 = c*(np.log((z - vert[i+2])/(z - vert[i+1]))) 
	vel2 = m*(vert[i+2] - vert[i+1])
	vel3 = m*(z - vert[i+1])*(np.log((z - vert[i+2])/(z - vert[i+1])))
	vel = np.conj((1j/(2*np.pi))*(vel1 + vel2 + vel3))		
	vel = vel*(tan[i])		
	return vel
	

def vel_body(body_list,z):
	vel = 0+0j
	for body in body_list:
		vert = body.get_vert()
		tan = body.get_utan()
		strg = body.get_strg()
		if body.lin == True:
			method = vel_panel_lin
		else:		
			method = vel_panel_const
		for i in range(body.npanel):
			vel += method(vert,tan,strg,z,i)
	return vel


def get_coefficient_mat(body_list):
	n = 0
	ctrl_pnts = np.array([])
	norm = np.array([])
		
	for body in body_list:
		n += body.npanel
		ctrl_pnts = np.append(ctrl_pnts,body.get_ctrl_pnts())
		norm = np.append(norm,body.get_unorm())
	A = np.zeros((n,n))
	for i in range(n):
		pntr = 0
		for body in body_list:
			vert = body.get_vert()
			tan = body.get_utan()
			if body.lin == True:
				method = coeff_gamma_lin
			else:		
				method = coeff_gamma_const
			
			for j in range(body.npanel):
				A[:,j+pntr] = dot(coeff_gamma_lin(vert,tan,ctrl_pnts[:],j),norm[:]) 
			pntr += body.npanel		
	
	pntr = 0			
	for b in range(len(body_list)):		
		panel_len = body_list[b].get_pnl_len()	
		A = np.append(A,np.zeros((1,n)))
		A = np.resize(A,(n+b+1,n))
		if body_list[b].lin == False:
			method = coeff_gamma_const
			A[-1,pntr:pntr + body_list[b].npanel] = panel_len[:]
		else:
			method = coeff_gamma_lin
			A[-1,pntr:pntr + body_list[b].npanel] = 0.5*(panel_len[:]+np.insert(panel_len,0,panel_len[-1])[:-1])  
		pntr = body_list[b].npanel		
				
	
	return A

	
def get_rhs(body_list,free_stream,vort_elems):
	n = 0
	ctrl_pnts = np.array([])
	norm = np.array([])
	for body in body_list:
		n += body.npanel
		ctrl_pnts = np.append(ctrl_pnts,body.get_ctrl_pnts())
		norm = np.append(norm,body.get_unorm())
	b = np.zeros((n,1))
	pos,strg,delta = np.array([vort_elems[i].pos for i in range(len(vort_elems))]),np.array([vort_elems[i].strg for i in range(len(vort_elems))]),vort_elems[0].delta
		
	
	b[:,0] += dot(-itg.vel_vort(ctrl_pnts,pos,strg,delta),norm)
	for i in range(n):
		b[i] += dot(-free_stream,norm[i])		
		x = 1
		for body in body_list:
			
			b[i] += dot(body.V_body,norm[i])
			b = np.append(b,[0])
			b = np.resize(b,(n + x,1))
			x += 1
	
	return b		

def solve_gamma(body_list,A,b):
	A_star = cp.copy(A)
	b_star = cp.copy(b)
	
	A_star[:,0] = 0.5*A_star[:,0]
	A_star = np.insert(A_star, np.shape(A_star)[1], values=A_star[:,0], axis=1)
	pntr = 0
	gamma = linalg.lstsq(A_star,b_star)[0]
	for body in body_list:
		
		body.strg[:-1] = gamma[pntr:pntr + body.npanel]
		body.strg[-1] = body.strg[0]
		pntr = body.npanel		
	err = cp.copy(sum(abs(np.dot(A_star,gamma)-b_star)))
	
	return gamma

def slip_nullify(body_list,vort_list):
	for body in body_list:
		ctrl_pnts = body.get_ctrl_pnts()
		norm = body.get_unorm()
		gamma = body.get_strg()
		pnl = body.get_pnl_len()
		vort_list2 = []
		# print(body.npanel)	
		for i in range(body.npanel):
			vort_list2.append(vortex((abs(ctrl_pnts[i]) + pnl[i]/np.pi)*norm[i],0.5*(gamma[i][0] + gamma[i+1][0])*pnl[i],pnl[i]/np.pi))
	return vort_list2
	
class sld_bd:
	def __init__(self,vert,rad,ctr,linear = True):
		self.lin = linear
		self.rad = rad
		self.ctr = ctr
		self.list_vert = cp.copy(vert) 
		self.npanel = len(self.list_vert)		
		self.list_vert.append(vert[0])
		#print(self.list_vert)
		self.V_body = 0 + 0j		
		self.control_points = []
		self.unit_tangent_v = []
		self.unit_normal_v = []
		self.panel_len = []
		self.strg = [[1]]*(len(self.list_vert))
		for i in range(len(self.list_vert)-1):
			#print(i)
			self.control_points.append(0.5*(self.list_vert[i] + self.list_vert[i+1]))
			self.panel_len.append(abs(self.list_vert[i+1] - self.list_vert[i]))
			self.unit_tangent_v.append((self.list_vert[i+1] - self.list_vert[i])/self.panel_len[i])
			self.unit_normal_v.append(1j*self.unit_tangent_v[i])

	def get_vert(self):
		vert = np.array(self.list_vert)
		vert = np.insert(vert,0,self.list_vert[-2])
		return vert

	def get_utan(self):
		return np.array(self.unit_tangent_v)
		
	def get_unorm(self):
		return np.array(self.unit_normal_v)
		
	def get_pnl_len(self):
		return np.array(self.panel_len)
		
	def get_ctrl_pnts(self):
		return np.array(self.control_points)
		
	def get_strg(self):
		return np.array(self.strg)
		
	def move_body(self,dt):
		for i in range(len(self.list_vert)):
			self.list_vert[i] += self.V_body*dt
		for i in range(len(self.control_points)):		
			self.control_points[i] += self.V_body*dt

class vortex:
	def __init__(self,pos,strg,delta):
		self.pos = cp.copy(pos)
		self.poscp = cp.copy(pos)
		self.strg = cp.copy(strg)
		self.delta = cp.copy(delta)


def depenetrator(body_list,vort_list):
	for vort in vort_list:
		for body in body_list:
			# print(abs(vort.pos - body.ctr))
			if abs(vort.pos - body.ctr) < body.rad:
				vort.pos = 2*body.rad*(vort.pos - body.ctr)/abs(vort.pos - body.ctr) - (vort.pos - body.ctr) + body.ctr  
				# print(vort.pos)
			else:
				pass

def methodofimages(strg,z,cy_rad):
	vel = itg.vel_vort(z,(cy_rad**2)/np.conj(z),-strg,itg.kernel_d(0)) + itg.vel_vort(z,0j,strg,itg.kernel_d(0))
	return vel


	

if __name__ == "__main__":
	pass
