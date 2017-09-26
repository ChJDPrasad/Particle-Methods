import numpy as np
import copy as cp
from scipy import linalg
import matplotlib.pyplot as plt
from numba import jit
import integrator as itg


def dot(x1,x2):
	return x1.real*x2.real + x1.imag*x2.imag


def lstsqn(a,b):
	return np.dot(linalg.inv(np.dot(a.T,a)),np.dot(a.T,b))

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
	
class sld_bd:
	def __init__(self,vert,linear = False):
		self.lin = linear
		self.list_vert = cp.copy(vert) 
		self.npanel = len(self.list_vert)		
		self.list_vert.append(vert[0])
		#print(self.list_vert)
		self.V_body = 0 + 0j		
		self.control_points = []
		self.unit_tangent_v = []
		self.unit_normal_v = []
		self.panel_len = []
		self.strg = [1]*(len(self.list_vert)+1)
		self.coeff_mat = np.zeros((self.npanel,self.npanel-1),dtype = complex)
		self.rhs = np.zeros((len(self.list_vert),1))
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
		

	def vel_body(self,z):
		vel = 0+0j
		vert = self.get_vert()
		tan = self.get_utan()
		strg = self.get_strg()
		if self.lin == True:
			method = vel_panel_lin
		else:		
			method = vel_panel_const
		for i in range(self.npanel):
			vel += method(vert,tan,strg,z,i)
		return vel

	
	def get_coefficient_mat(self):
		n = self.npanel	
		A = np.zeros((n+1,n),dtype = complex)
		vert = self.get_vert()
		tan = self.get_utan()
		norm = self.get_unorm()
		panel_len = self.get_pnl_len()	
		ctrl_pnts = self.get_ctrl_pnts()		
		if self.lin == False:
			method = coeff_gamma_const
			A[n,:] = panel_len[:]
		else:
			method = coeff_gamma_lin
			A[n,:] = 0.5*(panel_len[:]+np.insert(panel_len,0,panel_len[-1])[:-1])  
		for j in range(n):
			A[:-1,j] = dot(method(vert,tan,ctrl_pnts[:],j),norm[:])
		
		#self.coeff_mat = cp.copy(A)	
		return A
		
	def get_rhs(self,free_stream,vort_elems):
		b = np.zeros((len(self.list_vert),1))
		for i in range(len(self.list_vert)-1):		
			b[i] += dot(-free_stream + self.V_body ,self.unit_normal_v[i])
			for elem in vort_elems:
				b[i] += dot(-itg.vel_vort(self.control_points[i],elem.pos,elem.strg,elem.kernel),self.unit_normal_v[i]) 
		self.rhs = cp.copy(b)		
		return b

	def solve_gamma(self,A,b,exclude = -1):
		if exclude == -1:
			A_star = cp.copy(A)
			b_star = cp.copy(b)
		else:
			A_star = np.delete(A,exclude,0)
			b_star = np.delete(b,exclude,0)
		#print(np.shape(A_star),np.shape(b_star))
		self.strg[0:-1] = linalg.lstsq(A_star,b_star)[0]
		self.strg[-1] = self.strg[0]		
		return self.strg

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
		self.vel_field = itg.vel_vort
		self.kernel = itg.kernel_d(delta)




def methodofimages(strg,z,cy_rad):
	vel = itg.vel_vort(z,(cy_rad**2)/np.conj(z),-strg,itg.kernel_d(0)) + itg.vel_vort(z,0j,strg,itg.kernel_d(0))
	return vel


	

if __name__ == "__main__":
	pass
