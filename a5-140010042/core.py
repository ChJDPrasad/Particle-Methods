import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.random import multivariate_normal as nmdist
import copy as cp
from scipy.integrate import dblquad

from mpl_toolkits.mplot3d import Axes3D


class vort_part():
	def __init__(self,strg,pos,div):
		self.pos = pos
		self.strg = strg
		self.no_chil = div
		self.list_chil = []	

	def split(self):
		if self.strg < 1e-3:
			self.list_chil.append(vort_part(self.strg,self.pos,self.no_chil))
		else:
			[self.list_chil.append(vort_part(self.strg/self.no_chil,self.pos,self.no_chil)) for i in range(self.no_chil)]
			

	def get_nxtgen(self):
		self.split()
		return self.list_chil	

def get_pos(list_part):
	pos = np.array([cp.copy(part.pos) for part in list_part],dtype = complex)
	return pos



def diff_step(list_part,nu,dt):
	
	nxtgen = []
	for part in list_part:
		nxtgen += part.get_nxtgen()

	zetax, zetay = nmdist([0, 0],[[2*nu*dt, 0],[0, 2*nu*dt]],len(nxtgen)).T
	zeta = zetax + 1j*zetay
	for i in range(len(nxtgen)):
		nxtgen[i].pos += zeta[i]
	return nxtgen


def update_data(i,nxtgen,lines,nu,dt):
	pos = get_pos(nxtgen[i+1])
	lines.set_data(pos[:].real,pos[:].imag)
	return lines,

def simulate(list_part,t,nu,dt,anim):
	
	nt = int(t/dt)
	pos = get_pos(list_part)
	nxtgen = []
	nxtgen.append(list_part)
	for i in range(nt-1):
		nxtgen.append(diff_step(nxtgen[i],nu,dt))
	
	if anim:
		fig = plt.figure()
		ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
		lines, = ax.plot(pos[:].real,pos[:].imag,'o')
		anim = animation.FuncAnimation(fig, update_data, frames=nt - 1, interval=dt,repeat=False,fargs=(nxtgen,lines,nu,dt))
		plt.show()
	return nxtgen[-1]

def findindex(part,dx,dy,Nx,Ny):
	
	
	i,delx = divmod(part.pos.real,dx)
	i += round(delx/dx) + 0.5*(Nx - 1)
	x = delx/dx - round(delx/dx)
	j,dely = divmod(part.pos.imag,dy)
	j += round(dely/dy) + 0.5*(Ny - 1)
	y = dely/dy - round(dely/dy)


	return x,y,int(i),int(j)



def cal_contri(Gamma,part,dx,dy,Nx,Ny):
	x,y,i,j = findindex(part,dx,dy,Nx,Ny)

	temp2 = part.strg*(1 - x**2)
	temp3 = part.strg*0.5*x*(1 + x)
	temp1 = part.strg*0.5*x*(- 1 + x)

	Gamma[i,j] += temp2*(1 - y**2)
	Gamma[i,j+1] += temp2*0.5*y*(1 + y)
	Gamma[i,j-1] += temp2*0.5*y*(- 1 + y)

	Gamma[i+1,j] += temp3*(1 - y**2)
	Gamma[i+1,j+1] += temp3*0.5*y*(1 + y)
	Gamma[i+1,j-1] += temp3*0.5*y*(- 1 + y)

	Gamma[i-1,j] += temp1*(1 - y**2)
	Gamma[i-1,j+1] += temp1*0.5*y*(1 + y)
	Gamma[i-1,j-1] += temp1*0.5*y*(- 1 + y)



def exact_sol(z, mu, t, dx, dy, z0=0j, g0=1.):
	 

	def gamma_dist_xy(x, y):
		return np.exp(-(x ** 2 + y ** 2) / (4. * mu * t)) / (4 * np.pi * mu * t)

	@np.vectorize
	def exact_gamma(z):
		x, y = z.real, z.imag
		return dblquad(gamma_dist_xy,x - 0.5 * dx,x + 0.5 * dx,lambda x: y - 0.5 * dy,lambda x: y + 0.5 * dy)[0]

	return exact_gamma(z)

def remesh(finalgen,dx,dy,exact_sol):
	X,Y = np.mgrid[-2.5:2.5:dx,-2.5:2.5:dy]
	Nx = len(np.arange(-2.5,2.5,dx))
	Ny = len(np.arange(-2.5,2.5,dy))
	
	Z = X + 1j*Y
	Gamma = np.zeros_like(Z,dtype = float)
	for part in finalgen:
		cal_contri(Gamma,part,dx,dy,Nx,Ny)
	
	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X,Y,Gamma)
	plt.savefig('vorticity_distribution_for_' + str(len(finalgen)) + '_particles')

	Gamma_exact = exact_sol(Z,0.1,1,dx,dy)
	fig = plt.figure(2)
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X,Y,Gamma_exact)
	plt.savefig('Exact_solution_of_diffusion_for_1_sec')
	
	return sum(sum(abs(Gamma - Gamma_exact))),len(finalgen)

def Q1(npart,anim = False):
	ini_part = vort_part(1,0j,npart)
	list_part= []
	list_part.append(ini_part)
	
	finalgen = simulate(list_part,1,0.1,0.025,anim)
	return remesh(finalgen,0.04,0.04,exact_sol)	


if __name__ == "__main__":
	npart = [40,50,60,70,80,20,90,100,110,120,130,140,150,30]
	err = []
	finalpart = []
	for i in range(len(npart)):
		temp = Q1(npart[i])
		err.append(temp[0])
		finalpart.append(temp[1])
		print('simulation completed for ' + str(temp[1]) + ' particles')
		#print(err)	
	plt.figure(3)
	plt.plot(finalpart,err,'-o')
	plt.savefig('Error_for_different_no_of_particles')	