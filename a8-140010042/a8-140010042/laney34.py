import numpy as np
import matplotlib.pyplot as plt

def euler(x,u,dt):
	return np.array([x[i] + u[i]*dt for i in range(len(x))]) 

def sph_kernel_cubicsp(q,h,dim):
	sigma = np.array([2/(3*h),10/(7*np.pi*h**2),1/np.pi*h**3])
	if q <= 1:
		return sigma[dim-1]*(1 - (1.5*q**2)*(1 - 0.5*q))
	elif q<= 2:
		return 0.25*sigma[dim-1]*(2 - q)**3
	else:
		return 0

def sph_eval_fun_csp(x,fxi,xi,deni,mass,h,dim = 1):
	dv = mass/deni
	fx = 0.0
	for j in range(len(xi)):
		q = abs(x - xi[j])/h
		fx += fxi[j]*sph_kernel_cubicsp(q,h,dim)*dv[j]
	return fx

class domain():
	def __init__(self,IC,Nx,bounds,int_method,hdx,eps,periodic = True):
		self.bounds = bounds
		self.IC = IC
		self.x = np.linspace(bounds[0],bounds[1],Nx,endpoint = True)
		self.phi = np.array(list(map(IC,self.x)))
		self.mass = 1.0
		self.den = [self.mass/(self.x[2]- self.x[1])]*len(self.x)
		self.vel = [1]*len(self.x)
		self.h = hdx*(self.x[2]- self.x[1])
		self.eps = eps
		self.periodic = periodic
		
	def cal_den(self):
		for i in range(len(self.x_ext)):
			self.den_ext[i] = np.sum([self.mass*sph_kernel_cubicsp(abs(self.x_ext[i] - self.x_ext[j])/self.h,self.h,1) for j in range(len(self.x_ext))])

			   
	def extend_dom(self,no_part):
		self.ext = True
		self.no_part = no_part
		self.x_ext = np.concatenate((np.concatenate((self.x[len(self.x) - no_part:-1] - (self.x[-1] - self.x[0]),self.x)),self.x[1:no_part] + (self.x[-1] - self.x[0]) ))	
		# self.phi_ext = np.array(list(map(self.IC,self.x_ext)))
		self.phi_ext = np.concatenate((np.concatenate((self.phi[len(self.x) - no_part:-1],self.phi)),self.phi[1:no_part]))	
		self.den_ext = np.zeros_like(self.x_ext)
		self.den_ext[:-1] = np.array([self.mass/(self.x_ext[1:]- self.x_ext[:-1])])
		self.vel = [1]*len(self.x_ext)
		
		
	def get_vel(self):
			for i in range(len(self.x_ext)):
				self.vel[i] = self.phi_ext[i]	
				for j in range(len(self.x_ext)):
					self.vel[i] += self.eps*(self.mass/(0.5*(self.den_ext[i] + self.den_ext[j]))*(self.phi_ext[j] - self.phi_ext[i])*(sph_kernel_cubicsp(abs(self.x_ext[i] - self.x_ext[j])/self.h,self.h,1)))
					
			return self.vel

		
	# def sort(self):
	# 	return np.argsort(self.x)

	def periodic_chk(self):
		if self.periodic:
			mask = (self.x_ext < self.bounds[1] + 1e-5) & (self.x_ext > self.bounds[0] - 1e-5)
			self.x = self.x_ext[mask]
			self.phi = self.phi_ext[mask]
			self.den = self.den_ext[mask]


	def simulate(self,method,dt):
		self.extend_dom(self.no_part)
		self.cal_den()
		self.vel = self.get_vel()
		self.x_ext = method(self.x_ext,self.vel,dt)
		self.cal_den()
		self.periodic_chk()
		


				
		
def q3(Nx,eps):
	IC = lambda x: 1 if abs(x) < 1/3 else 0
	dom = domain(IC,Nx,[-1,1],euler,3,eps)
	dom.extend_dom(5)
	for i in range(int(12)):
		dom.simulate(euler,0.05)
	return dom


def q4(Nx,eps):
	IC = lambda x: 1 if abs(x) < 1/3 else -1
	dom = domain(IC,Nx,[-1,1],euler,3,eps)
	dom.extend_dom(5)
	for i in range(int(6)):
		dom.simulate(euler,0.05)
	return dom


if __name__ == "__main__":
	dom = q3(40,0.5)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(1)
	plt.plot(x,fx,label = 'Initial condition ')
	
	plt.plot(dom.x,dom.phi,'o',label = 'Computed soln at t = 0.6s with 40 points for e = 0.5')
	plt.title('Simulating ut + u ux = 0 for IC = 1 if |x| < 1/3 else 0')
	plt.savefig('q3_40_05.png')
	
	dom = q3(100,0.5)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(2)
	plt.plot(x,fx,label = 'Initial condition ')
	plt.plot(dom.x,dom.phi,'o',label = 'Computed soln at t = 0.6s with 100 points for e = 0.5')
	plt.title('Simulating ut + u ux = 0 for IC = 1 if |x| < 1/3 else 0')
	plt.savefig('q3_100_05.png')
	
	dom = q4(40,0.5)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(3)
	plt.plot(x,fx,label = 'Initial condition ')
	
	plt.plot(dom.x,dom.phi,'o',label = 'Computed soln at t = 0.3s with 40 points e = 0.5')
	plt.title('Simulating ut + u ux = 0 for IC = 1 if |x| < 1/3 else -1')
	plt.savefig('q4_40_05.png')
	
	dom = q4(100,0.5)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(4)
	plt.plot(x,fx,label = 'Initial condition ')
	plt.plot(dom.x,dom.phi,'o',label = 'Computed soln at t = 0.3s with 100 points e = 0.5')
	plt.title('Simulating ut + u ux = 0 for IC = 1 if |x| < 1/3 else -1')
	plt.savefig('q4_100_05.png')
	
	dom = q3(40,1.0)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(5)
	plt.plot(x,fx,label = 'Initial condition ')
	
	plt.plot(dom.x,dom.phi,'o',label = 'Computed soln at t = 0.6s with 40 points for e = 1.0')
	plt.title('Simulating ut + u ux = 0 for IC = 1 if |x| < 1/3 else 0')
	plt.savefig('q3_40_1.png')
	
	dom = q3(100,1.0)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(6)
	plt.plot(x,fx,label = 'Initial condition ')
	plt.plot(dom.x,dom.phi,'o',label = 'Computed soln at t = 0.6s with 100 points for e = 1.0')
	plt.title('Simulating ut + u ux = 0 for IC = 1 if |x| < 1/3 else 0')
	plt.savefig('q3_100_1.png')
	
	dom = q4(40,1.0)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(7)
	plt.plot(x,fx,label = 'Initial condition ')
	
	plt.plot(dom.x,dom.phi,'o',label = 'Computed soln at t = 0.3s with 40 points e = 1.0')
	plt.title('Simulating ut + u ux = 0 for IC = 1 if |x| < 1/3 else -1')
	plt.savefig('q4_40_1.png')

	dom = q4(100,1.0)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(8)
	plt.plot(x,fx,label = 'Initial condition ')
	plt.plot(dom.x,dom.phi,'o',label = 'Computed soln at t = 0.3s with 100 points e = 1.0')
	plt.title('Simulating ut + u ux = 0 for IC = 1 if |x| < 1/3 else -1')
	plt.savefig('q4_100_1.png')
	