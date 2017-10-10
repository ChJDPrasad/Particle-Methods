import numpy as np
import matplotlib.pyplot as plt

def euler(x,u,dt):
	return np.array([x[i] + u[i]*dt for i in range(len(x))]) 

class domain():
	def __init__(self,IC,Nx,bounds,int_method,periodic = True):
		self.bounds = bounds
		self.IC = IC
		self.x = np.linspace(bounds[0],bounds[1],Nx,endpoint = True)
		self.phi = np.array(list(map(IC,self.x)))
		self.mass = 1.0
		self.den = [self.mass/(self.x[2]- self.x[1])]*len(self.x)
		self.vel = [1]*len(self.x)
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
		# self.den_ext = np.zeros_like(self.x_ext)
		# self.den_ext = [self.den[0]]*len(self.x_ext)
		self.vel = [1]*len(self.x_ext)
		
		# print(self.x_ext,self.vel)

	def get_vel(self):
		return [self.vel[0]]*len(self.x)	

	def sort(self):
		return np.argsort(self.x)

	def periodic_chk(self):
		if self.periodic:
			for i in range(len(self.x)):
				if self.x[i] > self.bounds[1] + 1e-5:
					self.x[i] -= (self.bounds[1] - self.bounds[0])
				elif self.x[i] < self.bounds[0] - 1e-5:		
					self.x[i] += (self.bounds[1] - self.bounds[0])
				else:
					pass		
		if (abs(self.x[-1] - self.bounds[1]) < 1e-5) & (abs(self.x[0] - self.bounds[1]) < 1e-5):
			self.x[-1] -= (self.bounds[1] - self.bounds[0])
		elif (abs(self.x[-1] - self.bounds[0]) < 1e-5) & (abs(self.x[0] - self.bounds[0]) < 1e-5):
			self.x[0] += (self.bounds[1] - self.bounds[0])
		else:
			pass	
	def simulate(self,method,dt):
		# self.extend_dom(self.no_part)
		self.vel = self.get_vel()
		self.x = method(self.x,self.vel,dt)
		self.periodic_chk()
		


				
		


def q1(Nx):
	IC = lambda x: -np.sin(np.pi*x)
	dom = domain(IC,Nx,[-1,1],euler)
	for i in range(int(40/0.05)):
		dom.simulate(euler,0.05)
	return dom

def q2(Nx):
	IC = lambda x: 1 if abs(x) < 1/3 else 0
	dom = domain(IC,Nx,[-1,1],euler)
	for i in range(int(40/0.05)):
		dom.simulate(euler,0.05)
	return dom

 
if __name__== "__main__":	
	dom = q1(40)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(1)
	plt.plot(x,fx,label = 'Exact soln at t = 40s')
	idx = dom.sort()
	plt.plot(dom.x[idx],dom.phi[idx],'o',label = 'Computed soln at t = 40s with 40 points')
	plt.title('Simulating ut + ux = 0 for IC = -sin(pi x)')
	plt.savefig('q1_40.png')
	
	dom = q1(100)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(2)
	plt.plot(x,fx,label = 'Exact soln at t = 40s')
	idx = dom.sort()
	plt.plot(dom.x[idx],dom.phi[idx],'o',label = 'Computed soln at t = 40s with 100 points')
	plt.title('Simulating ut + ux = 0 for IC = -sin(pi x)')
	plt.savefig('q1_100.png')
	
	dom = q2(40)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(3)
	plt.plot(x,fx,label = 'Exact soln at t = 40s')
	idx = dom.sort()
	plt.plot(dom.x[idx],dom.phi[idx],'o',label = 'Computed soln at t = 40s with 40 points')
	plt.title('Simulating ut + ux = 0 for IC = 1 if |x| < 1/3 else 0')
	plt.savefig('q2_40.png')

	dom = q2(100)
	x = np.linspace(-1,1,200,endpoint = True)
	fx = np.array(list(map(dom.IC,x)))
	plt.figure(4)
	plt.plot(x,fx,label = 'Exact soln at t = 40s')
	idx = dom.sort()
	plt.plot(dom.x[idx],dom.phi[idx],'o',label = 'Computed soln at t = 40s with 100 points')
	plt.title('Simulating ut + ux = 0 for IC = 1 if |x| < 1/3 else 0')
	plt.savefig('q2_100.png')

	