import numpy as np
import matplotlib.pyplot as plt

def sph_kernel_cubicsp(q,h,dim):
	sigma = np.array([2/(3*h),10/(7*np.pi*h**2),1/np.pi*h**3])
	if q <= 1:
		return sigma[dim-1]*(1 - (1.5*q**2)*(1 - 0.5*q))
	elif q<= 2:
		return 0.25*sigma[dim-1]*(2 - q)**3
	else:
		return 0

def sph_kernel_gaussian(q,h,dim):
	sigma = 1/((np.pi**0.5)*h)
	return (sigma**dim)*np.exp(-q**2)

def sph_kernel_cubicsp_grad(q,e,h,dim):
	sigma = np.array([2/(3*h),10/(7*np.pi*h**2),1/np.pi*h**3])
	temp = 0.0
	temp2 = 0.0
	if q <= 1:
		if q > 1e-4/h:
			temp = sigma[dim-1]*(-3*q + (9*q**2)/4)
			temp2 = sigma[dim-1]*(1 - (1.5*q**2)*(1 - 0.5*q))
	elif q <= 2:
			temp =  -0.25*sigma[dim-1]*3*((2 - q)**2)	
			temp2 = 0.25*sigma[dim-1]*((2 - q)**3)	
	return temp*e/h
	
def sph_kernel_gaussian_grad(q,e,h,dim):
	sigma = 1/((np.pi**0.5)*h)
	return -(sigma**dim)*np.exp(-q**2)*2*q*e/h




def sph_eval_fun_csp(x,fxi,xi,hdx,dim = 1):
	if np.shape(fxi) != np.shape(xi):
		print("shape err",np.shape(fxi),np.shape(xi))
		return 0
	else:
		pass	
	
	dx = abs(xi[0] - xi[1])	
	h = hdx*dx
	dv = dx
	fx = 0.0
	for j in range(len(xi)):
		q = abs(x - xi[j])/h
		# print(fxi[j],sph_kernel_cubicsp(q,h,dim),q)
		fx += fxi[j]*sph_kernel_cubicsp(q,h,dim)*dv
	return fx

def sph_eval_grad_csp(x,fxi,xi,hdx,dim = 1):
	if np.shape(fxi) != np.shape(xi):
		print("shape err",np.shape(fxi),np.shape(xi))
		return 0
	else:
		pass	
	
	dx = abs(xi[0] - xi[1])	
	h = hdx*dx
	dv = dx
	fx = 0.0
	for j in range(len(xi)):
		q = abs(x - xi[j])/h
		if q > 1e-4/h:
			e = (x - xi[j])/abs(x - xi[j])
		else:
			e = 0.0
		print(fxi[j],sph_kernel_cubicsp_grad(q,e,h,dim),e,q)		
		fx += fxi[j]*sph_kernel_cubicsp_grad(q,e,h,dim)*dv
	return fx
		

def sph_eval_fun_gau(x,fxi,xi,hdx,dim = 1):
	if np.shape(fxi) != np.shape(xi):
		print("shape err",np.shape(fxi),np.shape(xi))
		return 0
	else:
		pass	
	
	dx = abs(xi[0] - xi[1])	
	h = hdx*dx
	dv = dx
	fx = 0.0
	for j in range(len(xi)):
		q = abs(x - xi[j])/h
		fx += fxi[j]*sph_kernel_gaussian(q,h,dim)*dv
	return fx
		
def sph_eval_grad_gau(x,fxi,xi,hdx,dim = 1):
	if np.shape(fxi) != np.shape(xi):
		print("shape err",np.shape(fxi),np.shape(xi))
		return 0
	else:
		pass	
	
	dx = abs(xi[0] - xi[1])	
	h = hdx*dx
	dv = dx
	fx = 0.0
	for j in range(len(xi)):
		q = abs(x - xi[j])/h

		if q > 1e-10:
			e = (x - xi[j])/abs(x - xi[j])
		else:
			e = 0.0
		fx += fxi[j]*sph_kernel_gaussian_grad(q,e,h,dim)*dv		
	return fx
	 

			


