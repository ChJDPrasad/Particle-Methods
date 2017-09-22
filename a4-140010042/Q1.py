from VPM import *
import integrator as itg
def Q1(linear = True):
	plt.figure()
	for nop in np.linspace(20,100,num=3,endpoint = True):
		cy_rad = 1.0  										#	cylinder of radius 
		V_inf = 1.0 + 0.0j
		npanel = nop
		thetas = np.linspace(0,2*np.pi,num=npanel,endpoint = True)
		vert = lambda th: -cy_rad*np.cos(th) + 1j*cy_rad*np.sin(th)
		vert = list(map(vert,thetas))
		#print(vert)
	
		vort_elems = []
		vort_elems.append(vortex(0,0,0))		

		body = sld_bd(vert,linear)
		A = body.get_coefficient_mat()
		b = body.get_rhs(V_inf,vort_elems)
		gamma = body.solve_gamma(A,b,exclude = -1)
		#print(gamma[0:-1])	
		#print(np.sum(gamma[0:-1]))
		#print(gamma)	
		error = np.dot(A,gamma[0:-1]) - b
		error = error*error
		error = error.sum() 
		#print(error)	
		#X, Y= np.mgrid[-5:5:100j,-5:5:100j]
		#Z = X + 1j*Y
		#print(Z)
	 	
	
	
		npoints = 200		
		thetas = np.linspace(0,2*np.pi,num=npoints,endpoint = True)
		
	
		vel = np.zeros((npoints,1),dtype = complex)	
		count = 0	
		rad = np.linspace(1.1,2.5,num=25,endpoint = True)
		error = np.zeros_like(rad)
		#print(np.shape(vel),np.shape(error),np.shape(rad))

		for j in range(len(rad)):
			points = lambda th: rad[j]*np.cos(th) + 1j*rad[j]*np.sin(th)
			points = list(map(points,thetas))
			temp = np.zeros_like(vel)
			exact = lambda z: np.conj(V_inf*(1 - (cy_rad**2/z**2)))
			exact = list(map(exact,points))
		
			for i in range(len(points)):
				vel[i] = body.vel_body(points[i]) + V_inf
				#vel[j,i] += [itg.vel_vort(z,elem.pos,elem.strg,elem.kernel) for elem in vort_elems]
				temp[i] = (abs(vel[i] - exact[i]))#/abs(exact[i]))
				#print(temp[i])
			error[j] = abs(np.sum(temp)/len(points))
			#print(error[j])	 
			#print(j)
			count +=1
			
		
	
	
	
		
		plt.plot(rad,error,label = str(nop))
	if linear == True:
		string = ' for Linear Gamma Dist'
		s = 'linear_error.png'
	else:
		string = ' for Const Gamma Dist'
		s = 'const_error.png'	
	plt.title('Error vs radius curve for different no of panels' + string)    		
	plt.legend(loc = 'best', prop = {'size' : 10})		
	plt.savefig('q1' + s)

if __name__ == "__main__":
	Q1(linear = False)
	Q1(linear = True)	
